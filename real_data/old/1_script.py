import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
from torchvision.transforms import v2
from skimage import io
import torchtuples as tt
import matplotlib.pyplot as plt

if torch.cuda.is_available(): 
 dev = "cuda:1" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

torch.set_num_threads(5)

# Load data and crate dataset ------------------------------------------------------------------------------------------
# Global variables
TAB_FEATURES = ['Age (at index)', 'gender', 'idh.mutation'] #'ethnicity'
OUT_COLUMNS = ['censored', 'Survival.months']
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 128
NUM_WORKERS = 0

image_dir = 'data/' #'/opt/example-data/TCGA-SCNN/'
tabular_file = '/opt/example-data/TCGA-SCNN/all_data_custom.csv'

# Read the tabular data
tabular_data = pd.read_csv(tabular_file)

# Use binary encoding for column 'gender'
tabular_data['gender']  = (tabular_data['gender'] == 'male').astype(int)
tabular_data['Survival.months'] = tabular_data['Survival.months'] / 365
tabular_data['censored'] = tabular_data['censored'].astype(float)
tabular_data['idh.mutation'] = tabular_data['idh.mutation'].fillna(-1)
tabular_data['Age (at index)'] = tabular_data['Age (at index)'] / 100
#tabular_data[tabular_data['censored'] == 0]['censored'] = 1e-6

# Create a torch dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tabular_data, image_dir, transform=None, device = torch.device('cpu')):
        self.tabular_data = tabular_data
        self.image_dir = image_dir
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):

        # Get the images
        img_name = os.path.join(self.image_dir, self.tabular_data['full_path'].iloc[idx])
        image = torch.tensor(io.imread(img_name)).to(self.device) / 255
        image = image[:,:,:3]

        # Make channels first
        image = image.permute(2, 0, 1)

        # Transform the images (if required)
        if self.transform:
            image = self.transform(image)

        # Get the tabular data
        tabular = torch.tensor(self.tabular_data[TAB_FEATURES].iloc[idx].to_numpy(dtype=np.float32), device=self.device)

        # Get outcome (censored and survival time)
        event = np.array(self.tabular_data['censored'].iloc[idx], dtype=np.float32)
        time = np.array(self.tabular_data['Survival.months'].iloc[idx])


        return [image , tabular], tt.tuplefy(time, event).to_tensor()

# Define the transformations
transforms = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.ColorJitter(brightness=.2, hue=.1),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(size=IMAGE_SIZE),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_test = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.RandomCrop(size=IMAGE_SIZE),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create train and test dataset
train_ds = CustomDataset(tabular_data[tabular_data['train']], image_dir + 'train/', transform = transforms, device = device)
test_ds = CustomDataset(tabular_data[~tabular_data['train']], image_dir + 'test/', transform = transforms_test, device = device)

print('\n----Dataset information----')
print('Train dataset size:', len(train_ds))
print('Test dataset size:', len(test_ds))


# Check the dataloader
[image, tab], outcome = train_ds[1]

print('\n----Data shape----')
print('Image shape:', image.shape)
print('Tabular shape:', tab.shape)
print('Outcome shape:', outcome[0].shape, outcome[1].shape)

def collate_fn(batch):
    """Stacks the entries of a nested tuple"""
    return tt.tuplefy(batch).stack()

dl_train = torch.utils.data.DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
dl_test = torch.utils.data.DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

batch = next(iter(dl_train))
print('\n----Batch shape----')
print(batch.shapes())


# Define the model -----------------------------------------------------------------------------------------------------

# Use a pretrained ResNet model for the image data
net_images = torchvision.models.resnet18(num_classes = 50).to(device) #(weights = 'IMAGENET1K_V1').to(device)
print(batch.shapes())
print(batch.dtypes())
print(net_images)
print(net_images(batch[0][0]).shape)


# Define multi modal model
class MultiModalModel(nn.Module):
    def __init__(self, net_images, tabular_features, n_out = 1):
        super(MultiModalModel, self).__init__()
        self.net_images = net_images
        self.tabular_features = tabular_features
        self.n_out = n_out

        self.fc1 = nn.Linear(50 + len(tabular_features), 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.drop_0_3 = nn.Dropout(0.2)
        self.drop_0_5 = nn.Dropout(0.4)
        self.out = nn.Linear(64, n_out, bias = False)

    def forward(self, img, tab):
        img = self.net_images(img)
        img = self.drop_0_5(img)

        x = torch.cat((img, tab), dim=1)
        x = self.drop_0_3(x)
        x = self.relu(self.fc1(x))
        x = self.drop_0_3(x)
        x = self.relu(self.fc2(x))
        x = self.drop_0_3(x)
        x = self.out(x)

        return x

my_model = MultiModalModel(net_images, TAB_FEATURES).to(device)


import pycox
from pycox.models import CoxPH
from pycox.utils import kaplan_meier
from pycox.evaluation import EvalSurv
import torchtuples as tt

model = CoxPH(my_model, tt.optim.Adam(0.005, amsgrad=True, weight_decay = 1e-5), device = device)

print(batch[0].shapes())
pred = model.predict(batch[0])
print(pred.shape)

# Warm-up rounds
callbacks = None #[tt.cb.EarlyStopping(patience=5)]
epochs = 10
verbose = True
log = model.fit_dataloader(dl_train, epochs, callbacks, verbose = verbose, val_dataloader=dl_test)
res = log.plot()
plt.show()
plt.savefig('foo.png')

# Fine-tuning
callbacks = [tt.cb.EarlyStopping(patience=10)]
model.optimizer = tt.optim.Adam(0.001, amsgrad=True, weight_decay = 1e-4)
epochs = 10
verbose = True
log = model.fit_dataloader(dl_train, epochs, callbacks, verbose = verbose, val_dataloader=dl_test)
res = log.plot()
plt.show()
plt.savefig('foo.png')





dl_eval = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
inputs, target = next(iter(dl_eval))
print(inputs.shapes())
_ = model.compute_baseline_hazards(input = inputs, target = target)

surv = model.predict_surv_df(input = inputs)
surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
plt.show()
plt.savefig('foo2.png')

ev = EvalSurv(surv, target[0].cpu().detach().numpy(), target[1].cpu().detach().numpy(), censor_surv='km')
print(ev.concordance_td())


plt.figure(1)
time_grid = np.linspace(0, 14, 100)
_ = ev.brier_score(time_grid).plot()
plt.show()
plt.savefig('foo3.png')