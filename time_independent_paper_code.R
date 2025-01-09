# figure path
fig_path <- here::here("figures_paper")
if (!file.exists(fig_path))
  dir.create(fig_path)
fig <- function(x)
  here::here(fig_path, x)

### Gradient (Sensitivity)

## survival
grad_cox <- surv_grad(exp_coxtime, target = "survival", instance = c(1,2,3))
grad_deephit <- surv_grad(exp_deephit, target = "survival", instance = c(1,2,3))
grad_deepsurv <- surv_grad(exp_deepsurv, target = "survival", instance = c(1,2,3))

grad_plot <- cowplot::plot_grid(
  plot(grad_cox),
  plot(grad_deephit),
  plot(grad_deepsurv),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
grad_plot

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/grad_ntd_surv.pdf",
  plot = grad_plot,
  width = 14,
  height = 10,
  device = "pdf"
)

## hazard
grad_cox_haz <- surv_grad(exp_coxtime, target = "hazard", instance = c(1,2,3))
#grad_deephit <- surv_grad(exp_deephit, target = "hazard", instance = c(1,2))
grad_deepsurv_haz <- surv_grad(exp_deepsurv, target = "hazard", instance = c(1,2,3))

grad_plot_haz <- cowplot::plot_grid(
  plot(grad_cox_haz),
  plot(grad_deepsurv_haz),
  nrow = 2, labels = c("CoxTime", "DeepSurv"))
grad_plot_haz

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/grad_ntd_haz.pdf",
  plot = grad_plot_haz,
  width = 14,
  height = 7,
  device = "pdf"
)


### SmoothGrad (Sensitivity)

## survival
sg_cox <- surv_smoothgrad(exp_coxtime, target = "survival", instance = c(1,2,3), n = 50, noise_level = 0.4)
sg_deephit <- surv_smoothgrad(exp_deephit, target = "survival", instance = c(1,2,3), n = 50, noise_level = 0.4)
sg_deepsurv <- surv_smoothgrad(exp_deepsurv, target = "survival", instance = c(1,2,3), n = 50, noise_level = 0.4)

sg_grad_plot <- cowplot::plot_grid(
  plot(sg_cox),
  plot(sg_deephit),
  plot(sg_deepsurv),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
sg_grad_plot

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/sg_grad_plot.pdf",
  plot = sg_grad_plot,
  width = 14,
  height = 10,
  device = "pdf"
)


### Gradient x Input

## survival
library(ggplot2)
library(data.table)

df <- melt(data.table(train), id.vars = c("time", "status"))
df_inst <- melt(data.table(test, id = paste0(rownames(train), " (row: " , seq_len(nrow(train)), ")"))[c(1,2), ],
                id.vars = c("time", "status", "id"))
ggplot(data = df, aes(x = value)) +
  geom_point(data = df_inst, mapping = aes(x = value, y = 0, color = id)) +
  geom_density(alpha = 0.5, fill = "gray75") +
  theme_minimal() +
  facet_grid(cols = vars(variable), scales = "free")

grad_cox_in_surv <- surv_grad(exp_coxtime, instance = c(1,2,3), times_input = TRUE)
grad_deephit_in_surv <- surv_grad(exp_deephit, instance = c(1,2,3), times_input = TRUE)
grad_deepsurv_in_surv <- surv_grad(exp_deepsurv, instance = c(1,2,3), times_input = TRUE)

grad_plot_in_surv <- cowplot::plot_grid(
  plot(grad_cox_in_surv),
  plot(grad_deephit_in_surv),
  plot(grad_deepsurv_in_surv),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
grad_plot_in_surv

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/grad_plot_in_surv.pdf",
  plot = grad_plot_in_surv,
  width = 14,
  height = 10,
  device = "pdf"
)

## hazard

grad_cox_in_haz <- surv_grad(exp_coxtime, instance = c(1,2), times_input = TRUE, target = "hazard")
grad_deepsurv_in_haz <- surv_grad(exp_deepsurv, instance = c(1,2), times_input = TRUE, target = "hazard")

grad_plot_in_haz <- cowplot::plot_grid(
  plot(grad_cox),
  plot(grad_deepsurv),
  nrow = 2, labels = c("CoxTime", "DeepSurv"))
grad_plot_in_haz

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/grad_plot_in_haz.pdf",
  plot = grad_plot_in_haz,
  width = 14,
  height = 10,
  device = "pdf"
)

### SmoothGrad x Input

## survival
sg_cox_in_surv <- surv_smoothgrad(exp_coxtime, instance = c(1,2,3), n = 50, noise_level = 0.3,
                          times_input = TRUE)
sg_deephit_in_surv <- surv_smoothgrad(exp_deephit, instance = c(1,2,3), n = 50, noise_level = 0.3,
                              times_input = TRUE)
sg_deepsurv_in_surv <- surv_smoothgrad(exp_deepsurv, instance = c(1,2,3), n = 50, noise_level = 0.3,
                               times_input = TRUE)

sg_grad_plot_in_surv <- cowplot::plot_grid(
  plot(sg_cox_in_surv),
  plot(sg_deephit_in_surv),
  plot(sg_deepsurv_in_surv),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
sg_grad_plot_in_surv

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/sg_grad_plot_in_surv.pdf",
  plot = sg_grad_plot_in_surv,
  width = 14,
  height = 10,
  device = "pdf"
)


### IntegratedGradient

#### survival
### Zero baseline (should be proportional to Gradient x Input)

x_ref <- matrix(c(0,0,0), nrow = 1)
ig_cox_0 <- surv_intgrad(exp_coxtime, instance = c(1,2,3), n = 50, x_ref = x_ref)
ig_deephit_0 <- surv_intgrad(exp_deephit, instance = c(1,2,3), n = 50, x_ref = x_ref)
ig_deepsurv_0 <- surv_intgrad(exp_deepsurv, instance = c(1,2,3), n = 50, x_ref = x_ref)

ig_plot_0 <- cowplot::plot_grid(
  plot(ig_cox_0),
  plot(ig_deephit_0),
  plot(ig_deepsurv_0),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
ig_plot_0

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/ig_grad_plot_0.pdf",
  plot = ig_grad_plot_0,
  width = 14,
  height = 10,
  device = "pdf"
)

### Mean baseline

x_ref <- NULL # default: feature-wise mean
ig_cox_mean <- surv_intgrad(exp_coxtime, instance = c(1,2,3), n = 50, x_ref = x_ref)
ig_deephit_mean <- surv_intgrad(exp_deephit, instance = c(1,2,3), n = 50, x_ref = x_ref)
ig_deepsurv_mean <- surv_intgrad(exp_deepsurv, instance = c(1,2,3), n = 50, x_ref = x_ref)

ig_plot_mean <-cowplot::plot_grid(
  plot(ig_cox_mean),
  plot(ig_deephit_mean),
  plot(ig_deepsurv_mean),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
ig_plot_mean

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/ig_plot_0.pdf",
  plot = ig_plot_mean,
  width = 14,
  height = 10,
  device = "pdf"
)


### GradShap

gshap_cox <- surv_gradSHAP(exp_coxtime, instance = c(1,2,3), n = 50, num_samples = 100,
                           batch_size = 1000)
gshap_deephit <- surv_gradSHAP(exp_deephit, instance = c(1,2,3), n = 50, num_samples = 100,
                               batch_size = 1000)
gshap_deepsurv <- surv_gradSHAP(exp_deepsurv, instance = c(1,2,3), n = 50, num_samples = 100,
                                batch_size = 1000)

gshap_plot <-cowplot::plot_grid(
  plot(gshap_cox),
  plot(gshap_deephit),
  plot(gshap_deepsurv),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
gshap_plot

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/gshap_plot.pdf",
  plot = gshap_plot,
  width = 14,
  height = 10,
  device = "pdf"
)
