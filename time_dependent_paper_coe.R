# figure path
fig_path <- here::here("figures_paper")
if (!file.exists(fig_path))
  dir.create(fig_path)
fig <- function(x)
  here::here(fig_path, x)

test[c(3,12,22),]

### Gradient (Sensitivity)

grad_cox <- surv_grad(exp_coxtime, target = "survival", instance = c(3,12,22))
grad_deephit <- surv_grad(exp_deephit, target = "survival", instance = c(3,12,22))
grad_deepsurv <- surv_grad(exp_deepsurv, target = "survival", instance = c(3,12,22))

grad_plot <- cowplot::plot_grid(
  plot(grad_cox) + ylim(-0.25, 0.2),
  plot(grad_deephit) + ylim(-0.05, 0.05),
  plot(grad_deepsurv) + ylim(-0.2, 0.2),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
grad_plot

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/grad_td.pdf",
  plot = grad_plot,
  width = 14,
  height = 10,
  device = "pdf"
)

### SmoothGrad (Sensitivity)

sg_cox <- surv_smoothgrad(exp_coxtime, target = "survival", instance = c(3,12,22), n = 50, noise_level = 0.4)
sg_deephit <- surv_smoothgrad(exp_deephit, target = "survival", instance = c(3,12,22), n = 50, noise_level = 0.4)
sg_deepsurv <- surv_smoothgrad(exp_deepsurv, target = "survival", instance = c(3,12,22), n = 50, noise_level = 0.4)

sg_td_surv <- cowplot::plot_grid(
  plot(sg_cox) + ylim(-0.15, 0.1),
  plot(sg_deephit) + ylim(-0.05, 0.05),
  plot(sg_deepsurv) + ylim(-0.15, 0.1),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
sg_td_surv

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/sg_td.pdf",
  plot = sg_td_surv,
  width = 14,
  height = 10,
  device = "pdf"
)

### Gradient x Input

grad_cox <- surv_grad(exp_coxtime, instance = c(3,12,22), times_input = TRUE)
grad_deephit <- surv_grad(exp_deephit, instance = c(3,12,22), times_input = TRUE)
grad_deepsurv <- surv_grad(exp_deepsurv, instance = c(3,12,22), times_input = TRUE)

grad_plot_in_surv <- cowplot::plot_grid(
  plot(grad_cox) + ylim(-0.1, 0.15),
  plot(grad_deephit) + ylim(-0.05, 0.05),
  plot(grad_deepsurv) + ylim(-0.15, 0.1),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
grad_plot_in_surv

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/grad_in_td.pdf",
  plot = grad_plot_in_surv,
  width = 14,
  height = 10,
  device = "pdf"
)


### SmoothGrad x Input

sg_cox_in_surv <- surv_smoothgrad(exp_coxtime, instance = c(3,12,22), n = 50, noise_level = 0.3,
                                  times_input = TRUE)
sg_deephit_in_surv <- surv_smoothgrad(exp_deephit, instance = c(3,12,22), n = 50, noise_level = 0.3,
                                      times_input = TRUE)
sg_deepsurv_in_surv <- surv_smoothgrad(exp_deepsurv, instance = c(3,12,22), n = 50, noise_level = 0.3,
                                       times_input = TRUE)

sg_grad_plot_in_surv <- cowplot::plot_grid(
  plot(sg_cox_in_surv) + ylim(-0.1, 0.15),
  plot(sg_deephit_in_surv) + ylim(-0.025, 0.05),
  plot(sg_deepsurv_in_surv) + ylim(-0.1, 0.1),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
sg_grad_plot_in_surv

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/sg_in_td.pdf",
  plot = sg_grad_plot_in_surv,
  width = 14,
  height = 10,
  device = "pdf"
)

### IntegratedGradient

### Zero baseline (should be proportional to Gradient x Input)

x_ref <- matrix(c(0,0,0), nrow = 1)
ig_cox_0 <- surv_intgrad(exp_coxtime, instance = c(3,12,22), n = 50, x_ref = x_ref)
ig_deephit_0 <- surv_intgrad(exp_deephit, instance = c(3,12,22), n = 50, x_ref = x_ref)
ig_deepsurv_0 <- surv_intgrad(exp_deepsurv, instance = c(3,12,22), n = 50, x_ref = x_ref)

ig_plot_0 <- cowplot::plot_grid(
  plot(ig_cox_0) + ylim(-0.15, 0.1),
  plot(ig_deephit_0) + ylim(-0.05, 0.05),
  plot(ig_deepsurv_0) + ylim(-0.15, 0.1),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
ig_plot_0

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/ig_0_td.pdf",
  plot = ig_plot_0,
  width = 14,
  height = 10,
  device = "pdf"
)

### Mean baseline

x_ref <- NULL # default: feature-wise mean
ig_cox_mean <- surv_intgrad(exp_coxtime, instance = c(3,12,22), n = 50, x_ref = x_ref)
ig_deephit_mean <- surv_intgrad(exp_deephit, instance = c(3,12,22), n = 50, x_ref = x_ref)
ig_deepsurv_mean <- surv_intgrad(exp_deepsurv, instance = c(3,12,22), n = 50, x_ref = x_ref)

ig_plot_mean <-cowplot::plot_grid(
  plot(ig_cox_mean) + ylim(-0.15, 0.1),
  plot(ig_deephit_mean) + ylim(-0.05, 0.05),
  plot(ig_deepsurv_mean) + ylim(-0.15, 0.1),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
ig_plot_mean

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/ig_mean_td.pdf",
  plot = ig_plot_mean,
  width = 14,
  height = 10,
  device = "pdf"
)


### GradShap

gshap_cox <- surv_gradSHAP(exp_coxtime, instance = c(3,12,22), n = 50, num_samples = 100,
                           batch_size = 1000)
gshap_deephit <- surv_gradSHAP(exp_deephit, instance = c(3,12,22), n = 50, num_samples = 100,
                               batch_size = 1000)
gshap_deepsurv <- surv_gradSHAP(exp_deepsurv, instance = c(3,12,22), n = 50, num_samples = 100,
                                batch_size = 1000)

gshap_plot <-cowplot::plot_grid(
  plot(gshap_cox) + ylim(-0.15, 0.1),
  plot(gshap_deephit) + ylim(-0.05, 0.05),
  plot(gshap_deepsurv) + ylim(-0.15, 0.1),
  nrow = 3, labels = c("CoxTime", "DeepHit", "DeepSurv"))
gshap_plot

ggsave(
  "/Users/sophielangbein/Desktop/PhD/Survinng/plots_paper/gshap_td.pdf",
  plot = gshap_plot,
  width = 14,
  height = 10,
  device = "pdf"
)
