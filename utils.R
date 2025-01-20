################################################################################
#                            Utility functions
################################################################################


# Create survex::explainer -----------------------------------------------------
to_survex <- function(explainer, verbose = FALSE) {
  # Make sure 'survex' package is installed
  if (!requireNamespace("survex", quietly = TRUE)) {
    stop("Please install the 'survex' package")
  }
  
  # Get the model type
  model_type <- explainer$model$.classes[1]
  
  # Get the model and freeze all random components
  model <- explainer$model
  model$eval()
  
  # Get the whole dataset
  dataset <- explainer$data[[1]]
  
  # Get the input data as an array
  data <- dataset[, !(colnames(dataset) %in% c("time", "status")), ]
  
  # Get the outcome
  y <- Surv(time = dataset$time, event = dataset$status)
  
  # Get target labels
  target <- switch(model_type, DeepHit = "pmf", CoxTime = "hazard", DeepSurv = "hazard")
  
  # Get the predict function for the risk score
  # Note: 'newdata' needs to be transformed to a torch tensor to be used for
  # the torch model. After that, the output needs to be transformed back to an
  # array.
  predict_function <- function(model, newdata) {
    input <- torch_tensor(as.matrix(newdata))
    input <- model$preprocess_fun(input)
    out <- model(input, target = target)[[1]]
    out <- out$squeeze(dim = seq_len(out$dim())[-c(1, out$dim())])
    as.array(out)
  }
  
  # Get time points
  if (model_type == "DeepHit") {
    times <- explainer$model$time_bins
  } else {
    times <- explainer$model$t_orig
  }
  
  # Get the predict survival function
  # TODO: Doesn't work for SurvLime
  predict_survival_function <- function(model, newdata, times) {
    input <- torch_tensor(as.matrix(newdata))
    input <- model$preprocess_fun(input)
    out <- model(input, target = "survival")[[1]]
    out <- out$squeeze(dim = seq_len(out$dim())[-c(1, out$dim())])
    as.array(out)
  }
  
  # Get the predict cumulative hazard function
  # TODO: Doesn't work for SurvLime
  target <- switch(model_type, DeepHit = "cif", CoxTime = "cum_hazard", DeepSurv = "cum_hazard")
  predict_cumulative_hazard_function <- function(model, newdata, times) {
    input <- torch_tensor(as.matrix(newdata))
    input <- model$preprocess_fun(input)
    out <- model(input, target = target)[[1]]
    out <- out$squeeze(dim = seq_len(out$dim())[-c(1, out$dim())])
    as.array(out)
  }
  
  # Create survex explainer
  survex::explain_survival(
    model = model,
    data = data,
    verbose = verbose,
    y = y,
    predict_function = predict_function,
    label = model_type,
    times = times,
    predict_survival_function = predict_survival_function,
    predict_cumulative_hazard_function = predict_cumulative_hazard_function
  )
}

# Plot functions ---------------------------------------------------------------
plot_surv_pred <- function(x, model = "DeepSurv") {
  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)

  p <- ggplot(dat, aes(x = .data$time)) + NULL +
    geom_line(aes(
      y = .data$pred,
      group = .data$id,
      color = .data$id,
      linetype = .data$id
    )) +
    scale_color_manual(values = c("skyblue", "orange")) +
    theme_minimal(base_size = 16) +
    theme(legend.position = "bottom",
          plot.margin = margin(25, 5, 5, 5)) +
    labs(
      x = "Time",
      y = "Survival Prediction",
      color = "ID",
      linetype = "ID"
    )
  return(p)
}


plot_attribution <- function(x, normalize = FALSE, normalize_abs = FALSE, add_sum = FALSE,
                             add_pred = FALSE, add_ref = FALSE, add_diff = FALSE) {
  dat <- as.data.frame(x)
  dat$id <- as.factor(dat$id)

  # Normalize values if requested
  if (normalize) {
    dat$value <- ave(
      dat$value,
      dat$id,
      dat$time,
      FUN = function(x)
        x / sum(x)
    )
  }

  # Normalize values using absolute values if requested
  if (normalize_abs) {
    dat$value <- ave(
      dat$value,
      dat$id,
      dat$time,
      FUN = function(x)
        abs(x) / sum(abs(x))
    )
  }

  # Add sum of all attributions if requested
  if (add_sum) {
    dat_sum <- aggregate(as.formula(paste0(
      "value ~ ", paste0(setdiff(colnames(dat), c(
        "feature", "value"
      )), collapse = " + ")
    )), data = dat, sum)
    dat_sum$feature <- "Sum"
    dat <- rbind(dat, dat_sum)

    sum <- geom_line(data = dat_sum, aes(x = time, y = value),
                     linetype = "dashed", color = "#a6611a", linewidth = 1)
  } else {
    sum <- NULL
  }

  if (add_ref) {
    dat$pred_ref <- dat$pred - dat$pred_diff
    ref <- geom_line(data = dat_ref, aes(x = time, y = value),
                     linetype = "dashed", color = "#e66101", linewidth = 1)
  } else {
    ref <- NULL
  }

  if (add_pred) {
    pred <- geom_line(data = dat, aes(x = time, y = pred),
                      linetype = "dashed", color = "#b2abd2", linewidth = 1)
  } else {
    pred <- NULL
  }

  if (add_diff) {
    pred_diff <- geom_line(data = dat, aes(x = time, y = pred_diff),
                           linetype = "dashed", color = "#7fbf7b", linewidth = 1)
  } else {
    pred_diff <- NULL
  }


  p <- ggplot(dat, aes(x = .data$time)) + NULL +
    geom_line(aes(
      y = .data$value,
      group = .data$feature,
      color = .data$feature
    )) +
    geom_point(aes(
      y = .data$value,
      group = .data$feature,
      color = .data$feature
    ),
    size = 0.75) +
    sum +
    ref +
    pred +
    pred_diff +
    geom_hline(yintercept = 0, color = "black") +
    facet_wrap(vars(.data$id),
               scales = "free_x",
               labeller = as_labeller(function(a)
                 paste0("Instance ID: ", a))) +
    theme_minimal(base_size = 16) +
    theme(legend.position = "bottom") +
    labs(
      x = "Time",
      y = paste0("Attribution: ", unique(dat$method)),
      color = "Feature",
      linetype = NULL
    ) +
    scale_color_viridis_d()
  return(p)
}
