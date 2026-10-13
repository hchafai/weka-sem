#!/usr/bin/env Rscript

# Rebuilds the public validation fixtures and the independent lavaan reference.
# Run from the repository root with: Rscript validation/lavaan/generate-reference.R

suppressPackageStartupMessages(library(lavaan))

root <- normalizePath(getwd())
data_dir <- file.path(root, "validation", "data")
results_dir <- file.path(root, "validation", "results")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

write_arff <- function(data, relation, path) {
  connection <- file(path, open = "w", encoding = "UTF-8")
  on.exit(close(connection), add = TRUE)
  writeLines(sprintf("@relation %s", relation), connection)
  writeLines("", connection)
  for (name in names(data)) {
    writeLines(sprintf("@attribute %s numeric", name), connection)
  }
  writeLines(c("", "@data"), connection)
  write.table(data, connection, sep = ",", row.names = FALSE,
              col.names = FALSE, quote = FALSE, na = "?")
}

reference_rows <- list()

capture_reference <- function(dataset_id, data, model, arff_name) {
  fit <- sem(model, data = data, estimator = "ML", meanstructure = FALSE,
             fixed.x = FALSE, information = "observed")
  stopifnot(lavInspect(fit, "converged"))

  write_arff(data, dataset_id, file.path(data_dir, arff_name))

  measures <- fitMeasures(
    fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic")
  )
  fit_rows <- data.frame(
    dataset = dataset_id,
    type = "fit",
    name = names(measures),
    lhs = NA_character_, op = NA_character_, rhs = NA_character_,
    estimate = as.numeric(measures), se = NA_real_, z = NA_real_,
    p_value = NA_real_, std_all = NA_real_,
    stringsAsFactors = FALSE
  )

  parameters <- parameterEstimates(fit, standardized = TRUE)
  parameters <- parameters[parameters$op %in% c("=~", "~", "~~"), ]
  parameter_rows <- data.frame(
    dataset = dataset_id,
    type = "parameter",
    name = paste0(parameters$lhs, parameters$op, parameters$rhs),
    lhs = parameters$lhs,
    op = parameters$op,
    rhs = parameters$rhs,
    estimate = parameters$est,
    se = parameters$se,
    z = parameters$z,
    p_value = parameters$pvalue,
    std_all = parameters$std.all,
    stringsAsFactors = FALSE
  )
  reference_rows[[length(reference_rows) + 1L]] <<- rbind(fit_rows, parameter_rows)
}

data("HolzingerSwineford1939", package = "lavaan")
hs_data <- HolzingerSwineford1939[c(paste0("x", 1:9), "ageyr")]
hs_model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
  ageyr    ~ visual + textual + speed
"
capture_reference(
  "holzinger-swineford-1939", hs_data, hs_model,
  "holzinger-swineford-1939.arff"
)

data("PoliticalDemocracy", package = "lavaan")
pd_data <- PoliticalDemocracy[c("x1", "x2", "x3", "y1", "y2", "y3", "y4", "y8")]
pd_model <- "
  ind60 =~ x1 + x2 + x3
  dem60 =~ y1 + y2 + y3 + y4
  y8     ~ ind60 + dem60
"
capture_reference(
  "political-democracy", pd_data, pd_model,
  "political-democracy.arff"
)

reference <- do.call(rbind, reference_rows)
write.csv(reference, file.path(results_dir, "lavaan-reference.csv"),
          row.names = FALSE, na = "")

versions <- data.frame(
  component = c("R", "lavaan"),
  version = c(as.character(getRversion()), as.character(packageVersion("lavaan")))
)
write.csv(versions, file.path(results_dir, "reference-versions.csv"), row.names = FALSE)
