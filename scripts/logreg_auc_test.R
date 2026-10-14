#!/usr/bin/env Rscript
#
# Paired AUC test on precomputed predictions.
#
# Compares two sets of predicted probabilities on the same observations:
#   p_full:  predictions from the full model (e.g. Bayesian ensemble over
#            concepts + force-keep features)
#   p_restr: predictions from a restricted/baseline model (e.g. logistic
#            regression trained on force-keep features only)
# via DeLong's test for paired ROC curves (pROC::roc.test, method="delong").
# When --cluster-col is passed, also runs a cluster bootstrap on the AUC
# difference by resampling (y, p_full, p_restr) triples at the cluster level
# (DeLong's variance assumes i.i.d. observations, which is violated under
# clustering).
#
# Input CSV: columns y, p_full, p_restr, and optionally the cluster column.
#
# Output CSV is a single row with columns:
#   test, auc_restricted, auc_full, auc_diff, delong_statistic,
#   delong_pvalue, boot_ci_low, boot_ci_high, boot_pvalue, n_obs,
#   n_clusters, n_boot
#
# Usage:
#   Rscript logreg_auc_test.R --in <csv> --out <csv> \
#       [--cluster-col <name>] [--n-boot 1000] [--seed 0]

suppressPackageStartupMessages({
  library(pROC)
})

parse_args <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  args <- list(cluster_col = NA_character_, n_boot = 1000L, seed = 0L)
  i <- 1
  while (i <= length(argv)) {
    key <- argv[i]
    val <- if (i + 1 <= length(argv)) argv[i + 1] else NA
    if (key == "--in") { args$in_csv <- val; i <- i + 2 }
    else if (key == "--out") { args$out_csv <- val; i <- i + 2 }
    else if (key == "--cluster-col") { args$cluster_col <- val; i <- i + 2 }
    else if (key == "--n-boot") { args$n_boot <- as.integer(val); i <- i + 2 }
    else if (key == "--seed") { args$seed <- as.integer(val); i <- i + 2 }
    else stop(sprintf("Unknown argument: %s", key))
  }
  stopifnot(!is.null(args$in_csv), !is.null(args$out_csv))
  args
}

args <- parse_args()
set.seed(args$seed)
dat <- read.csv(args$in_csv, stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(all(c("y", "p_full", "p_restr") %in% names(dat)))

cluster_vec <- NULL
if (!is.na(args$cluster_col) && nzchar(args$cluster_col)) {
  stopifnot(args$cluster_col %in% names(dat))
  cluster_vec <- dat[[args$cluster_col]]
}

row <- data.frame(
  test = "paired_auc",
  auc_restricted = NA_real_,
  auc_full = NA_real_,
  auc_diff = NA_real_,
  delong_statistic = NA_real_,
  delong_pvalue = NA_real_,
  boot_ci_low = NA_real_,
  boot_ci_high = NA_real_,
  boot_pvalue = NA_real_,
  n_obs = nrow(dat),
  n_clusters = if (is.null(cluster_vec)) NA_integer_
               else length(unique(cluster_vec)),
  n_boot = if (is.null(cluster_vec)) NA_integer_ else args$n_boot,
  stringsAsFactors = FALSE
)

roc_full <- pROC::roc(dat$y, dat$p_full, quiet = TRUE, direction = "<")
roc_restr <- pROC::roc(dat$y, dat$p_restr, quiet = TRUE, direction = "<")
row$auc_full <- as.numeric(pROC::auc(roc_full))
row$auc_restricted <- as.numeric(pROC::auc(roc_restr))
row$auc_diff <- row$auc_full - row$auc_restricted

dl <- tryCatch(
  pROC::roc.test(roc_full, roc_restr, method = "delong", paired = TRUE),
  error = function(e) {
    message("DeLong test failed: ", conditionMessage(e)); NULL
  }
)
if (!is.null(dl)) {
  row$delong_statistic <- as.numeric(dl$statistic)
  row$delong_pvalue <- as.numeric(dl$p.value)
}

# Cluster bootstrap on the AUC difference: resample clusters of
# (y, p_full, p_restr) triples with replacement and recompute AUC difference.
# Yields a cluster-robust CI and a two-sided p-value from the bootstrap
# distribution recentered at zero.
if (!is.null(cluster_vec)) {
  clusters <- unique(cluster_vec)
  idx_by_cluster <- split(seq_len(nrow(dat)), cluster_vec)
  diffs <- numeric(args$n_boot)
  n_valid <- 0
  for (b in seq_len(args$n_boot)) {
    samp_clusters <- sample(clusters, length(clusters), replace = TRUE)
    samp_idx <- unlist(idx_by_cluster[as.character(samp_clusters)],
                       use.names = FALSE)
    d_b <- dat[samp_idx, , drop = FALSE]
    if (length(unique(d_b$y)) < 2) next
    a_f <- tryCatch(
      as.numeric(pROC::auc(pROC::roc(d_b$y, d_b$p_full, quiet = TRUE,
                                      direction = "<"))),
      error = function(e) NA_real_
    )
    a_r <- tryCatch(
      as.numeric(pROC::auc(pROC::roc(d_b$y, d_b$p_restr, quiet = TRUE,
                                      direction = "<"))),
      error = function(e) NA_real_
    )
    if (is.na(a_f) || is.na(a_r)) next
    n_valid <- n_valid + 1
    diffs[n_valid] <- a_f - a_r
  }
  diffs <- diffs[seq_len(n_valid)]
  if (n_valid > 10) {
    row$boot_ci_low <- quantile(diffs, 0.025, names = FALSE)
    row$boot_ci_high <- quantile(diffs, 0.975, names = FALSE)
    centered <- diffs - mean(diffs)
    row$boot_pvalue <- mean(abs(centered) >= abs(row$auc_diff))
    row$n_boot <- n_valid
  }
}

write.csv(row, args$out_csv, row.names = FALSE)
