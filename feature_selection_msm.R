#!/usr/bin/env Rscript
# ══════════════════════════════════════════════════════════════
# MSM-based Feature Selection → table2.1
# ══════════════════════════════════════════════════════════════
# Fits univariate continuous-time multi-state Markov models
# (one feature at a time) using the msm package.
#
# States used in modeling: 0, 1, 2
# Starting states included: 0, 2
# Observed transitions allowed by data definition:
#   0 -> {0,1,2}, 2 -> {0,1,2}
# (In continuous-time msm, self-transitions are represented implicitly
#  by waiting time; off-diagonal intensities are estimated directly.)
#
# Strategy:
#   1) Build panel data ONCE from full data
#   2) Fit ONE baseline model (no covariates) → stable Q
#   3) For each feature, fit msm with that covariate
#      (msm handles NA covariates by listwise deletion)
#   4) Extract HR / P → filter significant features
#
# Usage:
#   Rscript feature_selection_msm.R <data.csv> <out_dir>
# ══════════════════════════════════════════════════════════════

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript feature_selection_msm.R <data.csv> <out_dir>")
}

DATA_PATH <- args[1]
OUT_DIR   <- args[2]

# ── Parameters ──
MAXIT          <- 200
PARSCALE_Q     <- 0.1
PARSCALE_COV   <- 0.1
TIME_LIMIT_SEC <- 600
MAX_FACTOR_LEV <- 20

# Columns excluded from feature screening (user-defined)
BASE_NON_FEATURE_COLS <- c(
  "id", "gryhb_c", "t_start", "t_stop", "wave_exam", "time",
  "birthday_updated"
)
EVENT_COLS <- c("state_start", "state_stop")
EXCLUDE_COLS <- c(BASE_NON_FEATURE_COLS, EVENT_COLS)

# ── Helpers ──
coerce_feature_column <- function(x) {
  if (is.character(x)) {
    x_trim <- trimws(x)
    suppressWarnings(x_num <- as.numeric(x_trim))
    if (mean(!is.na(x_num)) >= 0.9) return(x_num)
    return(as.factor(x_trim))
  }
  if (is.logical(x)) return(as.integer(x))
  x
}

parse_date_safe <- function(x) {
  if (inherits(x, "Date")) return(x)
  if (is.numeric(x)) x_chr <- sprintf("%.0f", x)
  else x_chr <- as.character(x)
  x_chr <- trimws(x_chr)
  x_chr[x_chr == ""] <- NA_character_
  has_time <- !is.na(x_chr) & nchar(x_chr) >= 10
  x_chr[has_time] <- substr(x_chr[has_time], 1, 10)
  fmts <- c("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d")
  out <- rep(as.Date(NA), length(x_chr))
  for (f in fmts) {
    idx <- is.na(out) & !is.na(x_chr)
    if (!any(idx)) break
    parsed <- as.Date(x_chr[idx], format = f)
    pos <- which(idx)
    ok <- !is.na(parsed)
    if (any(ok)) out[pos[ok]] <- parsed[ok]
  }
  out
}

extract_cov_results <- function(fit, term_names, state_levels,
                                feature_name, n_ids, n_obs) {
  n_q <- fit$qmodel$npars
  trans_idx <- which(fit$qmodel$imatrix == 1, arr.ind = TRUE)
  trans_from  <- state_levels[trans_idx[, 1]]
  trans_to    <- state_levels[trans_idx[, 2]]
  trans_label <- paste0(trans_from, "->", trans_to)

  if (length(fit$opt$par) <= n_q) return(NULL)
  beta_all <- fit$opt$par[(n_q + 1):length(fit$opt$par)]

  se_all <- rep(NA_real_, length(beta_all))
  if (!is.null(fit$covmat)) {
    se_vec <- sqrt(diag(fit$covmat))
    if (length(se_vec) >= n_q + length(beta_all)) {
      se_all <- se_vec[(n_q + 1):(n_q + length(beta_all))]
    }
  }

  res <- list()
  k <- 1
  for (t in seq_along(term_names)) {
    for (j in seq_len(n_q)) {
      idx_b <- (t - 1) * n_q + j
      if (idx_b > length(beta_all)) next
      beta <- beta_all[idx_b]
      se   <- se_all[idx_b]
      z    <- if (!is.na(se) && se > 0) beta / se else NA_real_
      p    <- if (!is.na(z)) 2 * pnorm(-abs(z)) else NA_real_
      hr   <- exp(beta)

      res[[k]] <- data.frame(
        feature    = feature_name,
        term       = term_names[t],
        from_state = trans_from[j],
        to_state   = trans_to[j],
        transition = trans_label[j],
        beta       = beta,
        se         = se,
        z          = z,
        p          = p,
        hr         = hr,
        n_ids      = n_ids,
        n_obs      = n_obs,
        converged  = isTRUE(fit$opt$convergence == 0),
        stringsAsFactors = FALSE
      )
      k <- k + 1
    }
  }
  if (length(res) == 0) return(NULL)
  do.call(rbind, res)
}

# ── Load msm ──
if (!requireNamespace("msm", quietly = TRUE)) {
  install.packages("msm", repos = "https://cloud.r-project.org")
}
suppressMessages(library(msm))

# ══════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════
cat("Loading data:", DATA_PATH, "\n")
df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
cat(sprintf("  Loaded: %d rows × %d cols\n", nrow(df), ncol(df)))
df$state_start <- suppressWarnings(as.numeric(df$state_start))
df$state_stop  <- suppressWarnings(as.numeric(df$state_stop))

required_cols <- c("id", "state_start", "state_stop")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns: %s",
               paste(missing_cols, collapse = ", ")))
}

# Feature set definition from input columns only
feature_cols_input <- setdiff(names(df), EXCLUDE_COLS)
cat(sprintf("  Feature columns by definition: %d\n", length(feature_cols_input)))
if (length(feature_cols_input) != 91) {
  cat(sprintf("  ⚠ Expected 91 features by definition, found %d\n",
              length(feature_cols_input)))
}

# Parse dates
if ("t_start" %in% names(df)) df$t_start <- parse_date_safe(df$t_start)
if ("t_stop"  %in% names(df)) df$t_stop  <- parse_date_safe(df$t_stop)
if ("birthday_updated" %in% names(df))
  df$birthday_updated <- parse_date_safe(df$birthday_updated)

# Compute interval time in years
raw_time  <- suppressWarnings(as.numeric(df$time))
date_time <- suppressWarnings(as.numeric(df$t_stop - df$t_start)) / 365.25
time_years <- raw_time
bad_time <- is.na(time_years) | time_years <= 0
if (any(bad_time, na.rm = TRUE)) time_years[bad_time] <- date_time[bad_time]
df$time_years <- time_years

# Filter: valid intervals & starting states {0, 2}
valid <- !is.na(df$time_years) & df$time_years > 0 &
         !is.na(df$state_start) & !is.na(df$state_stop) &
         df$state_start %in% c(0, 2) &
         df$state_stop %in% c(0, 1, 2)
df <- df[valid, ]
df <- df[order(df$id, df$t_start), ]
cat(sprintf("  Valid rows (state_start ∈ {0,2}): %d\n", nrow(df)))

# ══════════════════════════════════════════════════════════════
# Build panel data ONCE
# ══════════════════════════════════════════════════════════════
state_levels <- c(0, 1, 2)
state_map    <- setNames(1:3, state_levels)

df$state_start_m <- state_map[as.character(df$state_start)]
df$state_stop_m  <- state_map[as.character(df$state_stop)]

cumt      <- ave(df$time_years, df$id, FUN = cumsum)
first_idx <- !duplicated(df$id)

base <- df[first_idx, ]
base$time_msm <- 0
base$state    <- base$state_start_m

follow <- df
follow$time_msm <- cumt
follow$state    <- follow$state_stop_m

dat <- rbind(base, follow)
dat <- dat[!is.na(dat$state) & !is.na(dat$time_msm), ]
dat <- dat[order(dat$id, dat$time_msm), ]

# Remove duplicate (id, time_msm) — keep last observation per time point
dat$dup_key <- paste0(dat$id, "_", sprintf("%.10f", dat$time_msm))
dat <- dat[!duplicated(dat$dup_key, fromLast = TRUE), ]
dat$dup_key <- NULL
dat <- dat[order(dat$id, dat$time_msm), ]

# Expected transition structure in msm state index:
# 1 <-> 3 and 1 -> 2, 3 -> 2 ; state 2 is absorbing.
Q_struct <- matrix(c(
  0, 1, 1,
  0, 0, 0,
  1, 1, 0
), nrow = 3, byrow = TRUE)

sanitize_subject_path <- function(sub_dat, q_struct, absorbing_state = 2) {
  if (nrow(sub_dat) <= 1) return(sub_dat)

  # If absorbing state appears, keep only up to first hit.
  hit_absorb <- which(sub_dat$state == absorbing_state)
  if (length(hit_absorb) > 0) {
    sub_dat <- sub_dat[seq_len(hit_absorb[1]), , drop = FALSE]
  }
  if (nrow(sub_dat) <= 1) return(sub_dat)

  # Remove tail after first illegal jump.
  keep_n <- nrow(sub_dat)
  for (i in seq_len(nrow(sub_dat) - 1)) {
    from_s <- sub_dat$state[i]
    to_s   <- sub_dat$state[i + 1]
    legal <- (from_s == to_s) || (q_struct[from_s, to_s] == 1)
    if (!legal) {
      keep_n <- i
      break
    }
  }
  sub_dat[seq_len(keep_n), , drop = FALSE]
}

sub_paths <- split(dat, dat$id)
dat <- do.call(
  rbind,
  lapply(sub_paths, sanitize_subject_path,
         q_struct = Q_struct, absorbing_state = 2)
)
dat <- dat[order(dat$id, dat$time_msm), ]

# Remove subjects with only 1 observation (need >=2 for msm)
obs_per_subj <- table(dat$id)
keep_ids <- names(obs_per_subj[obs_per_subj >= 2])
dat <- dat[dat$id %in% keep_ids, ]

cat(sprintf("  Panel data: %d observations, %d subjects\n",
            nrow(dat), length(unique(dat$id))))
cat(sprintf("  States observed: %s\n",
            paste(sort(unique(dat$state)), collapse = ", ")))
# Print observed transition table by event definition
tab <- table(from = df$state_start, to = df$state_stop)
cat("  Observed transitions (state_start -> state_stop):\n")
print(tab)

# ══════════════════════════════════════════════════════════════
# Fit baseline model (no covariates) ONCE
# ══════════════════════════════════════════════════════════════
cat("\n  Fitting baseline MSM (no covariates)...\n")

Q_init <- matrix(c(
  0,    0.1,  0.1,
  0,    0,    0,
  0.1,  0.1,  0
), nrow = 3, byrow = TRUE)

Qcrude <- tryCatch(
  crudeinits.msm(state ~ time_msm, subject = id,
                 data = dat, qmatrix = Q_init),
  error = function(e) Q_init
)
# Enforce structure
Qcrude[!is.finite(Qcrude)] <- 0
Qcrude[Qcrude < 0] <- abs(Qcrude[Qcrude < 0])
Qcrude[2, ] <- 0  # state 1 absorbing
for (rc in list(c(1,2), c(1,3), c(3,1), c(3,2))) {
  r <- rc[1]; cc <- rc[2]
  if (Qcrude[r, cc] <= 0 || !is.finite(Qcrude[r, cc])) {
    Qcrude[r, cc] <- 0.01
  }
  Qcrude[r, cc] <- min(Qcrude[r, cc], 5)
}
diag(Qcrude) <- 0

cat("  Crude Q init:\n")
print(Qcrude)

base_fit <- tryCatch(
  msm(state ~ time_msm, subject = id, data = dat,
      qmatrix = Qcrude,
      control = list(maxit = MAXIT)),
  error = function(e) {
    cat(sprintf("  ⚠ Baseline model failed: %s\n", e$message))
    NULL
  }
)

if (!is.null(base_fit)) {
  Qbase <- qmatrix.msm(base_fit)$estimate
  # Enforce structure
  Qbase[!is.finite(Qbase)] <- 0
  Qbase[Qbase < 0] <- 0
  diag(Qbase) <- 0
  Qbase[2, ] <- 0
  for (rc in list(c(1,2), c(1,3), c(3,1), c(3,2))) {
    r <- rc[1]; cc <- rc[2]
    if (Qbase[r, cc] <= 0) Qbase[r, cc] <- Qcrude[r, cc]
  }
  cat("  ✓ Baseline model converged.\n")
  cat("  Estimated Q:\n")
  print(Qbase)
} else {
  Qbase <- Qcrude
  cat("  Using crude Q init as fallback.\n")
}

# ══════════════════════════════════════════════════════════════
# Identify feature columns
# ══════════════════════════════════════════════════════════════
feature_cols <- feature_cols_input[feature_cols_input %in% names(dat)]

# Coerce features
for (col in feature_cols) {
  dat[[col]] <- coerce_feature_column(dat[[col]])
}

cat(sprintf("\n  Feature columns: %d\n", length(feature_cols)))

# ══════════════════════════════════════════════════════════════
# Per-feature covariate screening
# ══════════════════════════════════════════════════════════════
cat("\n" , paste(rep("=", 60), collapse = ""), "\n")
cat("  Per-feature MSM screening\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

results    <- list()
errors_lst <- list()
ri <- 1

total_features <- length(feature_cols)

for (fi in seq_along(feature_cols)) {
  feature <- feature_cols[fi]
  cat(sprintf("[%d/%d] %s ... ", fi, total_features, feature))

  x <- dat[[feature]]

  # --- Sanity checks ---
  # Count non-NA values
  n_obs_feat <- sum(!is.na(x))
  if (n_obs_feat < 30) {
    cat(sprintf("SKIP (only %d non-NA)\n", n_obs_feat))
    errors_lst[[length(errors_lst) + 1]] <- data.frame(
      feature = feature, error = sprintf("too few non-NA: %d", n_obs_feat),
      stringsAsFactors = FALSE)
    next
  }

  n_unique <- length(unique(na.omit(x)))
  if (n_unique < 2) {
    cat("SKIP (no variation)\n")
    errors_lst[[length(errors_lst) + 1]] <- data.frame(
      feature = feature, error = "no variation", stringsAsFactors = FALSE)
    next
  }
  if (is.factor(x) && nlevels(x) > MAX_FACTOR_LEV) {
    cat(sprintf("SKIP (factor %d levels)\n", nlevels(x)))
    errors_lst[[length(errors_lst) + 1]] <- data.frame(
      feature = feature, error = paste0("too many levels: ", nlevels(x)),
      stringsAsFactors = FALSE)
    next
  }

  # Build model formula
  mm <- tryCatch(
    model.matrix(as.formula(paste0("~", feature)), data = dat[!is.na(x), ]),
    error = function(e) NULL
  )
  if (is.null(mm)) {
    cat("SKIP (model.matrix failed)\n")
    errors_lst[[length(errors_lst) + 1]] <- data.frame(
      feature = feature, error = "model.matrix failed", stringsAsFactors = FALSE)
    next
  }
  term_names <- colnames(mm)
  term_names <- term_names[term_names != "(Intercept)"]
  if (length(term_names) == 0) {
    cat("SKIP (no terms)\n")
    errors_lst[[length(errors_lst) + 1]] <- data.frame(
      feature = feature, error = "no terms", stringsAsFactors = FALSE)
    next
  }

  # --- Fit covariate model ---
  last_fit_error <- NA_character_
  used_scaled_retry <- FALSE

  fit_one <- function(var_name, term_names_local) {
    tryCatch({
      if (TIME_LIMIT_SEC > 0) {
        setTimeLimit(elapsed = TIME_LIMIT_SEC, transient = TRUE)
      }

      # Use stable baseline Q for all features
      Qfit <- Qbase
      diag(Qfit) <- 0

      trans_idx   <- which(Qfit > 0, arr.ind = TRUE)
      trans_names <- paste0(trans_idx[, 1], "-", trans_idx[, 2])

      cov_list <- setNames(
        rep(list(as.formula(paste0("~", var_name))), length(trans_names)),
        trans_names
      )

      n_q        <- sum(Qfit > 0)
      n_cov_coef <- length(term_names_local)
      parscale   <- c(rep(PARSCALE_Q, n_q),
                      rep(PARSCALE_COV, n_cov_coef * n_q))

      msm(
        state ~ time_msm,
        subject    = id,
        data       = dat,
        qmatrix    = Qfit,
        covariates = cov_list,
        control    = list(maxit = MAXIT, parscale = parscale),
        hessian    = TRUE
      )
    }, error = function(e) {
      last_fit_error <<- e$message
      NULL
    }, finally = {
      if (TIME_LIMIT_SEC > 0) {
        setTimeLimit(elapsed = Inf, transient = FALSE)
      }
    })
  }

  fit <- fit_one(feature, term_names)

  # Retry with z-score scaled numeric feature if overflow occurs.
  if (is.null(fit) &&
      !is.na(last_fit_error) &&
      grepl("numerical overflow", last_fit_error, fixed = TRUE) &&
      !is.factor(x)) {
    x_num <- suppressWarnings(as.numeric(x))
    x_sd <- sd(x_num, na.rm = TRUE)
    if (!is.na(x_sd) && x_sd > 0) {
      x_mean <- mean(x_num, na.rm = TRUE)
      scaled_var <- ".__msm_scaled_tmp__"
      dat[[scaled_var]] <- (x_num - x_mean) / x_sd

      mm2 <- tryCatch(
        model.matrix(as.formula(paste0("~", scaled_var)),
                     data = dat[!is.na(dat[[scaled_var]]), ]),
        error = function(e) NULL
      )

      if (!is.null(mm2)) {
        term_names2 <- colnames(mm2)
        term_names2 <- term_names2[term_names2 != "(Intercept)"]
        if (length(term_names2) > 0) {
          fit2 <- fit_one(scaled_var, term_names2)
          if (!is.null(fit2)) {
            fit <- fit2
            term_names <- gsub(scaled_var, feature, term_names2, fixed = TRUE)
            used_scaled_retry <- TRUE
          }
        }
      }
      dat[[scaled_var]] <- NULL
    }
  }

  if (is.null(fit)) {
    err_msg <- ifelse(is.na(last_fit_error), "fit failed", last_fit_error)
    errors_lst[[length(errors_lst) + 1]] <- data.frame(
      feature = feature, error = err_msg, stringsAsFactors = FALSE)
    cat("FAILED\n")
    next
  }

  res <- extract_cov_results(
    fit         = fit,
    term_names  = term_names,
    state_levels = state_levels,
    feature_name = feature,
    n_ids       = length(unique(dat$id[!is.na(dat[[feature]])])),
    n_obs       = n_obs_feat
  )

  if (!is.null(res)) {
    if (used_scaled_retry) {
      res$term <- paste0(res$term, "_zfit")
    }
    results[[ri]] <- res
    ri <- ri + 1
    cat(sprintf("OK  (HR range: %.3f–%.3f)\n",
                min(res$hr, na.rm = TRUE),
                max(res$hr, na.rm = TRUE)))
  } else {
    cat("OK  (no covariate results)\n")
  }
}

# ══════════════════════════════════════════════════════════════
# Aggregate & Save
# ══════════════════════════════════════════════════════════════
if (length(results) > 0) {
  all_results <- do.call(rbind, results)
} else {
  all_results <- data.frame()
}

if (length(errors_lst) > 0) {
  errors_df <- do.call(rbind, errors_lst)
} else {
  errors_df <- data.frame()
}

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# table2.1 — full HR/P-value results
table21_path <- file.path(OUT_DIR, "table2.1.csv")
write.csv(all_results, table21_path, row.names = FALSE)
cat(sprintf("\n  Saved table2.1: %s  (%d rows)\n", table21_path, nrow(all_results)))

# Backward-compatible filename
write.csv(all_results, file.path(OUT_DIR, "table2.1_msm_hr_pvalues.csv"),
          row.names = FALSE)

# Errors log
err_path <- file.path(OUT_DIR, "table2.1_msm_errors.csv")
write.csv(errors_df, err_path, row.names = FALSE)
cat(sprintf("  Errors log: %s  (%d entries)\n", err_path, nrow(errors_df)))

# ── Filter significant features ──
if (nrow(all_results) > 0) {
  sig <- all_results[
    !is.na(all_results$p) &
    ((all_results$hr < 0.90) | (all_results$hr > 1.10)) &
    (all_results$p < 0.01),
  ]
  sig_features <- sort(unique(sig$feature))
} else {
  sig <- data.frame()
  sig_features <- character()
}

cat(sprintf("\n══════════════════════════════════════════════════\n"))
cat(sprintf("  Significant features (HR<0.90|>1.10, P<0.01): %d\n",
            length(sig_features)))
cat(sprintf("══════════════════════════════════════════════════\n"))
for (f in sig_features) {
  cat(sprintf("  ✓ %s\n", f))
}

# Save significant features
sig_feat_path <- file.path(OUT_DIR, "msm_significant_features.csv")
write.csv(data.frame(feature = sig_features), sig_feat_path, row.names = FALSE)

# Save significant detail
sig_detail_path <- file.path(OUT_DIR, "table2.1_msm_significant_detail.csv")
write.csv(sig, sig_detail_path, row.names = FALSE)

# Save MSM analysis set = id + states + selected features
keep_cols <- c("id", "state_start", "state_stop", sig_features)
keep_cols <- keep_cols[keep_cols %in% names(df)]
analysis_set <- df[, keep_cols, drop = FALSE]
analysis_set_path <- file.path(OUT_DIR, "analysis_set_msm.csv")
write.csv(analysis_set, analysis_set_path, row.names = FALSE)
cat(sprintf("  Analysis set: %s  (%d rows × %d cols)\n",
            analysis_set_path, nrow(analysis_set), ncol(analysis_set)))

cat(sprintf("\n  All results → %s/\n", OUT_DIR))
cat("  MSM feature selection complete.\n")
