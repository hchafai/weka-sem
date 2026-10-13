# Changelog

## 0.3.0 — 2026-10-13

- Added original-scale estimates with observed-information standard errors,
  Wald z statistics and two-sided p-values.
- Added completely standardized loadings, paths, residual variances and latent
  correlations.
- Corrected chi-square scaling and added full normal-theory likelihood AIC/BIC.
- Added lavaan 0.7-3 validation on Holzinger–Swineford 1939 and Political
  Democracy with documented numerical tolerances.
- Added prediction benchmarks against Weka LinearRegression, RandomForest and
  the official partialLeastSquares package.
- Added packaged Javadoc, an explicit Java 17 baseline, and a Maven rule that
  bans Spring dependencies.
- Refreshed GitHub Actions for Weka 3.8.7 and 3.9.6 and added reproducible
  R/lavaan validation.

## 0.2.0 — 2026-10-13

- Added maximum-likelihood covariance SEM for correlated reflective factors and
  one observed numeric outcome.
- Added joint estimation of loadings, measurement-error variances, latent
  covariance, structural paths and outcome residual variance.
- Added chi-square, CFI, TLI, RMSEA, SRMR, AIC, BIC and target R-squared.
- Added regression factor scoring without target leakage.
- Added `CovarianceSEMRegressor` and `CovarianceSEMFilter` Weka entry points.
- Retained the fixed-weight composite filter and meta-regressor from 0.1.0.
- Added synthetic recovery, cross-validation, missing-value and serialization
  tests.
