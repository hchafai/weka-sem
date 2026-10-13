# Changelog

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
