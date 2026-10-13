# CompositeSEM for Weka

[![CI](https://github.com/hchafai/weka-sem/actions/workflows/ci.yml/badge.svg)](https://github.com/hchafai/weka-sem/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

CompositeSEM is an experimental Weka extension containing two deliberately
separate model families: configurable fixed-weight composites and
maximum-likelihood covariance SEM. It provides:

- `weka.filters.supervised.attribute.CompositeSEMFilter`, which appends fitted
  construct scores to numeric Weka data;
- `weka.classifiers.meta.CompositeSEMRegressor`, which enriches a configurable
  Weka regressor with those scores;
- `weka.filters.supervised.attribute.CovarianceSEMFilter`, which appends
  regression factor scores and the fitted SEM prediction;
- `weka.classifiers.functions.CovarianceSEMRegressor`, which jointly estimates
  a reflective measurement model and paths to a numeric outcome;
- serializable Java models exposing loadings, error variances, latent covariance,
  path coefficients, R-squared and the fit diagnostics applicable to each
  estimator.

The composite implementation uses fixed, user-supplied measurement weights and
is not canonical iterative PLS-PM. The covariance implementation estimates
loadings, measurement errors, latent covariances, structural paths and outcome
residual variance by minimizing the normal-theory ML discrepancy function.

## Installation

CompositeSEM requires Java 17 or later and Weka 3.8.7 or later. Install the
published release directly:

```bash
java -cp weka.jar weka.core.WekaPackageManager \
  -install-package \
  https://github.com/hchafai/weka-sem/releases/download/v0.2.0/CompositeSEM-0.2.0.zip
```

Alternatively, build and test the package from this repository:

```bash
scripts/build-package.sh
scripts/test-package.sh
```

Then install the local archive:

```bash
java -cp weka.jar weka.core.WekaPackageManager \
  -install-package output/CompositeSEM-0.2.0.zip
```

The components then appear under Weka's supervised attribute filters, meta
classifiers and function classifiers. They can also be addressed by full class
name from `weka.Run`.

## Construct specification

Use one semicolon-separated definition per construct:

```text
Activation=engagement,comments:2,sentiment;Reach=followers,views
```

The optional value after `:` is a signed measurement weight. For the example,
the Activation weights are normalized to 0.25, 0.50 and 0.25. Attribute names
must identify numeric attributes and cannot identify the class.

## Meta-regressor example

```bash
java weka.Run weka.classifiers.meta.CompositeSEMRegressor \
  -constructs "Activation=engagement,comments,sentiment;Reach=followers,views" \
  -ridge 1.0E-6 \
  -W weka.classifiers.trees.RandomForest -- -I 200 -depth 12 \
  -t campaigns.arff -c last
```

The package includes `examples/composite-sem-demo.arff`, which can be used in
place of `campaigns.arff` for a quick installation check.

By default, the base regressor receives the original predictors, one
standardized attribute per construct and a structural performance score. Use
`-latent-only` to train only on the generated scores and `-no-performance` to
omit the structural performance score.

## Filter example

```bash
java weka.Run weka.filters.supervised.attribute.CompositeSEMFilter \
  -constructs "Activation=engagement,comments,sentiment;Reach=followers,views" \
  -i campaigns.arff -o campaigns-with-sem.arff -c last
```

The filter learns all means, standard deviations, composite scaling and path
coefficients on its first batch. Later batches are transformed with the stored
model. In cross-validation, use it through a Weka meta-classifier or construct a
new filter for every training fold.

## Covariance SEM

The covariance estimator uses the same measurement syntax, but ignores optional
fixed weights because its loadings are estimated. Each reflective factor must
have at least three indicators, and an indicator may belong to only one factor:

```text
Activation=engagement,comments,sentiment;Reach=followers,views,frequency
```

The first loading of each factor is fixed to one for marker-variable
identification. All remaining loadings, diagonal measurement-error variances,
the full latent covariance matrix, paths from every factor to the numeric class,
and the class residual variance are estimated jointly. Positive variances and a
positive-definite latent covariance are guaranteed by exponential and Cholesky
parameterizations.

Run the covariance SEM regressor:

```bash
java weka.Run weka.classifiers.functions.CovarianceSEMRegressor \
  -constructs "Activation=engagement,comments,sentiment;Reach=followers,views,frequency" \
  -regularization 1.0E-6 -max-iterations 300 -tolerance 1.0E-6 \
  -t campaigns.arff -c last
```

The report includes chi-square and its degrees of freedom, CFI, TLI, RMSEA,
SRMR, AIC, BIC, target R-squared, convergence state, loadings, latent covariance
and structural paths. The corresponding filter is:

```bash
java weka.Run weka.filters.supervised.attribute.CovarianceSEMFilter \
  -constructs "Activation=engagement,comments,sentiment;Reach=followers,views,frequency" \
  -i campaigns.arff -o campaigns-with-covariance-sem.arff -c last
```

Missing indicators are mean-imputed using training-set statistics. Missing
classes are excluded from estimation. At scoring time the numeric class is not
used: factor scores are regression scores computed from indicators only.

## Scientific scope

Version 0.2 supports correlated reflective latent predictors with direct paths
to one observed numeric outcome. It does reproduce the full indicator/outcome
covariance matrix, but it does not yet parse arbitrary latent graphs, correlated
measurement errors, ordinal estimators, multigroup models or FIML missing-data
likelihoods. Fit does not establish causality. Before an official Weka repository
submission, numerical results must be benchmarked against established SEM tools
on published reference datasets.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The related
open-access preprint is:

Hicham CHAFAI (2026), *From Influencer Selection to Post-Campaign Value: An
Explainable Analytics Framework for Influencer Marketing Campaigns*.
[https://doi.org/10.5281/zenodo.23321537](https://doi.org/10.5281/zenodo.23321537)
