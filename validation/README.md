# Scientific validation

This directory separates scientific validation from the tiny synthetic ARFF in
`examples/`. The latter remains an installation smoke test and is not evidence
of model fit, inferential validity, or predictive quality.

## Public reference data

Two datasets distributed with the GPL-licensed `lavaan` package are used:

1. **Holzinger–Swineford 1939** (301 observations): the nine-indicator,
   three-factor mental-ability dataset from the official lavaan CFA tutorial.
   CompositeSEM fits those three reflective factors with `ageyr` as the observed
   outcome.
2. **Political Democracy** (75 observations): the classic Bollen democracy and
   industrialization data. CompositeSEM validates the supported submodel with
   `ind60` and `dem60` as correlated latent predictors and `y8` as the observed
   outcome. This is intentionally not presented as a reproduction of Bollen's
   full latent-endogenous SEM, which the current engine cannot express.

Sources:

- <https://lavaan.ugent.be/tutorial/cfa.html>
- <https://github.com/cran/lavaan/blob/master/man/HolzingerSwineford1939.Rd>
- <https://github.com/cran/lavaan/blob/master/man/PoliticalDemocracy.Rd>

The ARFF files are deterministic exports from the datasets bundled with
`lavaan`. Rebuild them and the reference table with:

```bash
Rscript validation/lavaan/generate-reference.R
```

## Independent reference and tolerances

`lavaan-reference.csv` is generated with normal-theory ML, marker-variable
identification, no mean structure, and the **observed information matrix**. The
latter matches CompositeSEM's numerical observed-Hessian Wald inference.

The automated comparison uses deliberately explicit absolute tolerances:

| Quantity | Absolute tolerance |
| --- | ---: |
| chi-square, AIC, BIC | 0.02 |
| CFI, TLI, RMSEA, SRMR | 0.001 |
| loadings and structural paths | 0.005 |
| fully standardized loadings and paths (`std.all`) | 0.005 |
| observed-information standard errors | 0.005 |
| p-values | 0.005 |

The estimator uses a `1e-10` numerical covariance ridge for this validation,
up to 1,500 BFGS iterations, and an optimization tolerance of `1e-8`. The ridge
is used only while estimating parameters; reported fit indices are evaluated
against the unregularized sample covariance matrix.

The comparison establishes numerical agreement for the model family currently
implemented. It does not validate unsupported arbitrary path graphs, ordinal
estimators, FIML missingness, multigroup models, causal claims, or universal fit
cutoffs.

## Prediction benchmark

`prediction-benchmarks.csv` reports repeated ten-fold cross-validation for:

- `CovarianceSEMRegressor`;
- Weka `LinearRegression`;
- Weka `RandomForest`;
- Weka's official `partialLeastSquares` package (`PLSClassifier`, SIMPLS).

The seeds are fixed to 1, 17, and 42. RMSE, MAE, and Pearson correlation are
averaged across repetitions. These results characterize predictive behaviour;
they are not framed as a claim that SEM must outperform general-purpose
predictors.
