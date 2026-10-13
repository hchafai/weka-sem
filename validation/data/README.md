# Dataset provenance

The ARFF fixtures in this directory are generated without modification of the
numeric observations by `validation/lavaan/generate-reference.R` from datasets
distributed with the GPL-licensed R package `lavaan`.

- `holzinger-swineford-1939.arff`: 301 rows, variables `x1`–`x9` and `ageyr`.
- `political-democracy.arff`: 75 rows, variables `x1`–`x3`, `y1`–`y4`, and `y8`.

Their scientific provenance and the exact supported submodels are documented in
the parent `validation/README.md`. The project is GPL-3.0-or-later.
