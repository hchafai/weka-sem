#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$PROJECT_DIR/output/CompositeSEM-0.3.0.zip"
DATA="$PROJECT_DIR/examples/composite-sem-demo.arff"
COMPOSITE_SPEC="Activation=engagement,comments:2,sentiment;Reach=followers,views"
COVARIANCE_SPEC="Activation=engagement,comments,sentiment;Reach=followers,views,frequency"

if [[ -n "${MAVEN_BIN:-}" ]]; then
  MVN="$MAVEN_BIN"
elif command -v mvn >/dev/null 2>&1; then
  MVN="$(command -v mvn)"
else
  echo "Maven 3.6.3 or later is required, or set MAVEN_BIN." >&2
  exit 1
fi

"$PROJECT_DIR/scripts/build-package.sh" >/dev/null
mkdir -p "$PROJECT_DIR/tmp"
"$MVN" -f "$PROJECT_DIR/pom.xml" -q dependency:build-classpath \
  -Dmdep.includeScope=test -Dmdep.outputFile="$PROJECT_DIR/tmp/classpath.txt"
WEKA_CLASSPATH="$(awk -v RS=: '$0 ~ /\.jar$/ {if (result) result=result ":" $0; else result=$0} END {print result}' \
  "$PROJECT_DIR/tmp/classpath.txt")"

ISOLATED_WEKA_HOME="$(mktemp -d "${TMPDIR:-/tmp}/weka-sem-home.XXXXXX")"
FILTERED_ARFF="$(mktemp "${TMPDIR:-/tmp}/weka-sem-filtered.XXXXXX.arff")"
REPORT="$(mktemp "${TMPDIR:-/tmp}/weka-sem-report.XXXXXX.txt")"
trap 'rm -rf "$ISOLATED_WEKA_HOME"; rm -f "$FILTERED_ARFF" "$REPORT"' EXIT
export WEKA_HOME="$ISOLATED_WEKA_HOME"

java -Dweka.packageManager.offline=true -cp "$WEKA_CLASSPATH" \
  weka.core.WekaPackageManager -install-package "$ARCHIVE" >/dev/null

java -Dweka.packageManager.offline=true -cp "$WEKA_CLASSPATH" \
  weka.Run weka.classifiers.meta.CompositeSEMRegressor \
  -t "$DATA" -c last -constructs "$COMPOSITE_SPEC" \
  -W weka.classifiers.trees.RandomForest -- -I 40 -depth 8 >"$REPORT"
grep -q 'Composite path model' "$REPORT"

java -Dweka.packageManager.offline=true -cp "$WEKA_CLASSPATH" \
  weka.Run weka.classifiers.functions.CovarianceSEMRegressor \
  -t "$DATA" -c last -constructs "$COVARIANCE_SPEC" \
  -max-iterations 350 -no-cv >"$REPORT"
grep -q 'Covariance SEM (maximum likelihood)' "$REPORT"
grep -q 'Converged: true' "$REPORT"

java -Dweka.packageManager.offline=true -cp "$WEKA_CLASSPATH" \
  weka.Run weka.filters.supervised.attribute.CovarianceSEMFilter \
  -constructs "$COVARIANCE_SPEC" -i "$DATA" -o "$FILTERED_ARFF" -c last
grep -q '@attribute sem_ml_activation numeric' "$FILTERED_ARFF"
grep -q '@attribute sem_ml_prediction numeric' "$FILTERED_ARFF"

echo "CompositeSEM package installation and execution checks passed."
