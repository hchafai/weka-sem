#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${MAVEN_BIN:-}" ]]; then
  MVN="$MAVEN_BIN"
elif command -v mvn >/dev/null 2>&1; then
  MVN="$(command -v mvn)"
else
  echo "Maven 3.6.3 or later is required, or set MAVEN_BIN." >&2
  exit 1
fi

if ! command -v Rscript >/dev/null 2>&1; then
  echo "R and lavaan are required to regenerate the independent reference." >&2
  exit 1
fi

cd "$PROJECT_DIR"
Rscript validation/lavaan/generate-reference.R
"$MVN" --batch-mode verify
"$MVN" -q test-compile dependency:build-classpath \
  -Dmdep.includeScope=test -Dmdep.outputFile="$PROJECT_DIR/tmp/benchmark-classpath.txt"

BENCHMARK_CLASSPATH="$(cat "$PROJECT_DIR/tmp/benchmark-classpath.txt")"
java -Dweka.packageManager.offline=true \
  -cp "$PROJECT_DIR/target/test-classes:$PROJECT_DIR/target/classes:$BENCHMARK_CLASSPATH" \
  com.hopeproject.influencer.sem.weka.PredictionBenchmarkRunner

echo "Scientific validation and prediction benchmarks passed."
