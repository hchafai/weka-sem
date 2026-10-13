#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="0.2.0"
OUTPUT_DIR="$PROJECT_DIR/output"
JAR="$PROJECT_DIR/target/CompositeSEM.jar"

if [[ -n "${MAVEN_BIN:-}" ]]; then
  MVN="$MAVEN_BIN"
elif command -v mvn >/dev/null 2>&1; then
  MVN="$(command -v mvn)"
else
  echo "Maven 3.6.3 or later is required, or set MAVEN_BIN." >&2
  exit 1
fi

"$MVN" -f "$PROJECT_DIR/pom.xml" -q verify

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/weka-sem-package.XXXXXX")"
trap 'rm -rf "$STAGE_DIR"' EXIT

cp "$JAR" "$STAGE_DIR/CompositeSEM.jar"
cp "$PROJECT_DIR/Description.props" "$STAGE_DIR/Description.props"
cp "$PROJECT_DIR/LICENSE" "$STAGE_DIR/LICENSE"
cp "$PROJECT_DIR/README.md" "$STAGE_DIR/README.md"
cp "$PROJECT_DIR/CITATION.cff" "$STAGE_DIR/CITATION.cff"
cp "$PROJECT_DIR/CHANGELOG.md" "$STAGE_DIR/CHANGELOG.md"
cp "$PROJECT_DIR/pom.xml" "$STAGE_DIR/pom.xml"
cp -R "$PROJECT_DIR/src" "$STAGE_DIR/src"
cp -R "$PROJECT_DIR/examples" "$STAGE_DIR/examples"

mkdir -p "$OUTPUT_DIR"
STAGED_ARCHIVE="$STAGE_DIR/CompositeSEM-$VERSION.zip"
(cd "$STAGE_DIR" && /usr/bin/zip -q -r "$STAGED_ARCHIVE" \
  CompositeSEM.jar Description.props LICENSE README.md CITATION.cff CHANGELOG.md pom.xml src examples)
cp "$STAGED_ARCHIVE" "$OUTPUT_DIR/CompositeSEM-$VERSION.zip"

if /usr/bin/zipinfo -1 "$OUTPUT_DIR/CompositeSEM-$VERSION.zip" \
  | grep -Eq '(^|/)(\.DS_Store|__MACOSX|\._)'; then
  echo "macOS metadata was found in the package archive" >&2
  exit 1
fi

shasum -a 256 "$OUTPUT_DIR/CompositeSEM-$VERSION.zip" \
  > "$OUTPUT_DIR/SHA256SUMS.txt"
cat "$OUTPUT_DIR/SHA256SUMS.txt"
