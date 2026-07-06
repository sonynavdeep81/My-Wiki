#!/usr/bin/env bash
# Lists every .md file under wiki/ not referenced (by basename) anywhere in index.md.
# Excludes wiki/schema/ (format templates loaded on demand, not catalog content).
set -euo pipefail
cd "$(dirname "$0")"

found=0
while IFS= read -r f; do
  base=$(basename "$f" .md)
  if ! grep -q "$base" index.md; then
    echo "$f"
    found=1
  fi
done < <(find wiki -name "*.md" -not -path "wiki/schema/*" | sort)

if [ "$found" -eq 0 ]; then
  echo "No orphans found."
fi
