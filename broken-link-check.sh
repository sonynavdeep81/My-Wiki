#!/usr/bin/env bash
# Lists every [[wikilink]] target used anywhere in *.md that doesn't resolve to
# an actual file. Handles: full-path links ([[wiki/concepts/x]]), bare-name
# links matched by filename ([[x]]), Obsidian-style aliases ([[slug|Display]]
# or [[slug\|Display]]), and title-case display names resolved via
# slugification ([[Attention Is All You Need]] -> attention-is-all-you-need.md).
set -euo pipefail
cd "$(dirname "$0")"

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
}

found=0
while IFS= read -r raw; do
  target=$(echo "$raw" | sed -E 's/\\?\|.*$//')
  [ -z "$target" ] && continue

  if [[ "$target" == wiki/* ]]; then
    if [ ! -f "$target" ] && [ ! -f "${target}.md" ]; then
      echo "$raw"
      found=1
    fi
  else
    slug=$(slugify "$target")
    if find wiki -iname "${target}.md" 2>/dev/null | grep -q .; then
      continue
    elif [ -n "$slug" ] && find wiki -iname "${slug}.md" 2>/dev/null | grep -q .; then
      continue
    else
      echo "$raw"
      found=1
    fi
  fi
done < <(grep -rhoE '\[\[[^]]+\]\]' --include='*.md' --exclude=CLAUDE.md --exclude=LESSONS.md . \
          | sed -E 's/\[\[(.*)\]\]/\1/' | sort -u)

if [ "$found" -eq 0 ]; then
  echo "No broken links found."
fi
