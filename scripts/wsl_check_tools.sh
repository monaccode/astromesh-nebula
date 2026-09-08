#!/usr/bin/env bash
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
for t in cmake make git gcc g++ ninja; do
  if command -v "$t" >/dev/null 2>&1; then
    echo "$t: $(command -v "$t")"
  else
    echo "$t: MISSING"
  fi
done
