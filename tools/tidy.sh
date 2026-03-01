#!/usr/bin/env bash
set -e
shopt -s globstar

cd "$(git rev-parse --show-toplevel)"

echo "=== ruff format ==="
ruff format

echo "=== ruff check --fix ==="
ruff check --fix

echo "=== pyright ==="
pyright -p .

echo "=== clang-format ==="
clang-format-19 -i cpp_src/**/*.{cpp,hpp}

