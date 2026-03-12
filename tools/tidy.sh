#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"

echo "=== ruff format ==="
ruff format

echo "=== ruff check --fix ==="
ruff check --fix

echo "=== pyright ==="
pyright -p .

echo "=== clang-format ==="
find cpp_src -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec clang-format-19 -i {} +

