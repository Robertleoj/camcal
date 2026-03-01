#!/usr/bin/env bash
set -euo pipefail

version=$(grep '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
tag="v$version"

echo "Releasing $tag"

# Must be on main
branch=$(git branch --show-current)
if [ "$branch" != "main" ]; then
    echo "Error: not on main (on $branch)"
    exit 1
fi

# No uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: uncommitted changes"
    exit 1
fi

# Local main must match remote
git fetch origin main --quiet
local_sha=$(git rev-parse HEAD)
remote_sha=$(git rev-parse origin/main)
if [ "$local_sha" != "$remote_sha" ]; then
    echo "Error: local main ($local_sha) differs from origin/main ($remote_sha)"
    exit 1
fi

# Tag must not already exist
if git rev-parse "$tag" >/dev/null 2>&1; then
    echo "Error: tag $tag already exists"
    exit 1
fi

git tag "$tag"
git push origin "$tag"

echo "Pushed $tag — PyPI release should be triggered"
