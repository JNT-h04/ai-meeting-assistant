#!/usr/bin/env bash
# Builds echo-src.tar.gz: the project without .venv, models, audio and private notes (~1 MB).
set -euo pipefail
cd "$(dirname "$0")/../.."
tmp=$(mktemp -d)
tar czf "$tmp/echo-src.tar.gz" --exclude=./.venv --exclude=./.models --exclude=./data --exclude=./notes \
    --exclude='*.tar.gz' --exclude=__pycache__ --exclude=.pytest_cache --exclude=./deploy/oracle/.env .
mv "$tmp/echo-src.tar.gz" . && rmdir "$tmp"
ls -l echo-src.tar.gz
