#!/usr/bin/env bash
set -euo pipefail
RAO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${PYTHON:-python}" -u -B "$RAO_ROOT/run.py" --source-model clip_vit "$@"
