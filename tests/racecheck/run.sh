#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

compute-sanitizer \
  --tool racecheck \
  --racecheck-report hazard \
  --error-exitcode 99 \
  --launch-timeout 0 \
  --print-limit 100 \
  .venv/bin/python tests/racecheck/fused_operators.py
