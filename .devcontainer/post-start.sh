#!/bin/bash
# Runs every time the container starts (including reopening an existing one).
# Cheap: pip no-ops on packages that are already the right version.
set -e

source /dory_env/bin/activate

WORKDIR="${containerWorkspaceFolder:-/workspaces/ALADIN}"

if [ -f "$WORKDIR/requirements-dev.txt" ]; then
    pip install -r "$WORKDIR/requirements-dev.txt" --quiet
    echo "DEV requiremets installed!"
fi