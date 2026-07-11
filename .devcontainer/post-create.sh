#!/bin/bash
# Runs once when the dev container is created (also on "Rebuild Container").
set -e

source /dory_env/bin/activate
# python3 -m pip install --upgrade pip --quiet

if [ -f "${containerWorkspaceFolder:-/workspaces/ALADIN}/requirements-dev.txt" ]; then
    echo "[postCreate] Installing requirements-dev.txt ..."
    pip install -r "${containerWorkspaceFolder:-/workspaces/ALADIN}/requirements-dev.txt"
else
    echo "[postCreate] No requirements-dev.txt found — skipping."
fi