#!/bin/bash
# tt — headless ticker-tape CLI wrapper
# Activates venv and forwards args to cli.py
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/venv/bin/activate"
python "$DIR/cli.py" "$@"
