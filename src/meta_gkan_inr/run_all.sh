#!/bin/bash

CONFIG_DIR="../../config"

for file in "$CONFIG_DIR"/*.yaml; do
    rel_path="$(basename "$file")"
    echo "Running with $rel_path"
    python3 learn.py "$rel_path"
done