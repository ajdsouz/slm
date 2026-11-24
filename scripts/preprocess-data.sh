#!/bin/bash

uv run scripts/preprocess-data.py \
    --dataset-name ZennyKenny/tactical-military-reasoning-v.1.0 \
    --num-proc 4 \
    --input-column scenario_description \
    --test-ratio 0.1 \
    --tokenizer Qwen/Qwen3.5-35B-A3B \
    --data-dir data/