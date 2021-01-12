#!/bin/bash

CONFIG_NAME=$1

shift 1

python /autoalbument_benchmarks/main.py --config-name "$CONFIG_NAME" "$@"
