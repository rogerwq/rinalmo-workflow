#!/bin/bash
set -e
export TRITON_CACHE_DIR=/tmp/triton_cache
python predict.py
