#!/usr/bin/env bash
# AMD ROCm environment variables for the RX 7900 GRE used during development.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
