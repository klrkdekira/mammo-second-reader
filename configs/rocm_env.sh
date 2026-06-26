#!/usr/bin/env bash
# AMD ROCm environment variables for the RX 7900 GRE used during development.
export HIP_CLANG_PATH=/opt/rocm/llvm/bin
export CPLUS_INCLUDE_PATH=/usr/include/c++/12:/usr/include/x86_64-linux-gnu/c++/12
# export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
