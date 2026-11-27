#!/bin/sh
# Resource probe for kernels/quant/q5k-dense-vdr4.md. Needs a llama.cpp tree at bc7898a2 for ggml-common.h:
#   stew675/llama.cpp @ af7a57ba8700f499fa6ca01d47470208deae8718 (= ggml-org/llama.cpp 919794500475 + rdna-boosts r368 patches).
# Measured 2026-11-27 with Debian hipcc 5.7.1 (HIP 5.7.31921, Debian clang 17.0.6) -- NOT the dest ROCm 7.14 compiler.
# -DGGML_CUDA_Q5K_DENSE_VDR4 force-enables the dense function on gfx1030 for the probe only; upstream gates it off there.
set -e
G=${LLAMA:-../llama}/ggml
hipcc --offload-arch=gfx1030 -O3 -std=c++17 -DGGML_CUDA_Q5K_DENSE_VDR4 -I$G/src -I$G/include \
  --cuda-device-only -c q5k_probe.cu -o q5k_probe.o -save-temps -Rpass-analysis=kernel-resource-usage > rpass.txt 2>&1
python3 table.py
