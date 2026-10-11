#!/usr/bin/env bash
# mma-probe build.
#
#   ./build.sh gfx1030                 # on the V620 box (hipcc from ROCm)
#   ./build.sh gfx1100                 # on the W7800 / 7900 XTX box
#   ./build.sh --device-only gfx1030   # ISA-only, no ROCm: plain clang + shim
#
# One --offload-arch per invocation, never both arches in one object.
# Two variants per family:
#   default : -fno-gpu-flush-denormals-to-zero (descriptor FLOAT_DENORM_MODE_32 = 3)
#   ftz     : -fgpu-flush-denormals-to-zero    (descriptor FLOAT_DENORM_MODE_32 = 0)
# fp-contract is pinned per family: off for DOT/WMMA (no compiler re-association
# around the instruction under test), fast-honor-pragmas for fma_baseline (FMA
# formation is wanted there, and mul_add_c opts out with
# '#pragma clang fp contract(off)'; plain 'fast' ignores that pragma).
# Never fast-math.
set -euo pipefail
cd "$(dirname "$0")"

DEVICE_ONLY=0
if [[ "${1:-}" == "--device-only" ]]; then DEVICE_ONLY=1; shift; fi
ARCH=${1:?usage: ./build.sh [--device-only] gfx1030|gfx1100}
case "$ARCH" in
  gfx1030) ID=1030; FAMS="dot2_f16 dot4_i8 fma_baseline" ;;
  gfx1100) ID=1100; FAMS="dot2_f16 dot2_bf16 wmma dot4_i8 fma_baseline" ;;
  *) echo "unsupported arch $ARCH (gfx1030 or gfx1100)"; exit 1 ;;
esac

OUT=${OUT:-build/$ARCH}
ROCM=${ROCM_PATH:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM/bin/hipcc}
CLANG=${CLANG:-clang}
if [[ $DEVICE_ONLY == 1 ]]; then
  LLVMBIN=${LLVMBIN:-$(dirname "$(command -v "$CLANG")")}
else
  LLVMBIN=${LLVMBIN:-$ROCM/llvm/bin}
fi
READELF=${READELF:-$(command -v "$LLVMBIN/llvm-readelf" || command -v llvm-readelf || true)}
OBJDUMP=${OBJDUMP:-$(command -v "$LLVMBIN/llvm-objdump" || command -v llvm-objdump || true)}

COMMON=(-O2 -std=c++17 -fno-fast-math -fno-finite-math-only -fno-unsafe-math-optimizations
        -DMMAPROBE_ARCH_ID=$ID -Wall -Wno-unused-function)

mkdir -p "$OUT"
{
  echo "date: $(date -Iseconds)"
  echo "arch: $ARCH  device_only: $DEVICE_ONLY"
  if [[ $DEVICE_ONLY == 1 ]]; then
    echo "compiler: $("$CLANG" --version | head -1)"
    echo "rocm: none (device-only build with shim/hip_shim.h, -nogpulib -nogpuinc)"
  else
    echo "hipcc: $("$HIPCC" --version 2>&1 | tr '\n' ' ' | head -c 400)"
    echo "rocm: $(cat "$ROCM/.info/version" 2>/dev/null || echo unknown)"
  fi
  echo "readelf: ${READELF:-missing}  objdump: ${OBJDUMP:-missing}"
} > "$OUT/toolchain.txt"

for VAR in default ftz; do
  mkdir -p "$OUT/$VAR"
  : > "$OUT/$VAR/flags.txt"
  if [[ $VAR == default ]]; then DEN=-fno-gpu-flush-denormals-to-zero; else DEN=-fgpu-flush-denormals-to-zero; fi
  for FAM in $FAMS; do
    if [[ $FAM == fma_baseline ]]; then CONTRACT=-ffp-contract=fast-honor-pragmas; else CONTRACT=-ffp-contract=off; fi
    FLAGS=("${COMMON[@]}" "$CONTRACT" "$DEN")
    echo "[$ARCH/$VAR] $FAM  ${FLAGS[*]}"
    echo "$FAM: ${FLAGS[*]}" >> "$OUT/$VAR/flags.txt"
    if [[ $DEVICE_ONLY == 1 ]]; then
      DEV=("$CLANG" -x hip --offload-arch="$ARCH" -nogpulib -nogpuinc --cuda-device-only
           -DMMAPROBE_DEVICE_ONLY "${FLAGS[@]}")
    else
      DEV=("$HIPCC" -x hip --offload-arch="$ARCH" --cuda-device-only "${FLAGS[@]}")
      "$HIPCC" -x hip --offload-arch="$ARCH" "${FLAGS[@]}" "$FAM.hip" -o "$OUT/$VAR/$FAM"
    fi
    "${DEV[@]}" -S "$FAM.hip" -o "$OUT/$VAR/$FAM.s"
    "${DEV[@]}" -c --no-gpu-bundle-output "$FAM.hip" -o "$OUT/$VAR/$FAM.co"
    if [[ -n "$READELF" ]]; then "$READELF" --notes "$OUT/$VAR/$FAM.co" > "$OUT/$VAR/$FAM.notes.txt"; fi
    if [[ -n "$OBJDUMP" ]]; then "$OBJDUMP" -d --mcpu="$ARCH" "$OUT/$VAR/$FAM.co" > "$OUT/$VAR/$FAM.dis.s"; fi
  done
done

python3 check_isa.py "$OUT" "$ARCH" | tee "$OUT/isa-check.txt"
