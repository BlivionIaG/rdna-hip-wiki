# Probe: HIP `*_sync` mask width on gfx1030 (2026-11-24)

Compiled through the Compiler Explorer (godbolt.org) REST API on 2026-11-24
(Europe/Paris). Two compilers, both with the ROCm 7.14.0 HIP headers that
Compiler Explorer installs at `/opt/compiler-explorer/libs/rocm/7.14.0` (path
visible in the `*.stderr.txt` files):

- `clang-rocm-71400` ("x86-64 clang rocm-7.14.0"): the clang that ships with
  ROCm 7.14.0, driven in HIP mode with
  `-x hip --rocm-path=/opt/compiler-explorer/libs/rocm/7.14.0`. Files
  `*.amdclang.*`.
- `hipclang-trunk-rocm-71400` ("clang trunk rocm-7.14.0"): upstream clang
  trunk (build date not exposed) with the same headers. Files without
  `amdclang` in the name.

| Source | Output | Extra options (all: `--offload-arch=gfx1030 -O3 --cuda-device-only -S`) |
|---|---|---|
| `k.hip` | `k.amdclang.ndebug.s` | ROCm clang, `-DNDEBUG` |
| `k.hip` | `k.amdclang.debug.s` | ROCm clang, no `NDEBUG` |
| `bad.hip` | `bad.amdclang.stderr.txt` | ROCm clang, `-DNDEBUG` (exit code 1) |
| `k.hip` | `k.ndebug.s` | clang trunk, `-DNDEBUG` |
| `k.hip` | `k.debug.s` | clang trunk, no `NDEBUG` |
| `bad.hip` | `bad.stderr.txt` | clang trunk, `-DNDEBUG` (exit code 1) |

Raw API output. The first line (`code N`) is dropped from the `.s` files and
ANSI colour codes are stripped from the `.stderr.txt` files.
