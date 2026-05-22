# ollama-mi50-support

Patches and configuration to make **Ollama 0.24.0** run correctly on **AMD Radeon Instinct MI50 32GB** (gfx906/Vega20) under **ROCm 7.2.3**.

---

## Background

ROCm 7.2.3 dropped official support for gfx906 (Vega20). The AMD-distributed rocBLAS packages (Ubuntu) no longer include TensileLibrary files for gfx906, and Ollama's bundled rocBLAS follows the same omission. Without Tensile kernels, any rocBLAS call on gfx906 crashes immediately.

Additionally, some GGML kernels in the HIP backend have compilation issues or incorrect dispatches when built for gfx906 — specifically in `out-prod.cu` and `solve_tri.cu`.

This repository documents the fixes and the Tensile injection approach that restores full rocBLAS functionality.

---

## Problems Identified and Fixed

### 1. rocBLAS unavailable for gfx906 (primary issue)

**Root cause:**  
AMD's ROCm 7.x Ubuntu packages (and Ollama's bundled rocBLAS) have no TensileLibrary for gfx906. Any GEMM operation that reaches rocBLAS produces:
```
rocBLAS error: Cannot read .../TensileLibrary.dat: Illegal seek for GPU arch: gfx906
```

**Fix — Tensile injection from rocBLAS 6.4.3 (Arch Linux):**  
The Arch Linux package of rocBLAS 6.4.3 maintained gfx906 support even after AMD's Ubuntu packages dropped it. Extracting the 156 gfx906 files from that package and placing them in Ollama's bundled rocBLAS library directory restores full functionality.

See [How to Apply → Tensile Injection](#2-inject-tensile-files-for-gfx906) for the exact steps.

With Tensile available, the `ggml-cuda.cu` bypasses that routed gfx906 around rocBLAS are no longer needed and have been removed.

### 2. `out-prod.cu` compilation errors

**Root cause:**  
A Vega20 branch inside `ggml_cuda_out_prod` referenced variables declared later in the function and called `vega20_gemm_tiled` from a different translation unit.

**Fix:**  
- Moved all variable declarations before the Vega20 branch
- Defined a local `vega20_out_prod_gemm` template kernel inside `out-prod.cu`

### 3. `solve_tri.cu` — triangular solve crash for gfx906

**Root cause:**  
`cublasStrsmBatched` calls rocBLAS internally. Without Tensile, it crashes for dims outside the fast-path range (`n>64` or `k>32`).

**Fix:**  
Added `solve_tri_vega20` kernel for forward substitution, dispatched when `cc==VEGA20` and dims are outside the fast path.

### 4. `mmvq.cu` — LDS-staged Q4_K kernel

Added a dedicated `mul_mat_vec_q4K_vega20` kernel for Q4_K decode on gfx906:
- 256 threads (4 wavefronts × 64 threads), `__launch_bounds__(256, 10)`
- Stages the entire Q4_K row into LDS via 128-bit `uint4` coalesced loads
- Dispatched for `type == GGML_TYPE_Q4_K && cc == GGML_CUDA_CC_VEGA20 && !has_fusion`

Performance is equivalent to the standard GCN kernel (~12.8 t/s on gemma4:31b).

---

## Performance Results

Hardware: AMD Radeon Instinct MI50 32GB (gfx906, PCIe Gen3), ROCm 7.2.3  
Model: gemma4:31b (61 layers fully on GPU)

| State | Decode (t/s) | Prefill (t/s) |
|---|---|---|
| Without patches (rocBLAS crashes, fallback kernels) | ~13.4 | — |
| With patches + Tensile injection | **~12.8** | **54–83** |

The prefill rate (54–83 t/s) scales with prompt length as rocBLAS optimizes better for larger matrices. Decode is single-token generation (N=1 GEMM).

Other models:
| Model | Layers GPU | Decode |
|---|---|---|
| llama3.2:3b | full | ~85 t/s |
| qwen3.6:35b (MoE) | 41/41 | ~31 t/s |
| gemma4:31b (dense) | 61/61 | ~12.8 t/s |

---

## Files in this Repository

| File | Source path on VM | Purpose |
|---|---|---|
| `mmvq.cu` | `.../ggml-cuda/mmvq.cu` | LDS-staged Q4_K decode kernel for gfx906 |
| `out-prod.cu` | `.../ggml-cuda/out-prod.cu` | Fixed outer-product — variable order + local kernel |
| `ggml-cuda.cu` | `.../ggml-cuda/ggml-cuda.cu` | Removed vega20 GEMM bypasses (rocBLAS now works) |
| `solve_tri.cu` | `.../ggml-cuda/solve_tri.cu` | Triangular solve bypass for gfx906 |
| `CMakeLists.txt` | `/home/llm/ollama-hip-build/CMakeLists.txt` | Build file with all compile definitions |
| `ollama-rocm.conf` | `/etc/systemd/system/ollama.service.d/rocm.conf` | Ollama service environment configuration |
| `gpu-watchdog.sh` | `/usr/local/bin/gpu-watchdog.sh` | Thermal watchdog — SIGTERM at 85°C, SIGKILL at 90°C |
| `gpu-bench.sh` | `/usr/local/bin/gpu-bench.sh` | Benchmark wrapper with inter-run thermal cooldown |

All source files are from the Ollama 0.24.0 fork at `/home/llm/ollama-src/`.

---

## Requirements

- **GPU**: AMD Radeon Instinct MI50 (gfx906 / Vega20). May also apply to other gfx906 variants (RX Vega 56/64) with testing.
- **ROCm**: 7.2.3 (tested).
- **Ollama**: 0.24.0 (source build required — this patches the GGML HIP backend)
- **Build tools**: cmake >= 3.21, hipcc (from ROCm), make
- **Runtime**: psmisc (`apt install psmisc`) — required by `gpu-watchdog.sh` for `fuser`
- **OS**: Linux (tested on Ubuntu 22.04 inside Proxmox VM)

---

## How to Apply

### 1. Clone Ollama source and copy patched files

```bash
git clone https://github.com/ollama/ollama.git ollama-src
cd ollama-src
git checkout v0.24.0

SRC_DIR=ml/backend/ggml/ggml/src/ggml-cuda
cp /path/to/this-repo/mmvq.cu      $SRC_DIR/mmvq.cu
cp /path/to/this-repo/out-prod.cu  $SRC_DIR/out-prod.cu
cp /path/to/this-repo/ggml-cuda.cu $SRC_DIR/ggml-cuda.cu
cp /path/to/this-repo/solve_tri.cu $SRC_DIR/solve_tri.cu
```

### 2. Inject Tensile files for gfx906

Ollama's bundled rocBLAS does not include gfx906 kernels. Download them from the Arch Linux archive of rocBLAS 6.4.3, which maintained gfx906 support:

```bash
# Download the Arch Linux package (~251 MB)
wget "https://archive.archlinux.org/packages/r/rocblas/rocblas-6.4.3-3-x86_64.pkg.tar.zst" \
     -O rocblas-6.4.3-arch.pkg.tar.zst

# Extract and collect gfx906 files
mkdir -p rocblas-extract
tar -xf rocblas-6.4.3-arch.pkg.tar.zst -C rocblas-extract
cd rocblas-extract/opt/rocm/lib/rocblas/library
find . -name "*gfx906*" | tar -czf /tmp/tensile-gfx906.tar.gz -T -
cd -

# Install into Ollama's bundled rocBLAS (156 files)
sudo tar -xzf /tmp/tensile-gfx906.tar.gz \
     -C /usr/local/lib/ollama/rocm/rocblas/library/
```

Verify:
```bash
ls /usr/local/lib/ollama/rocm/rocblas/library/ | grep gfx906 | wc -l
# Should print 156
```

### 3. Build

```bash
mkdir -p ollama-hip-build
cp /path/to/this-repo/CMakeLists.txt ollama-hip-build/
cd ollama-hip-build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS=gfx906 \
  -DGGML_HIP=ON

cmake --build . --target ggml-hip -j$(nproc)
```

Expected output:
```
[  1%] Building HIP object CMakeFiles/ggml-hip.dir/.../ggml-cuda.cu.o
[  2%] Linking HIP shared module libggml-hip.so
[100%] Built target ggml-hip
```

### 4. Install

```bash
sudo cp libggml-hip.so /usr/local/lib/ollama/rocm/libggml-hip.so
sudo systemctl restart ollama
```

### 5. Configure Ollama service

Create `/etc/systemd/system/ollama.service.d/rocm.conf` (content in `ollama-rocm.conf`):

```ini
[Service]
Environment=HSA_OVERRIDE_GFX_VERSION=9.0.6
Environment=OLLAMA_HOST=0.0.0.0
Environment=OLLAMA_CONTEXT_LENGTH=8192
Environment=OLLAMA_FLASH_ATTENTION=1
Environment=OLLAMA_KV_CACHE_TYPE=q8_0
Environment=OLLAMA_KEEP_ALIVE=10m
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

`HSA_OVERRIDE_GFX_VERSION=9.0.6` is required to tell ROCm's HSA runtime to treat the GPU as gfx906.

### 6. Verify GPU inference

```bash
ollama run gemma4:31b "What is the capital of France?"
```

Check GPU usage (rocm-smi does not work for gfx906):
```bash
cat /sys/class/drm/card*/device/gpu_busy_percent
```

GPU should hit 100% during generation and drop to 0% immediately after.

---

## Known Limitations

- **Fusion path (has_fusion=true)**: The LDS Q4_K kernel only applies to non-fused decode. When `has_fusion=true` (SwiGLU/GeGLU projections), the standard GCN kernel is used — this path is stable and correct.

- **GPU utilization monitoring**: `rocm-smi --showuse` does not report correctly for gfx906. Use `/sys/class/drm/card*/device/gpu_busy_percent` instead.

- **DPM disabled in VM passthrough**: Under PCIe passthrough, the GPU is locked at ~925 MHz (half of max 1746 MHz). This is a Proxmox/KVM limitation, not a software issue — all performance numbers above reflect this constraint.

---

## Thermal Management

The MI50 under PCIe passthrough can reach 90–94°C under sustained load. The hardware thermal limit at 94–95°C triggers a hard reset that takes down the Proxmox host. Two tools mitigate this:

### GPU Watchdog (`gpu-watchdog.sh`)

A systemd service polling the temperature sensor every **1 second**:

- **≥ 80°C**: logs a warning
- **≥ 85°C**: SIGTERM to all processes with open GPU file descriptors; `systemctl stop ollama`
- **≥ 90°C**: SIGKILL to any survivors
- **≤ 45°C** (after a kill event): restarts `ollama.service` automatically

Install:
```bash
sudo cp gpu-watchdog.sh /usr/local/bin/gpu-watchdog.sh
sudo chmod +x /usr/local/bin/gpu-watchdog.sh
sudo cp gpu-watchdog.service /etc/systemd/system/
sudo systemctl enable --now gpu-watchdog
```

Monitor live:
```bash
journalctl -u gpu-watchdog -f
```

### Benchmark Cooldown (`gpu-bench.sh`)

Runs benchmarks with automatic inter-run cooldown:

```bash
gpu-bench.sh [model] [prompt] [runs]
# Example:
gpu-bench.sh gemma4:31b "What is 2+2?" 3
```

Waits in 30-second intervals between runs until GPU drops below 45°C.

### Check GPU temperature

```bash
cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input | awk '{print $1/1000 "°C"}'
```

---

## Infrastructure Reference (VM)

- Ollama source: `/home/llm/ollama-src/`
- Build dir: `/home/llm/ollama-hip-build/`
- Installed lib: `/usr/local/lib/ollama/rocm/libggml-hip.so`
- Tensile files: `/usr/local/lib/ollama/rocm/rocblas/library/` (156 gfx906 files from rocBLAS 6.4.3)
- Service override: `/etc/systemd/system/ollama.service.d/rocm.conf`
- GPU watchdog: `/usr/local/bin/gpu-watchdog.sh` + `/etc/systemd/system/gpu-watchdog.service`
- Benchmark script: `/usr/local/bin/gpu-bench.sh`
