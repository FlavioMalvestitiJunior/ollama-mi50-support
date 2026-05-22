# ollama-mi50-support

Custom patches and kernels to make **Ollama 0.24.0** run correctly and efficiently on **AMD Radeon Instinct MI50 32GB** (gfx906/Vega20) under **ROCm 7.2.3**.

---

## Background

ROCm 7.2.3 dropped official support for gfx906 (Vega20). There are no TensileLibrary entries for gfx906, so **rocBLAS is unavailable** for that architecture. Ollama's GGML HIP backend falls back to its own CUDA-style kernels (MMVQ, mulmat, etc.) — but those kernels were written assuming the CUDA warp size (32 threads), while the physical wavefront on gfx906 is **64 threads**.

This mismatch caused a silent correctness bug where only **62% of Q4_K quantization blocks** were processed during decode. Models still produced coherent-looking text (RMSNorm in the transformer compensates for missing activations), making the bug very hard to detect without direct measurement.

---

## Problems Identified and Fixed

### 1. GCN/GENERIC dispatch mismatch in `mmvq.cu` (correctness bug)

**Root cause:**  
The MMVQ kernel is compiled without `-DGCN`, so the device code uses `GENERIC` parameters (`nwarps=4`, warp_size=32, 128 threads).  
But `get_device_table_id(cc)` on the host detects gfx906 and selects `GCN` parameters (`nwarps=2`), launching only **128 threads** (2 warps × 64 physical threads).  

With 128 threads and `nwarps=2` from the host perspective:
```
blocks_per_iter = vdr * nwarps * warp_size / qi = 2 * 2 * 64 / 32 = 8
```
For gemma4:31b with `ne00=5376`: `blocks_per_row = 5376/256 = 21 blocks`  
With `blocks_per_iter=8`: only iterations covering kbx in {0..7, 8..15} are launched — **blocks 16..20 are never processed (24% missing, 62% coverage).**

**Fix:**  
Implemented a dedicated `mul_mat_vec_q4K_vega20` kernel that:
- Uses 256 threads (4 wavefronts x 64 threads) — explicit and unambiguous
- Stages the entire Q4_K row into LDS (Local Data Share) via `uint4` (128-bit) coalesced loads
- Uses `blocks_per_iter = 2 * 4 * 64 / 32 = 16`, covering all 21 blocks in 2 iterations
- Computes cross-warp reduction manually (no warp shuffle ambiguity)

This kernel is dispatched only for `type == GGML_TYPE_Q4_K && cc == GGML_CUDA_CC_VEGA20 && !has_fusion`.

### 2. `out-prod.cu` compilation errors

**Root cause:**  
A previous patch added a Vega20 branch inside `ggml_cuda_out_prod` that used variables declared later in the function (`dps2`, `dps3`, `s02`, `s03`, `s12`, `s13`, `s2`, `s3`, `src1_T`) and called `vega20_gemm_tiled` from `ggml-cuda.cu` — a different translation unit, not visible to `out-prod.cu`.

**Fix:**  
- Moved all variable declarations before the Vega20 branch
- Defined a local `vega20_out_prod_gemm` template kernel inside `out-prod.cu` (simple correctness path — out-product is not performance-critical)

### 3. `ggml-cuda.cu` — rocBLAS bypass for gfx906

- Added `vega20_gemm_tiled<T0,T1>` — tiled GEMM kernel for F16/F32 (shared memory 16x16 tiles)
- Added `vega20_gemm_batched_tiled<T0,T1>` — batched version (3D grid)
- Added `ggml_cuda_mul_mat_batched_vega20()` — dispatch for batched F16/F32 operations
- Set `no_cublas = (cc == GGML_CUDA_CC_VEGA20)` in `ggml_cuda_mul_mat` — avoids calling rocBLAS (which has no gfx906 TensileLibrary and would crash)
- Skip `cublas_handle()` in `ggml_backend_cuda_graph_reserve` for gfx906

### 4. `solve_tri.cu` — triangular solve for gfx906

- Added `solve_tri_vega20` kernel for forward substitution when problem dims exceed the fast kernel's range (`n>64` or `k>32`)
- Dispatched as: if `cc==VEGA20` and dims outside fast path, use custom kernel instead of `cublasStrsmBatched`

---

## Performance Results

Hardware: AMD Radeon Instinct MI50 32GB (gfx906, PCIe Gen3), ROCm 7.2.3  
Model: gemma4:31b (61 layers fully on GPU)

| State | t/s | Coverage | HBM2 bandwidth used | Correctness |
|---|---|---|---|---|
| Buggy baseline (GCN dispatch mismatch) | 13.37 | 62% | ~130 GB/s (14.4%) | Wrong |
| After fix, LDS 128-thread uint32 | 11.15 | 100% | ~175 GB/s | Correct |
| After fix, LDS 256-thread uint4 | **12.82** | 100% | ~201 GB/s (22.3%) | **Correct** |

**Why fixing correctness reduced throughput:**  
Processing 61% more Q4_K blocks per token means ~61% more memory bandwidth needed. Even with the optimized LDS kernel, the increase in correct compute slightly lowers throughput compared to the buggy (incomplete) baseline. The bandwidth efficiency more than doubles (14.4% to 22.3% of theoretical 900 GB/s).

Other models on GPU:
| Model | Layers GPU | Speed |
|---|---|---|
| llama3.2:3b | full | ~85 t/s |
| qwen3.6:35b (MoE) | 41/41 | ~31 t/s |
| gemma4:31b (dense) | 61/61 | ~12.82 t/s |

---

## Files in this Repository

| File | Source path on VM | Purpose |
|---|---|---|
| `mmvq.cu` | `.../ggml-cuda/mmvq.cu` | MMVQ kernel — main fix (LDS-staged Q4_K kernel for gfx906) |
| `out-prod.cu` | `.../ggml-cuda/out-prod.cu` | Fixed outer-product — variable order + local kernel |
| `ggml-cuda.cu` | `.../ggml-cuda/ggml-cuda.cu` | rocBLAS bypass, vega20 GEMM kernels, cublas skip |
| `solve_tri.cu` | `.../ggml-cuda/solve_tri.cu` | Triangular solve bypass for gfx906 |
| `CMakeLists.txt` | `/home/llm/ollama-hip-build/CMakeLists.txt` | Build file with all compile definitions |
| `ollama-rocm.conf` | `/etc/systemd/system/ollama.service.d/rocm.conf` | Ollama service environment configuration |

All source files are from the Ollama 0.24.0 fork at `/home/llm/ollama-src/`.

---

## Requirements

- **GPU**: AMD Radeon Instinct MI50 (gfx906 / Vega20). May also apply to other gfx906 variants (RX Vega 56/64) with testing.
- **ROCm**: 7.2.3 (tested). ROCm 6.x may work too; verify HIP API compatibility.
- **Ollama**: 0.24.0 (source build required — this patches the GGML HIP backend)
- **Build tools**: cmake >= 3.21, hipcc (from ROCm), make
- **OS**: Linux (tested on Ubuntu 22.04 inside Proxmox VM)

---

## How to Apply

### 1. Clone Ollama source

```bash
git clone https://github.com/ollama/ollama.git ollama-src
cd ollama-src
git checkout v0.24.0
```

### 2. Copy patched files

```bash
SRC_DIR=ollama-src/ml/backend/ggml/ggml/src/ggml-cuda

cp mmvq.cu      $SRC_DIR/mmvq.cu
cp out-prod.cu  $SRC_DIR/out-prod.cu
cp ggml-cuda.cu $SRC_DIR/ggml-cuda.cu
cp solve_tri.cu $SRC_DIR/solve_tri.cu
```

### 3. Create build directory

```bash
mkdir -p ollama-hip-build
cp CMakeLists.txt ollama-hip-build/
cd ollama-hip-build
```

### 4. Configure with CMake

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS=gfx906 \
  -DGGML_HIP=ON
```

Or use the included `CMakeLists.txt` directly if it already contains the right source paths.

### 5. Build

```bash
cmake --build . --target ggml-hip -j$(nproc)
```

Expected output:
```
[  1%] Building HIP object CMakeFiles/ggml-hip.dir/.../ggml-cuda.cu.o
[  2%] Linking HIP shared module libggml-hip.so
[100%] Built target ggml-hip
```

### 6. Install

```bash
sudo cp libggml-hip.so /usr/local/lib/ollama/rocm/libggml-hip.so
sudo systemctl restart ollama
```

### 7. Configure Ollama service

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

`HSA_OVERRIDE_GFX_VERSION=9.0.6` is required to tell ROCm's HSA runtime to treat the GPU as gfx906 (it may report a different version string on some driver versions).

### 8. Verify GPU inference

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

- **Fusion path (has_fusion=true)**: The LDS kernel fix only applies to non-fused Q4_K decode. When `has_fusion=true` (SwiGLU/GeGLU gate+up projections), the original GCN dispatch is still used — this path retains the 62% coverage bug but is stable. Attempting to change its dispatch causes a HIP graph capture crash (2513-node graph) that has not been resolved.

- **rocBLAS not available for gfx906 on ROCm 7.2.3**: There is no TensileLibrary for gfx906 in this ROCm version. All GEMM is done via custom kernels. Performance for large batch sizes may be lower than on officially supported architectures.

- **GPU utilization monitoring**: `rocm-smi --showuse` does not report correctly for gfx906. Use `/sys/class/drm/card*/device/gpu_busy_percent` instead.

---

## Architecture Notes

**Why LDS staging?**  
Each Q4_K block is 144 bytes (2B scale `d` + 2B `dmin` + 12B scales + 128B `qs`). Without staging, the original kernel performs multiple passes over HBM2 with uncoalesced 32-bit loads. Staging the entire row into LDS (Local Data Share) via 128-bit `uint4` loads requires only one pass with 256 threads, saturating HBM2 bandwidth more efficiently.

**Why 256 threads (4 wavefronts)?**  
gfx906 CUs have 40 wavefront slots. With occupancy hint `__launch_bounds__(256, 10)`, 10 blocks x 4 wavefronts = 40 wavefronts — maximum theoretical occupancy. Each block processes one output row independently.

**Why not use `-DGCN`?**  
Adding `-DGCN` as a compile definition would select 64-thread warp_size in the device code — this would fix the MMVQ dispatch without a custom kernel. However, it requires modifying `CMakeLists.txt` and affects all kernels (some may break with 64-thread assumptions). The targeted per-kernel fix is safer.

**Block size math for gemma4:31b:**
- `ne00 = 5376` (hidden dimension)
- `blocks_per_row = 5376 / 256 = 21`
- LDS kernel: `blocks_per_iter = 2 * 4 * 64 / 32 = 16` — covers 21 blocks in 2 iterations (0..15, 16..20)
- Buggy GCN dispatch: `blocks_per_iter = 2 * 2 * 64 / 32 = 8` — covers only 16 of 21 blocks

---

## Infrastructure Reference (VM)

- Ollama source: `/home/llm/ollama-src/`
- Build dir: `/home/llm/ollama-hip-build/`
- Installed lib: `/usr/local/lib/ollama/rocm/libggml-hip.so`
- Service override: `/etc/systemd/system/ollama.service.d/rocm.conf`
