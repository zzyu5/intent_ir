# ORG Row Kernel Audit — 2026-03-09

This audit checks whether the near-`1.00x` results for `row_sum`, `row_max`, and `layer_norm_persistent` come from real `R x H -> P` planning or from backend degeneration into a Triton copy.

## Kernels

### `row_sum`
- Guided perf: `qps_intentir=309041.6264`, `qps_native=308490.5060`, `ratio=1.0017865069`
- Guided candidate: `row_sum_axis1_v2:ROW_REDUCE_BLOCK_THREADS=128,ROW_REDUCE_SHARED_STAGE=1,ROW_REDUCE_VECTOR_WIDTH=2`
- Native source default: `BLOCK_N=min(1024,next_power_of_two(N))`; for `N=256`, source-level `BLOCK_N=256`
- Native launch shape from `ncu`: block `(128,1,1)`, grid `(4,1,1)`
- Guided launch shape from `ncu`: block `(128,1,1)`, grid `(4,1,1)`

#### PTX / SASS differences
- Native PTX has `ld.global.v2.b32`; guided PTX has two scalar `ld.global.f32`
- Native PTX has 2 `ld.shared`, 2 `st.shared`, 2 `bar.sync`; guided PTX has 5 `ld.shared`, 4 `st.shared`, 3 `bar.sync`
- Native cubin resource usage: `REG=15`, `SHARED=1024`
- Guided cubin resource usage: `REG=17`, `SHARED=2048`
- Native SASS top ops: `FADD(8)`, `SHFL(7)`, `BAR(2)`
- Guided SASS top ops: `FADD(5)`, `SHFL(5)`, `BAR(3)`, more `MOV/LEA`

#### Bandwidth
- Native median `dram__throughput.avg.pct_of_peak_sustained_elapsed = 0.14%`
- Guided median `dram__throughput.avg.pct_of_peak_sustained_elapsed = 0.17%`
- Native median `dram__bytes.sum = 7936`
- Guided median `dram__bytes.sum = 7680`

#### Interpretation
- Guided is not assembly-identical to Triton native.
- The kernel is nowhere near DRAM roofline. This is not "both hit peak bandwidth"; it is a tiny row reduction in the same latency regime.
- Guided wins slightly because it lands in a similar effective decomposition with marginally lower measured runtime, not because the PTX is copied.

## `row_max`

- Guided perf: `qps_intentir=243238.6666`, `qps_native=239950.9374`, `ratio=1.0137016728`
- Guided candidate: `row_max_axis1_v2:ROW_REDUCE_BLOCK_THREADS=128,ROW_REDUCE_SHARED_STAGE=1,ROW_REDUCE_VECTOR_WIDTH=2`
- Native source default: `BLOCK_N=min(1024,next_power_of_two(N))`; for `N=256`, source-level `BLOCK_N=256`
- Native launch shape from `ncu`: block `(128,1,1)`, grid `(4,1,1)`
- Guided launch shape from `ncu`: block `(128,1,1)`, grid `(4,1,1)`

#### PTX / SASS differences
- Native PTX has `ld.global.v2.b32`; guided PTX has two scalar `ld.global.f32`
- Native PTX has 8 `max.f32`; guided PTX lowers max path differently and does not match native instruction arrangement
- Native cubin resource usage: `REG=15`, `SHARED=1024`
- Guided cubin resource usage: `REG=17`, `SHARED=2048`
- Native SASS top ops: `FMNMX(8)`, `SHFL(7)`, `BAR(2)`
- Guided SASS top ops: `FMNMX(5)`, `SHFL(5)`, `BAR(3)`, more `MOV/STS`

#### Bandwidth
- Native median `dram__throughput.avg.pct_of_peak_sustained_elapsed = 0.14%`
- Guided median `dram__throughput.avg.pct_of_peak_sustained_elapsed = 0.17%`
- Native median `dram__bytes.sum = 7936`
- Guided median `dram__bytes.sum = 7680`

#### Interpretation
- Again, not assembly-identical.
- Also nowhere near memory peak.
- The near-`1.0x` result comes from converging to a comparable row-reduction execution shape, not from copying Triton PTX.

## `layer_norm_persistent`

- Guided perf: `qps_intentir=220944.2832`, `qps_native=219490.9721`, `ratio=1.0066212799`
- Guided candidate: `layer_norm_axis1_v1:LAYER_NORM_BLOCK_THREADS=32,LAYER_NORM_PERSISTENT_ROW=1,LAYER_NORM_VECTOR_WIDTH=2`
- Native path: `kernel_adapter:layer_norm_persistent`, source uses Triton autotune configs via `runtime.get_tuned_config("layer_norm_persistent")`
- Native launch shape from `ncu`: block `(32,4,1)`, grid `(4,1,1)`
- Guided launch shape from `ncu`: block `(32,1,1)`, grid `(4,1,1)`

#### PTX / SASS differences
- Native PTX has:
  - 6 `ld.global.v4.b32`
  - 2 `st.global.v4.b32`
  - 11 `fma.rn.f32`
  - 14 `shfl.sync`
  - 5 `bar.sync`
- Guided PTX has:
  - no `ld.global.v4.b32`
  - no `st.global.v4.b32`
  - no explicit `fma.rn.f32`
  - 12 `bar.sync`
  - far more scalar shared-memory steps
- Native cubin resource usage: `REG=40`, `SHARED=1024`
- Guided cubin resource usage: `REG=20`, `SHARED=2048`
- Native SASS top ops: `FADD(33)`, `FMUL(14)`, `FFMA(12)`, `SHFL(14)`, `BAR(5)`
- Guided SASS top ops: `BAR(12)`, `IADD(9)`, `SHF(9)`, `LDG(6)`, `FMUL(7)`, `FADD(5)`

#### Bandwidth
- Native median `dram__throughput.avg.pct_of_peak_sustained_elapsed = 0.31%`
- Guided median `dram__throughput.avg.pct_of_peak_sustained_elapsed = 0.12%`
- Native median `dram__bytes.sum = 25088`
- Guided median `dram__bytes.sum = 6400`

#### Interpretation
- This is the strongest proof that guided is not copying Triton.
- The generated PTX/SASS structure is materially different, launch shape is materially different, and resource usage is materially different.
- The near-`1.0x` result therefore does **not** mean "same code"; it means that on this small benchmark shape both implementations end up in a similar wall-clock regime despite very different instruction structure.
- This kernel is also not close to DRAM roofline.

## Planner exploration evidence

### `row_sum`
- Candidate list size: 9
- Explored:
  - `BLOCK_THREADS`: `32`, `64`, `128`
  - `VECTOR_WIDTH`: `1`, `2`, `4`
  - `SHARED_STAGE`: `0`, `1`
- Winner: `128, shared=1, vec=2`

### `row_max`
- Candidate list size: 9
- Explored:
  - `BLOCK_THREADS`: `32`, `64`, `128`
  - `VECTOR_WIDTH`: `1`, `2`, `4`
  - `SHARED_STAGE`: `0`, `1`
- Winner: `128, shared=1, vec=2`

### `layer_norm_persistent`
- Candidate list size: 9
- Explored:
  - `BLOCK_THREADS`: `32`, `64`, `128`
  - `VECTOR_WIDTH`: `1`, `2`, `4`
  - `PERSISTENT_ROW`: fixed `1`
- Winner: `32, persistent=1, vec=2`

## Bottom line

- The three near-`1.00x` kernels are **not** evidence of backend cheating by copying Triton PTX/SASS.
- They are also **not** evidence that guided has found meaningfully better hardware utilization.
- For all three kernels, measured DRAM throughput is far below peak; these runs are not saturating the 5090D memory system.
- The right scientific interpretation is:
  - `R x H -> P` is generating distinct implementations
  - those implementations happen to land in nearly the same latency regime as Triton native on these small benchmark shapes
  - the current evidence does **not** support a claim of deeper hardware-optimality for these three kernels
