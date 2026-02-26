# SCRIPTS_TESTS_GOVERNANCE

## 1. scripts 治理状态

- scripts catalog schema: `intentir_scripts_catalog_v1`
- entries 总数: `39`
- active 数量: `39`
- deprecated 数量: `0`

### 按 owner_lane 分布

- `backend_compiler_rewrite`: 10
- `coverage_integrity`: 9
- `ir_arch_quality`: 5
- `mlir_migration`: 1
- `workflow`: 14

### 目录扫描与 catalog 对齐

- catalog 未覆盖的 `.py` 脚本: `0`
- catalog 声明但文件不存在: `1`
  - unknown_value: `scripts/rvv_probe.sh`

## 2. tests 治理状态

- tests catalog schema: `intentir_tests_catalog_v1`
- entries 总数: `27`

### 按 track 分布

- `backend_compiler`: 4
- `backend_contract`: 2
- `cli`: 1
- `core_ir`: 4
- `coverage_integrity`: 2
- `frontend_triton`: 3
- `ir_arch_quality`: 4
- `workflow`: 6
- `workflow_contract`: 1

### 目录扫描与 catalog 对齐

- catalog 未覆盖的 `test_*.py`: `3`
  - unknown_value: `tests/backends/test_mlir_contract_cutover.py`
  - unknown_value: `tests/intentir/test_mlir_migration.py`
  - unknown_value: `tests/intentir/test_mlir_source_scripts.py`
- catalog 声明但文件不存在: `0`

## 3. 结论

- scripts 层面覆盖基本完整；仅 `scripts/rvv_probe.sh` 属于 catalog 声明但非 `.py` 入口。
- tests 层面存在 3 个未入 catalog 的测试文件，建议在后续治理中补登记或明确归档。