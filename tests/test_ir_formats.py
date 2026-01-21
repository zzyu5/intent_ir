import json

from verify.ir_formats import validate_mlir_linalg_text, validate_tile_dsl_json


def test_validate_mlir_linalg_text_accepts_simple_module():
    txt = """
module {
  func.func @k(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}
""".strip()
    assert validate_mlir_linalg_text(txt) == []


def test_validate_mlir_linalg_text_rejects_missing_linalg():
    txt = "module { func.func @k() { return } }"
    errs = validate_mlir_linalg_text(txt)
    assert any("linalg" in e for e in errs)


def test_validate_mlir_linalg_text_rejects_missing_indexing_maps():
    txt = """
module {
  func.func @k(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}
""".strip()
    errs = validate_mlir_linalg_text(txt)
    assert any("indexing_maps" in e for e in errs)


def test_validate_tile_dsl_json_accepts_minimal_shape():
    obj = json.loads(
        """
        {
          "schema_version": "tile_dsl_v0",
          "kernel": "matmul",
          "schedule": {
            "tile": {"M": 64, "N": 64, "K": 16},
            "vec_width": 8,
            "num_threads": 4
          }
        }
        """
    )
    assert validate_tile_dsl_json(obj) == []


def test_validate_tile_dsl_json_checks_axes_against_io_spec():
    obj = {"schema_version": "tile_dsl_v0", "kernel": "k", "schedule": {"tile": {"X": 16}}}
    io_spec = {"tensors": {"A": {"dtype": "f32", "shape": ["M", "N"]}}}
    errs = validate_tile_dsl_json(obj, io_spec=io_spec)
    assert errs


def test_validate_tile_dsl_json_rejects_bad_tile():
    obj = {"schema_version": "tile_dsl_v0", "kernel": "k", "schedule": {"tile": {"M": 0}}}
    errs = validate_tile_dsl_json(obj)
    assert errs
