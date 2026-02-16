using namespace intentir_cuda_codegen;
#ifdef INTENTIR_CUDA_CODEGEN_PYBIND
namespace py = pybind11;

static std::string lower_from_json_str(const std::string& intent_json, const std::string& bindings_json) {
  Intent intent = parse_intent(json::parse(intent_json));
  json bindings = json::parse(bindings_json);
  json out = lower_intent_to_cuda(intent, bindings);
  return out.dump();
}

#ifndef INTENTIR_CUDA_CODEGEN_PYBIND_MODULE_NAME
#define INTENTIR_CUDA_CODEGEN_PYBIND_MODULE_NAME intentir_cuda_codegen_ext
#endif

PYBIND11_MODULE(INTENTIR_CUDA_CODEGEN_PYBIND_MODULE_NAME, m) {
  m.doc() = "IntentIR CUDA C++ codegen (pybind11 wrapper)";
  m.def("lower_from_json_str", &lower_from_json_str, py::arg("intent_json"), py::arg("bindings_json"));
}
#endif

