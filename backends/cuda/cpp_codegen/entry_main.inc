using namespace intentir_cuda_codegen;
#ifndef INTENTIR_CUDA_CODEGEN_NO_MAIN
int main(int argc, char** argv) {
  try {
    std::string intent_path;
    std::string bindings_path;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--intent" && i + 1 < argc) intent_path = argv[++i];
      else if ((a == "--bindings" || a == "--shapes") && i + 1 < argc) bindings_path = argv[++i];
      else if (a == "-h" || a == "--help") {
        std::cout << "usage: intentir_cuda_codegen --intent <intent.json> --bindings <bindings.json>\\n";
        return 0;
      } else {
        fail("unknown arg: " + a);
      }
    }
    if (intent_path.empty() || bindings_path.empty()) {
      std::cerr << "missing --intent/--bindings (use --help)\\n";
      return 2;
    }

    Intent intent = parse_intent(read_json_file(intent_path));
    json bindings_json = read_json_file(bindings_path);
    json out = lower_intent_to_cuda(intent, bindings_json);
    std::cout << out.dump() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "intentir_cuda_codegen error: " << e.what() << "\\n";
    return 3;
  }
}
#endif
