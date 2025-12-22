# LLM 调用与 Intent 提取速记

- 推荐配置：在本地创建 `intent_ir/llm_providers.local.json`（已在 `.gitignore` 里，避免把 key 提交到仓库）。
  - 参考模板：`intent_ir/llm_providers.example.json`
- 可选环境变量（用于 CI）：`INTENTIR_<MODEL>_API_KEY` / `INTENTIR_<MODEL>_BASE_URL`
  - 例如：`INTENTIR_CLAUDE_SONNET_4_5_API_KEY`（把 `-`/`.` 替换成 `_`，并大写）
- 统一入口：`intent_ir.llm_client.chat_completion`（底层 API 封装）与 `intent_ir.llm_intent.extract_intent_json`（从 Triton 源码提取 Intent-IR v1.1 候选 JSON）。
- 示例（使用本地 providers 文件）：
  ```bash
  python -c "from intent_ir.llm_intent import extract_intent_json; print(extract_intent_json(open('kernel.py').read()))"
  ```
- 返回结果可直接交给 Task2 的 parser/validator；后续 pass 管线（Task4.5）负责补齐 broadcast/epilogue 规范化。
- 若在受限网络环境下运行，保持离线模式并仅调用解析/验证逻辑；需要真实提取时再开启网络与密钥。
