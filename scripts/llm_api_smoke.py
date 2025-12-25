#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的 LLM API 手动测试脚本（不参与 pytest 默认测试集）。

说明：
- 真实 API 调用会消耗额度/受限于 provider 速率，因此不应放在 CI/pytest 默认路径里。
- 这里直接复用 `intent_ir.llm` 的 provider/fallback/cache 逻辑。
"""

import json

from intent_ir.llm import DEFAULT_MODEL, chat_completion


def run_api_smoke(*, model: str = DEFAULT_MODEL) -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you? Reply in one short sentence."},
    ]

    print("=" * 60)
    print("开始测试 API...")
    print(f"Model: {model}")
    print("=" * 60)

    try:
        print("\n发送请求中...")
        resp = chat_completion(messages, model=model, timeout=180, temperature=0.2, max_tokens=256, use_cache=False)
        raw = resp.raw
        print("\n响应内容:")
        print(json.dumps(raw, indent=2, ensure_ascii=False)[:4000])
        print("\n" + "=" * 60)
        print("助手回复:")
        print(resp.first_message())
        print("=" * 60)
    except Exception as e:
        print(f"❌ 调用失败: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    run_api_smoke()
