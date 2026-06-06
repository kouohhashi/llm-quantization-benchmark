## 結果サマリー

| モデル | 量子化 | ctx | tok/s | TTFT | VRAM | JSON有効率 | F1 | ROUGE-L | judge |
|--------|--------|-----|-------|------|------|------------|-----|---------|-------|
| Qwen_Qwen3-8B-Q4_K_M.gguf | GGUF_Q4_K_M | 16384 | 21.677 | 6.337 | 7.693GB | 1.0 | 0.612 | - | - |
| Qwen_Qwen3-8B-Q4_K_M.gguf | GGUF_Q4_K_M | 32768 | 7.962 | 18.447 | 7.462GB | 1.0 | 0.619 | - | - |
| Qwen_Qwen3-8B-Q4_K_M.gguf | GGUF_Q4_K_M | 8192 | 28.64 | 2.519 | 6.617GB | 1.0 | 0.615 | - | - |
| google_gemma-4-E4B-it-Q4_K_M.gguf | GGUF_Q4_K_M | 16384 | 38.492 | 2.36 | 5.107GB | 1.0 | 0.365 | - | - |
| google_gemma-4-E4B-it-Q4_K_M.gguf | GGUF_Q4_K_M | 32768 | 35.825 | 4.181 | 6.491GB | 1.0 | 0.343 | - | - |
| google_gemma-4-E4B-it-Q4_K_M.gguf | GGUF_Q4_K_M | 8192 | 41.015 | 1.307 | 4.502GB | 1.0 | 0.25 | - | - |
