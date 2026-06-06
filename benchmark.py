#!/usr/bin/env python3
"""
benchmark.py — LLM量子化比較実験 ベンチマークスクリプト (Machine A / Ubuntu)

指定モデル・量子化・コンテキスト長でテキストを推論し、
TTFT・生成速度・ピークVRAMを自動計測して results.csv に追記する。

使用例:
  python benchmark.py --model_type gguf \
      --model_path models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf \
      --task json_extract --input_file data/test_data_T1.txt \
      --context_len 8192

CSVカラム:
  timestamp, model_name, quant_type, context_len, task_type, data_id,
  ttft_sec, gen_speed_tps, peak_vram_gb, offload_layers, output_valid, output_text
"""
import argparse
import csv
import datetime
import json
import os
import re
import sys
import threading
import time

CSV_FIELDS = [
    "timestamp", "model_name", "quant_type", "context_len", "task_type",
    "data_id", "ttft_sec", "gen_speed_tps", "peak_vram_gb",
    "offload_layers", "output_valid", "output_text",
]

# ---------------------------------------------------------------------------
# プロンプトテンプレート (指示書 セクション4)
# ---------------------------------------------------------------------------
PROMPT_SUMMARY = """以下は会議のトランスクリプトです。
会議の要点を200〜300字で要約してください。

## トランスクリプト
{transcript}

## 要約
"""

PROMPT_JSON = """以下の会議トランスクリプトから、アクションアイテムを抽出してください。
必ず以下のJSONスキーマに従って出力してください。それ以外のテキストは出力しないでください。

スキーマ：
{{
  "action_items": [
    {{
      "task": "タスクの内容（文字列）",
      "assignee": "担当者名（文字列、不明な場合はnull）",
      "due_date": "期日（YYYY-MM-DD形式、不明な場合はnull）",
      "priority": "high/medium/low のいずれか"
    }}
  ]
}}

## トランスクリプト
{transcript}

## JSON出力
"""

# タスクごとの最大生成トークン数
MAX_TOKENS = {"summary": 512, "json_extract": 1536}


# ---------------------------------------------------------------------------
# VRAM サンプラ (pynvml をバックグラウンドでポーリングしてピークを記録)
# ---------------------------------------------------------------------------
class VramSampler(threading.Thread):
    def __init__(self, gpu_index=0, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._nvml = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as e:  # pragma: no cover
            print(f"[warn] pynvml 初期化失敗、VRAM計測無効: {e}", file=sys.stderr)

    def run(self):
        if not self._nvml:
            return
        while not self._stop.is_set():
            try:
                mi = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                if mi.used > self.peak_bytes:
                    self.peak_bytes = mi.used
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        try:
            self.join(timeout=2)
        except Exception:
            pass
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    @property
    def peak_gb(self):
        return round(self.peak_bytes / (1024 ** 3), 3) if self.peak_bytes else None


# ---------------------------------------------------------------------------
# 出力の妥当性チェック
# ---------------------------------------------------------------------------
def strip_json(text):
    """コードフェンスや前後のテキストを除去して最初のJSONオブジェクトを抽出。"""
    t = text.strip()
    # ```json ... ``` を除去
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    # 最初の { から対応する } までを素朴に抽出
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return None


def strip_think(text):
    """Qwen3等のthinkingブロック <think>...</think> を除去。"""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def check_valid(task_type, text):
    if task_type == "json_extract":
        cand = strip_json(text)
        if cand is None:
            return False
        try:
            json.loads(cand)
            return True
        except Exception:
            return False
    # summary: 非空かつ最低限の長さ
    return len(text.strip()) >= 10


# ---------------------------------------------------------------------------
# GGUF 推論 (llama-cpp-python)
# ---------------------------------------------------------------------------
def run_gguf(model_path, messages, context_len, max_tokens, n_gpu_layers):
    """GGUFをチャットテンプレート(create_chat_completion)で推論。
    GGUFに埋め込まれたchat_templateを自動適用する。"""
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_ctx=context_len,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    # モデルの総レイヤ数 (メタデータから best-effort)
    total_layers = None
    try:
        for k, v in (llm.metadata or {}).items():
            if k.endswith("block_count"):
                total_layers = int(v)
                break
    except Exception:
        pass

    t0 = time.time()
    first_t = None
    n_tok = 0
    chunks = []
    for ch in llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=0.0, stream=True
    ):
        piece = ch["choices"][0]["delta"].get("content")
        if piece is None:
            continue
        if first_t is None:
            first_t = time.time()
        chunks.append(piece)
        n_tok += 1
    t_end = time.time()

    text = "".join(chunks)
    ttft = round(first_t - t0, 4) if first_t else None
    gen_dur = (t_end - first_t) if first_t else None
    gen_speed = round(n_tok / gen_dur, 2) if gen_dur and gen_dur > 0 else None

    offload = f"{n_gpu_layers}" if total_layers is None else f"{n_gpu_layers}/{total_layers}"
    try:
        llm.close()
    except Exception:
        pass
    del llm
    return text, ttft, gen_speed, offload


# ---------------------------------------------------------------------------
# AWQ 推論 (vLLM) — vLLM導入後に本格対応。現状は枠のみ。
# ---------------------------------------------------------------------------
def run_awq(model_path, messages, context_len, max_tokens, gpu_mem_util=0.90):
    """AWQをvLLMでチャットテンプレート適用して推論。
    vLLMの.chat()がトークナイザのchat_templateを自動適用する。"""
    # この機は nvcc/ninja がPATHに無くFlashInferのJITコンパイルが失敗するため、
    # FlashInferサンプラーを無効化してネイティブ実装を使う。
    # またEngineCoreサブプロセスでのCUDA再初期化エラー回避に spawn を強制。
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        quantization="awq",
        max_model_len=context_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.time()
    out = llm.chat([messages], sp)
    t_end = time.time()
    text = out[0].outputs[0].text
    n_tok = len(out[0].outputs[0].token_ids)
    # オフライン推論では TTFT を厳密に取れないため None。
    ttft = None
    gen_speed = round(n_tok / (t_end - t0), 2) if (t_end - t0) > 0 else None
    del llm
    return text, ttft, gen_speed, "awq(all)"


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def derive_data_id(input_file):
    m = re.search(r"(T\d+)", os.path.basename(input_file))
    return m.group(1) if m else os.path.splitext(os.path.basename(input_file))[0]


def model_name_from_path(path):
    base = os.path.basename(path.rstrip("/"))
    return base or path


def append_csv(output_file, row):
    new = not os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def clear_gpu_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="LLM量子化比較ベンチマーク")
    ap.add_argument("--model_type", required=True, choices=["gguf", "awq"])
    ap.add_argument("--model_path", required=True, help="モデルのパスまたはHF ID")
    ap.add_argument("--task", required=True, choices=["summary", "json_extract"])
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--context_len", required=True, type=int,
                    help="標準は 8192/16384/32768。AWQの限界探索で任意値も可。")
    ap.add_argument("--output_file", default="results.csv")
    ap.add_argument("--n_gpu_layers", type=int, default=-1,
                    help="GGUF: GPUへオフロードする層数 (-1=全層)")
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90,
                    help="AWQ(vLLM): gpu_memory_utilization (0<x<=1)")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Qwen3等のthinkingを有効化 (既定: 無効=/no_think)")
    args = ap.parse_args()

    with open(args.input_file, encoding="utf-8") as f:
        transcript = f.read()

    template = PROMPT_SUMMARY if args.task == "summary" else PROMPT_JSON
    user_content = template.format(transcript=transcript)
    # Qwen3系はthinking既定オフ: ユーザーメッセージ末尾に /no_think を付与
    if (not args.enable_thinking) and ("qwen" in args.model_path.lower()):
        user_content = user_content + "\n/no_think"
    messages = [{"role": "user", "content": user_content}]
    max_tokens = args.max_tokens or MAX_TOKENS[args.task]

    quant_type = "GGUF_Q4_K_M" if args.model_type == "gguf" else "AWQ_INT4"
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name_from_path(args.model_path),
        "quant_type": quant_type,
        "context_len": args.context_len,
        "task_type": args.task,
        "data_id": derive_data_id(args.input_file),
        "ttft_sec": "", "gen_speed_tps": "", "peak_vram_gb": "",
        "offload_layers": "", "output_valid": "", "output_text": "",
    }

    # GGUFのみ: 親プロセスでtorch.cuda経由のキャッシュ解放。
    # AWQ(vLLM)はEngineCoreをサブプロセスで起動するため、親でCUDAを初期化すると
    # "Cannot re-initialize CUDA in forked subprocess" になる→AWQでは呼ばない。
    if args.model_type == "gguf":
        clear_gpu_cache()
    sampler = VramSampler()
    sampler.start()
    try:
        if args.model_type == "gguf":
            text, ttft, gen_speed, offload = run_gguf(
                args.model_path, messages, args.context_len, max_tokens,
                args.n_gpu_layers)
        else:
            text, ttft, gen_speed, offload = run_awq(
                args.model_path, messages, args.context_len, max_tokens,
                args.gpu_mem_util)
        text = strip_think(text)  # thinkingブロックを除去してから記録
        row["ttft_sec"] = ttft
        row["gen_speed_tps"] = gen_speed
        row["offload_layers"] = offload
        row["output_valid"] = check_valid(args.task, text)
        # 出力は評価(evaluate.py)で全文を要するため切り詰めない。
        # 暴走防止に上限のみ設定(json_extract=1536tok≒数千字, summary=512tokを十分に収容)。
        row["output_text"] = text[:12000]
    except Exception as e:
        row["output_valid"] = "ERROR"
        row["output_text"] = f"{type(e).__name__}: {e}"[:512]
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        sampler.stop()
        row["peak_vram_gb"] = sampler.peak_gb
        clear_gpu_cache()

    append_csv(args.output_file, row)
    print("=== 結果 ===")
    for k in CSV_FIELDS:
        v = row[k]
        if k == "output_text" and isinstance(v, str) and len(v) > 200:
            v = v[:200] + " …"
        print(f"  {k:14}: {v}")


if __name__ == "__main__":
    main()
