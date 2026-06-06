# 実験進捗ログ — LLM量子化比較実験（Machine A / Ubuntu）

最終更新：2026-06-06（GGUF 2モデル 各12試行＝24試行 完了＋AWQ 8GB限界実測 完了）

## ★最新サマリー（GGUF Q4_K_M, RTX4060 Laptop 8GB）
集計: `results/summary_report.md` / `.csv`、生データ `results/results.csv`（GGUF24行・全valid／AWQ3行・全ERROR）。

**Qwen3-8B-Q4_K_M（36層, 重5.0GB）**
| ctx | tok/s | TTFT | VRAM | JSON有効率 | action_F1 | offload |
|----|------|------|------|-----------|-----------|---------|
| 8192  | 28.6 | 2.5s | 6.62GB | 1.0 | 0.615 | -1/36(全層) |
| 16384 | 21.7 | 6.3s | 7.69GB | 1.0 | 0.612 | 33/36(部分) |
| 32768 | 8.0  | 18.4s| 7.46GB | 1.0 | 0.619 | 18/36(部分) |

**Gemma4-E4B-it-Q4_K_M（42層, 重5.4GBだが実効4B級でKV小）**
| ctx | tok/s | TTFT | VRAM | JSON有効率 | action_F1 | offload |
|----|------|------|------|-----------|-----------|---------|
| 8192  | 41.0 | 1.3s | 4.50GB | 1.0 | 0.250 | -1/42(全層) |
| 16384 | 38.5 | 2.4s | 5.11GB | 1.0 | 0.365 | -1/42(全層) |
| 32768 | 35.8 | 4.2s | 6.49GB | 1.0 | 0.343 | -1/42(全層) |

**主要な知見（論文の核）**
- **速度 vs 品質のトレードオフが鮮明**: Gemma4-E4Bは全コンテキストで35〜41 tps＆VRAM 4.5〜6.5GBと軽快（32Kでも全層GPU・速度ほぼ不変）。一方JSON抽出のF1は0.25〜0.37と低い。Qwen3-8BはF1 0.61と高品質だが、8GBでは16K/32Kで部分オフロードを強いられ32Kは8 tpsまで失速。
- **「8GBの壁」はモデル依存**: Qwenはフル(-1)だと16K/32KでKVキャッシュ超過→`llama_context`生成失敗。ctx別部分オフロード（16K=33/32K=18層）が必須で、その分CPU↔GPU転送が律速。Gemma4-E4BはKVが小さく全コンテキスト全層GPUで収まる。→ run_experiments.sh は `ngl_for_ctx()`＋`NGL_OVERRIDE` でモデル別に切替（Gemmaは `NGL_OVERRIDE=-1`）。
- **Gemmaの低F1の内訳**: assignee/due_date/priority精度は全て1.0なのにF1が低い＝抽出した項目の属性は正確だが**項目の取りこぼし（低recall）**が主因。要・出力の定性確認。
- 修正済バグ: benchmark.py の出力512字切り詰め→12000字（JSON全文が保存されずF1評価が壊れていた）。

## ★AWQ（Qwen3-8B-AWQ / vLLM）8GB限界 実測（2026-06-06）
詳細・ログ: `results/awq/AWQ_limit_summary.md`＋`results/awq/*.log`。生CSV: `results/awq/awq_probe.csv`。
- **結論: AWQ INT4 は 8GB では実験最小条件 8K すら起動不可**。4bit重みでもGPU上で**5.71GiB**占有し、
  vLLMオーバーヘッド差引後KVに回せるのは最大**0.73GiB(util=0.97)≒5,280トークン**。8K(必要KV1.12GiB)に届かない。
- util スイープ(@8192): 0.90→KV0.19/推定最大長1392、0.93→0.42/3056、0.95→0.57/4160、0.97→0.73/5280。全て起動失敗(ValueError)。peak VRAMは常に7.98GiB（8GB上限に張り付き）。
- 必要KVはctx比例: 8K=1.12 / 16K=2.25 / 32K=4.5 GiB → 確保0.73GiBで全条件不足。
- **参考: AWQ自体は正常動作**。縮小条件 ctx=4096/util=0.97 では起動成功・推論完了（22.67 tps・有効JSON）。純粋なメモリ容量の壁であることを確認。
- 実験条件(8K/16K/32K)は results/results.csv に AWQ_INT4 の **ERROR行**として正式収録（起動失敗＝OOM相当）。
- 環境ノウハウ（benchmark.py run_awqに組込済）: ①AWQ時は親でtorch.cuda不使用＋`VLLM_WORKER_MULTIPROC_METHOD=spawn`（fork時のCUDA再初期化回避）。②`VLLM_USE_FLASHINFER_SAMPLER=0`（ninja/nvcc不要のネイティブサンプラー）。
- **論文的含意**: 「同一8GBでGGUFは全コンテキスト実行可、AWQ(vLLM)は最小条件すら起動不可」＝量子化方式×ランタイムのメモリ効率差が実用可否を分ける明確な事例。

担当機：Ubuntu（RTX4060 Laptop, VRAM 8GB）= 指示書の **Machine A**
親仕様：[`../experiment_instructions.md`](../experiment_instructions.md)

---

## 1. この実験で何をやろうとしているか

**リサーチクエスチョン**：同一モデルを GGUF・AWQ・MLX で量子化したとき、
日本語ビジネス文書タスク（会議録の要約・JSON抽出）で
**速度・品質・メモリ使用量**にどんな差が出るか。

**Machine A（この機）の担当範囲**：
- 量子化方式：**GGUF Q4_K_M** と **AWQ INT4**
- モデル：**Qwen3-8B**、**Gemma4-E4B**
- コンテキスト長 8K / 16K / 32K を変数とし、VRAMオーバーフロー時の性能劣化も実測
- ※ MLX と大型モデル（Qwen3-32B 等）は Machine B（Mac mini, 後日）

**ユーザーが決めた進行順**：
1. テストデータ T1–T3・正解データ生成 ← ✅ 完了
2. GGUF で全条件を回して results.csv にデータ蓄積（最優先）← ✅ 完了（2モデル24試行）
3. AWQ（vLLM）の8GB起動可否＝「限界の定量化」← ✅ 完了（最小条件8Kすら起動不可と確定）
4. 残: 参照要約（ROUGE-L）／Machine B（MLX）

---

## 2. 環境（構築済み・すべてユーザー領域、sudo不要）

| 区分 | 内容 |
|------|------|
| GPU | NVIDIA RTX 4060 Laptop 8GB / Driver 580 / CUDA toolkit 12.8（`/usr/local/cuda-12.8`） |
| Python | `uv` で 3.11.15 → venv は `~/paper1/.venv`（pip非同梱、`uv pip install` を使用） |
| GGUF ランタイム | **llama-cpp-python 0.3.26**（CUDA cu124 プリビルド wheel）／GPUオフロード検証済 |
| AWQ ランタイム | **vLLM 0.22.1**（import検証済・未実走） |
| 計測 | pynvml / psutil / gputil / rouge-score |
| Ollama | 導入済（既存 `gemma4:e2b`。指示書のE4Bとは別物なので実験には未使用） |

**ディスクがタイト**（空き約30GB）→ モデルは使う都度DL・終わったら削除する運用。

---

## 3. 成果物（リポジトリ内ファイル）

```
paper1/
├── experiment_instructions.md      # 親仕様（ユーザー提供）
├── benchmark.py                    # 推論＋計測→results.csv 追記
├── evaluate.py                     # F1/JSON有効率/ROUGE-L/集計レポート
├── run_experiments.sh              # GGUF全条件ランナー（12試行/モデル）
├── models/gguf/
│   ├── Qwen_Qwen3-8B-Q4_K_M.gguf            # 5.0GB, DL済
│   └── google_gemma-4-E4B-it-Q4_K_M.gguf   # 5.4GB, DL済（bartowski/google_gemma-4-E4B-it-GGUF）
├── data/
│   ├── smoke_test.txt              # スモーク用の短い会議録
│   ├── test_data_T1.txt            # 週次MTG  2,591 tok / AI 7件
│   ├── test_data_T2.txt            # 月次レビュー 6,515 tok / AI 12件
│   └── test_data_T3.txt            # 四半期計画 12,681 tok / AI 20件
├── ground_truth_T1.json            # 正解アクションアイテム（F1計算用）
├── ground_truth_T2.json
├── ground_truth_T3.json
├── results/
│   ├── results.csv                 # 生データ（24行＝2モデル×12試行・全valid）
│   ├── summary_report.{md,csv}     # 集計レポート
│   └── _scratch/                   # 使い捨て（探索ログ・切詰め版CSV等）
└── docs/progress.md                # 本ファイル
```

### run_experiments.sh のオフロード方針（モデル別）
- 既定は Qwen3-8B 向けに `ngl_for_ctx()` が ctx別の安全上限（8K=-1, 16K=33, 32K=18層）を返す。
- 環境変数 `NGL_OVERRIDE` を設定すると全コンテキストでその値を強制。Gemma4-E4Bは全層GPUで収まるため
  `NGL_OVERRIDE=-1 ./run_experiments.sh gguf <path> results/results.csv` で実行した。

### benchmark.py の主な仕様
- `--model_type gguf|awq --model_path … --task summary|json_extract --input_file … --context_len 8192|16384|32768 [--output_file results.csv] [--enable_thinking]`
- **チャットテンプレート方式**（`create_chat_completion` / vLLM `.chat()`）でinstructモデルを正しく評価
- Qwen3 の thinking は既定オフ（`/no_think` 付与）＋ `<think>` ブロックは除去
- 計測：TTFT（最初のトークンまで）、生成速度 tok/s、ピークVRAM（pynvmlでバックグラウンドサンプリング）、オフロード層数、JSON有効性
- CSV列：`timestamp, model_name, quant_type, context_len, task_type, data_id, ttft_sec, gen_speed_tps, peak_vram_gb, offload_layers, output_valid, output_text`
- エラー時も内容を記録して継続

### テストデータ／正解データ
- 完全な架空データ。Qwenトークナイザで実トークン数を測定しサイズ調整。
- アクションアイテムは「担当者・期日が明確なもの／曖昧（未定）なもの」を混在。
- 正解の `task` は部分一致で照合する想定（表現揺れを吸収）。

---

## 4. これまでに確認できたこと（確定結果）

- **GGUF 2モデル × 全条件が完走**（Qwen3-8B / Gemma4-E4B、各12試行＝計24試行、全valid）。
  数値は上部★サマリー、生データ `results/results.csv`、集計 `results/summary_report.{md,csv}`。
- **chatテンプレ適用で品質改善**：生補完では末尾に解説文が付き日付の年も誤った（2023）が、
  chatテンプレ＋/no_think では純粋なJSONのみを出力し、日付も正しく2026年に。
- **「8GBの壁」はモデル依存**：Qwen3-8Bはフルオフロード(-1)だと16K/32KでKVキャッシュ超過→
  `llama_context`生成失敗→ctx別部分オフロード(16K=33/32K=18層)が必須で32Kは8 tpsまで失速。
  Gemma4-E4BはKVが小さく全コンテキスト全層GPUで収まり35〜41 tpsを維持。
- **速度/メモリ vs 品質のトレードオフ**：Gemmaは軽快だが抽出F1が0.25〜0.37と低い（属性精度は1.0＝低recall）。
  QwenはF1≈0.61と高品質。JSON有効率は両モデル全条件で1.0。
- **AWQ(vLLM)は8GBで起動不可**：4bit重み5.71GiB＋オーバーヘッドでKVが0.73GiB止まり（≒5,280tok）、
  最小条件8K(必要KV1.12GiB)に届かない。縮小4096では起動成功(22.67tps)＝純粋なメモリ容量の壁。詳細は★AWQ節。

---

## 5. 今やっていること（現在地）

- **GGUF 2モデルの本実験は完了**（Qwen3-8B / Gemma4-E4B 各12試行＝計24試行、全valid）。
  結果は上部★サマリーと `results/summary_report.md` 参照。
  - 各モデルの有効な (データ×コンテキスト)：T1{8K,16K,32K}・T2{16K,32K}・T3{32K} = 6 条件
  - × 2タスク（summary / json_extract）= 12 試行 / モデル
- **AWQ（vLLM）の8GB限界実測も完了**（8K最小条件すら起動不可と確定。上部★AWQ節参照）。
- 残る作業は **参照要約の作成（ROUGE-L用）** と **Machine B（MLX）**（§6）。

---

## 6. 次にやること

1. ~~**Qwen3-8B GGUF 12試行**~~ ✅ **完了（2026-06-06）**。results.csv に12行、全valid。集計済（上部サマリー参照）。
   - フルオフロード(-1)では16K/32KでKVキャッシュ超過→`llama_context`生成失敗。ctx別の部分オフロード（16K=33層, 32K=18層）で全条件成功。`run_experiments.sh`の`ngl_for_ctx()`に確定。
2. ~~**Gemma4-E4B GGUF** をDL → 同じ12試行~~ ✅ **完了（2026-06-06）**。全層GPU(-1)で12試行・全valid、results.csv に追記済。
3. **参照要約 `reference_summary_T*.txt`** を作成（要約タスクのROUGE-L評価に必要）。← 未着手（別途検討中）
4. ~~`evaluate.py` で集計~~ ✅ 24試行を合算集計済（`results/summary_report.{md,csv}`）。
5. ~~**AWQ（vLLM）**：8GBで起動できるか限界実測~~ ✅ **完了（2026-06-06）**。
   結論: 8K最小条件すら起動不可（KV不足）。utilスイープ・縮小条件成功・ERROR正式収録まで記録。詳細は上記★AWQ節 / `results/awq/AWQ_limit_summary.md`。
6. （後日）Machine B（Mac mini）で MLX・大型モデル。

---

## 6.5 次回の再開手順（そのまま続けるために）

GGUF 2モデル＋AWQ限界実測まで完了済。残るは **参照要約(ROUGE-L)** と **Machine B(MLX)**。

0) 環境確認：
```bash
cd ~/paper1
nvidia-smi
ls models/gguf/ models/awq/          # 残存モデル確認
```

1) **参照要約 `data/reference_summary_T1〜T3.txt`（200〜300字の模範要約）を作成** → 再集計でROUGE-L算出：
```bash
.venv/bin/python evaluate.py --results results/results.csv --gt_dir . --data_dir data
# → results/summary_report.{csv,md}（既定は cwd 出力なので results/ へ移動して運用）
```

2) Gemma の低F1（0.25〜0.37）の裏取り: results.csv の Gemma json_extract 出力を定性確認し、
   取りこぼし傾向（recall低下）を具体例で示す。

3) AWQを再走する必要が出た場合（参考）:
```bash
# 起動成功する縮小条件の例（実験条件8K以上は起動不可）
.venv/bin/python benchmark.py --model_type awq --model_path models/awq/Qwen3-8B-AWQ \
  --task json_extract --input_file data/test_data_T1.txt \
  --context_len 4096 --gpu_mem_util 0.97 --output_file results/awq/awq_probe.csv
```

4) （後日）Machine B（Mac mini）で MLX・大型モデル。

> 状態メモ（2026-06-06更新）：**GGUF 2モデル24試行＋AWQ限界実測 完了**。
> `results/results.csv`（GGUF24 valid＋AWQ3 ERROR）、`results/summary_report.{md,csv}`、
> `results/awq/AWQ_limit_summary.md`＋ログ一式。**次は参照要約(ROUGE-L)とMachine B(MLX)**。
> AWQモデル `models/awq/Qwen3-8B-AWQ`(5.7GB) は実験不可と判明済→ディスク逼迫時は削除可。
> `results/_scratch/` と `smoke_*` / `verify_chat.csv` は使い捨て。

---

## 7. 既知の論点・留意点

- **8GBでのAWQ → 実測で確定**：重み5.71GiB＋vLLMオーバーヘッドでKVが0.73GiB止まり、8Kでも起動不可。
  「AWQ(vLLM)は8GBでは最小条件すら実用不可」が論文の知見として確定（指示書§14と整合）。詳細は★AWQ節。
- **32K×8GB → 実測で確定**：Qwen3-8Bはフル(-1)で16K/32Kとも`llama_context`生成失敗。
  部分オフロード(16K=33/32K=18層)で全条件実行でき、32Kは8 tpsまで速度劣化（CPU↔GPU転送が律速）。
  Gemma4-E4Bは32Kでも全層GPUで6.5GBに収まり速度劣化ほぼ無し（モデル依存）。
- **chatテンプレ vs 生補完**：本実験は chatテンプレで統一（論文に明記）。
- **モデルリポジトリ名の訂正**：指示書の `bartowski/Qwen3-8B-GGUF` は実在せず、
  正しくは `bartowski/Qwen_Qwen3-8B-GGUF`（新命名）。AWQは公式 `Qwen/Qwen3-8B-AWQ`。
  Gemma4-E4B GGUFも指示書の `bartowski/gemma-4-e4b-it-GGUF` ではなく
  実在は **`bartowski/google_gemma-4-E4B-it-GGUF`**（ファイル `google_gemma-4-E4B-it-Q4_K_M.gguf` 5.4GB）。
- **Gemma4-E4B**：AWQ版が存在しない場合は指示書§14に従い GGUF Q5_K_M 等で代替し明記。
