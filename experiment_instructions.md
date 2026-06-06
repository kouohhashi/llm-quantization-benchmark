# LLM量子化比較実験 指示書
# GGUF vs AWQ vs MLX — 日本語ビジネス文書タスクにおける品質・速度・メモリの実測評価

作成日：2026年6月
対象Claude：Claude Code（Ubuntu / Mac mini）

---

## 0. この指示書の使い方

この指示書はテクニカルペーパー執筆のための実験を設計したものです。
Claudeへの指示として読み、各セクションの「Claudeへの指示」に従って
セットアップ・実験・結果収集を進めてください。
疑問点や詰まった箇所はその都度聞いてください。

---

## 1. 実験の目的

### リサーチクエスチョン
「同一モデルをGGUF・AWQ・MLXで量子化した場合、
日本語ビジネス文書タスク（要約・構造化抽出）において
速度・品質・メモリ使用量にどのような差が生じるか？」

### 論文としての独自性
- 既存比較のほとんどは英語汎用ベンチマーク（MMLU等）での評価
- 本実験は日本語ビジネス文書（会議録）という実務タスクで比較
- コンテキスト長（8K/16K/32K）を変数として、VRAMオーバーフロー時の
  性能劣化を実測する点も新規性あり

---

## 2. ハードウェア構成

### Machine A（Ubuntu）
- GPU：NVIDIA RTX4060 Laptop（VRAM 8GB）
- 担当量子化：GGUF Q4_K_M、AWQ INT4
- 担当モデル：Qwen3-8B、Gemma4-E4B

### Machine B（Mac mini）※後日追加実験
- メモリ：64GB統合メモリ（Apple Silicon）
- 担当量子化：GGUF Q4_K_M、MLX 4bit
- 担当モデル：Qwen3-32B、Qwen3-30B-A3B（MoE）

---

## 3. 実験対象モデル

### Machine A で使うモデル

| ID | モデル名 | パラメータ | 量子化 | 想定VRAM |
|----|----------|------------|--------|----------|
| A1 | Qwen3-8B | 8B | GGUF Q4_K_M | ~5GB |
| A2 | Qwen3-8B | 8B | AWQ INT4 | ~5GB |
| A3 | Gemma4-E4B | 4B(MoE) | GGUF Q4_K_M | ~3GB |
| A4 | Gemma4-E4B | 4B(MoE) | AWQ INT4 | ~3GB |

### Machine B で使うモデル（後日）

| ID | モデル名 | パラメータ | 量子化 | 想定メモリ |
|----|----------|------------|--------|------------|
| B1 | Qwen3-32B | 32B | GGUF Q4_K_M | ~20GB |
| B2 | Qwen3-32B | 32B | MLX 4bit | ~20GB |
| B3 | Qwen3-30B-A3B | 30B(MoE) | GGUF Q4_K_M | ~17GB |
| B4 | Qwen3-30B-A3B | 30B(MoE) | MLX 4bit | ~17GB |

---

## 4. タスク設計

### タスクA：日本語会議録の要約

**入力：** 日本語会議トランスクリプト（架空、後述のテストデータ参照）
**出力：** 会議の要約文（200〜300字程度）
**プロンプトテンプレート：**

```
以下は会議のトランスクリプトです。
会議の要点を200〜300字で要約してください。

## トランスクリプト
{transcript}

## 要約
```

### タスクB：アクションアイテムのJSON抽出

**入力：** 同じ日本語会議トランスクリプト
**出力：** 下記スキーマに従ったJSON
**プロンプトテンプレート：**

```
以下の会議トランスクリプトから、アクションアイテムを抽出してください。
必ず以下のJSONスキーマに従って出力してください。それ以外のテキストは出力しないでください。

スキーマ：
{
  "action_items": [
    {
      "task": "タスクの内容（文字列）",
      "assignee": "担当者名（文字列、不明な場合はnull）",
      "due_date": "期日（YYYY-MM-DD形式、不明な場合はnull）",
      "priority": "high/medium/low のいずれか"
    }
  ]
}

## トランスクリプト
{transcript}

## JSON出力
```

---

## 5. テストデータ仕様

### 概要
架空の日本語会議トランスクリプトを3種類用意する。
長さを変えることでコンテキスト長の影響を測定できるようにする。

| データID | 会議種別 | トークン数目安 | コンテキスト対応 |
|----------|----------|---------------|-----------------|
| T1 | 週次進捗MTG（短） | ~3,000 tokens | 8K・16K・32K全て |
| T2 | 月次レビュー（中） | ~7,000 tokens | 16K・32Kのみ |
| T3 | 四半期計画MTG（長） | ~14,000 tokens | 32Kのみ |

### Claudeへの指示（テストデータ生成）
```
以下の仕様で架空の日本語会議トランスクリプトを生成してください。

【T1：週次進捗MTG】
- 参加者：4名（田中PM、佐藤エンジニア、鈴木デザイナー、山本営業）
- 会議時間：30分
- 内容：SaaSプロダクトの週次進捗確認、バグ報告、来週のタスク確認
- アクションアイテムが5〜7個含まれるようにする
- 担当者・期日が明確なものと曖昧なものを混在させる
- トークン数：約3,000

【T2：月次レビュー】
- 参加者：6名（役員含む）
- 会議時間：60分
- 内容：KPIレビュー、課題共有、翌月計画
- アクションアイテムが10〜12個含まれるようにする
- トークン数：約7,000

【T3：四半期計画MTG】
- 参加者：8名（部門横断）
- 会議時間：120分
- 内容：Q3振り返り、Q4目標設定、予算確認、リソース調整
- アクションアイテムが15〜20個含まれるようにする
- トークン数：約14,000

各トランスクリプトはリアルな会話形式（「田中：〜」）で書いてください。
ファイル名：test_data_T1.txt、test_data_T2.txt、test_data_T3.txt
```

---

## 6. 測定項目

### 速度・メモリ系
| 指標 | 単位 | 測定方法 |
|------|------|----------|
| TTFT | 秒 | 最初のトークンが出るまでの時間 |
| 生成速度 | tok/s | 生成中の平均トークン/秒 |
| ピークVRAM | GB | nvidia-smi で測定 |
| オフロード層数 | 層 | llama.cppのログから取得 |

### 品質系
| 指標 | 対象タスク | 測定方法 |
|------|------------|----------|
| JSON有効率 | タスクB | json.loads()でパース成功率 |
| アクションアイテム抽出F1 | タスクB | 正解セットとの比較 |
| 要約ROUGE-L | タスクA | rouge-scoreライブラリ |
| 要約LLM-judge | タスクA | GPT-4o or Claude APIで採点 |

---

## 7. 実験条件マトリクス

コンテキスト長 × モデル × 量子化 の全組み合わせ：

```
Machine A の実験：
  モデル    量子化       8K   16K   32K
  ─────────────────────────────────────
  Qwen3-8B  GGUF Q4_K_M  ✓    ✓     ✓
  Qwen3-8B  AWQ INT4      ✓    ✓     ✓
  Gemma4-E4B GGUF Q4_K_M ✓    ✓     ✓
  Gemma4-E4B AWQ INT4     ✓    ✓     ✓

  × データ T1（3条件） + T2（16K・32Kのみ） + T3（32Kのみ）
  → 合計：各モデル×量子化で約8試行 × 4条件 = 約32試行
```

---

## 8. セットアップ手順（Machine A / Ubuntu）

### Claudeへの指示（環境構築）

```
以下の環境をUbuntu上に構築してください。

【Step 1】NVIDIA ドライバ・CUDA確認
  nvidia-smi でドライババージョンとCUDAバージョンを確認して報告してください。

【Step 2】llama.cpp（GGUF用）インストール
  conda または venv で Python 3.11 環境を作成し、
  llama-cpp-python をCUDAサポート付きでインストールしてください。
  pip install llama-cpp-python --extra-index-url \
    https://abetlen.github.io/llama-cpp-python/whl/cu121

【Step 3】vLLM（AWQ用）インストール
  pip install vllm
  AWQ対応確認のため以下を実行：
  python -c "from vllm import LLM; print('vllm ok')"

【Step 4】モデルのダウンロード
  以下のモデルをHugging Face からダウンロードしてください：
  
  GGUF:
  - bartowski/Qwen3-8B-GGUF（Q4_K_M）
  - bartowski/gemma-4-e4b-it-GGUF（Q4_K_M）
  
  AWQ:
  - Qwen/Qwen3-8B-AWQ
  - （Gemma4-E4BのAWQが存在するか確認し、なければGGUFのみ）

【Step 5】計測ツールの準備
  pip install rouge-score psutil gputil
  nvidia-smi dmon コマンドが使えるか確認してください。
```

---

## 9. ベンチマークスクリプト仕様

### Claudeへの指示（スクリプト作成）

```
以下の仕様でPythonベンチマークスクリプトを作成してください。

ファイル名：benchmark.py

【機能】
1. 指定モデル・量子化・コンテキスト長でテキストを推論
2. TTFT・生成速度・ピークVRAMを自動計測
3. 結果をCSVに追記保存（results.csv）
4. JSON出力の場合はパース成功/失敗を記録

【CLI引数】
  --model_type  : gguf or awq
  --model_path  : モデルのパスまたはHF ID
  --task        : summary or json_extract
  --input_file  : テストデータのパス
  --context_len : 8192 or 16384 or 32768
  --output_file : 結果CSV出力先（デフォルト：results.csv）

【results.csvのカラム】
  timestamp, model_name, quant_type, context_len, task_type,
  data_id, ttft_sec, gen_speed_tps, peak_vram_gb,
  offload_layers, output_valid, output_text

【計測方法】
  - TTFT：最初のトークン生成までのtime.time()差分
  - gen_speed：総生成トークン数 / 生成時間
  - peak_vram：nvidia-smi経由でpynvmlを使用
  - offload_layers：llama.cppのn_gpu_layers設定から判断

【注意事項】
  - 各試行の前にGPUメモリをクリア（torch.cuda.empty_cache()）
  - エラー時もCSVにエラー内容を記録して続行
  - 推論結果は最初の512文字までをoutput_textに保存
```

---

## 10. 正解データ（グランドトゥルース）の作成

### Claudeへの指示

```
テストデータ（T1・T2・T3）に対して、正解アクションアイテムリストを
以下のJSON形式で作成してください。

ファイル名：ground_truth_T1.json、ground_truth_T2.json、ground_truth_T3.json

形式：
{
  "action_items": [
    {
      "task": "〇〇の修正を完了する",
      "assignee": "佐藤",
      "due_date": "2026-06-13",
      "priority": "high"
    },
    ...
  ]
}

これはF1スコア計算の基準として使います。
taskフィールドは部分一致で照合するので、
表現の揺れを吸収できるよう標準的な表現にしてください。
```

---

## 11. 評価スクリプト仕様

### Claudeへの指示（評価スクリプト作成）

```
以下の仕様で評価スクリプトを作成してください。

ファイル名：evaluate.py

【機能1】JSON抽出タスクの評価
  - results.csvのoutput_textをパースしてJSON有効率を計算
  - ground_truth_T*.jsonとtaskフィールドを部分一致比較してF1計算
  - assignee・due_date・priorityの一致率も個別に計算

【機能2】要約タスクの評価
  - rouge-scoreでROUGE-Lを計算（参照要約が必要、後述）
  - オプション：--use_llm_judge フラグでLLM採点も実行

【機能3】集計レポート生成
  - モデル × 量子化 × コンテキスト長 の組み合わせ別に集計
  - summary_report.csvとsummary_report.mdを出力

【出力サンプル（summary_report.md）】

## 結果サマリー

| モデル | 量子化 | ctx | tok/s | VRAM | JSON有効率 | F1 |
|--------|--------|-----|-------|------|------------|-----|
| Qwen3-8B | GGUF | 8K | 40.2 | 4.8GB | 92% | 0.71 |
| Qwen3-8B | AWQ | 8K | 48.1 | 4.6GB | 96% | 0.74 |
| ...      |      |    |       |       |            |     |
```

---

## 12. 実験実施の順序

```
Week 1（Ubuntu / Machine A）:
  Day 1: 環境構築・モデルダウンロード（セクション8）
  Day 2: テストデータ生成・正解データ作成（セクション5・10）
          スクリプト作成・動作確認（セクション9・11）
  Day 3: 本実験実施（セクション7の全組み合わせ）
          ※自動化スクリプトで一晩回してもOK

Week 2（Mac mini / Machine B）:
  Day 1: Mac mini環境構築（MLX・llama.cpp）
  Day 2: 大型モデル実験（B1〜B4）
  Day 3: 結果統合・分析・グラフ生成

Week 3: 論文執筆
```

---

## 13. 論文構成（アウトライン）

```
Title: 日本語ビジネス文書タスクにおけるLLM量子化方式の比較評価：
       GGUF・AWQ・MLXの速度・品質・メモリトレードオフ

1. Introduction
   - ローカルLLM普及の背景
   - 量子化選択の実務的重要性
   - 本論文の貢献（日本語タスク実測・コンテキスト長影響の定量化）

2. Background
   - 量子化の基礎（FP16→INT4）
   - GGUF・AWQ・MLXの技術概要
   - 関連研究（JSONSchemaBench、既存量子化比較論文）

3. Experimental Setup
   - ハードウェア構成
   - モデル選定の理由
   - タスク設計（要約・JSON抽出）
   - 評価指標

4. Results
   4.1 速度比較（tok/s・TTFT）
   4.2 メモリ使用量比較
   4.3 コンテキスト長の影響（VRAMオーバーフロー時の性能劣化）
   4.4 品質比較（JSON有効率・F1・ROUGE-L）
   4.5 速度×品質のトレードオフ分析

5. Discussion
   - 「AWQはGGUFより高品質」という通説の検証結果
   - 日本語タスク特有の傾向
   - 実務的な選択指針（フローチャート形式で提示）

6. Conclusion
   - 主要な発見のまとめ
   - 限界と今後の課題

Appendix
   - 実験に使用したプロンプト全文
   - テストデータの統計情報
   - 全実験結果の生データ（CSV）
```

---

## 14. 注意事項・免責

- AWQ版モデルが存在しない場合はGGUF Q5_K_Mで代替し、その旨を論文に明記する
- LLM-as-judgeを使う場合はAPIコストが発生する（GPT-4o or Claude API）
- コンテキスト長32Kの実験はVRAMオーバーフローが発生する可能性が高く、
  その際は意図的にオーバーフローさせて「限界の定量化」として記録する
- テストデータは完全な架空データを使用すること（実際の会議録は使わない）
- 結果はすべてresults/ディレクトリに保存し、論文のAppendixとして公開予定

---

## 15. Claudeへの最初の指示

この指示書を読んだら、まず以下を実施してください：

1. `nvidia-smi` を実行してGPU環境を確認・報告する
2. `python --version` と `pip --version` を確認する
3. 利用可能なディスク容量を確認する（モデルは合計20〜30GB必要）
4. セクション8のStep 1〜2から順番に環境構築を開始する

準備ができたら「環境構築完了。次のステップに進みます」と報告してください。
