#!/usr/bin/env python3
"""
evaluate.py — LLM量子化比較実験 評価スクリプト (指示書 セクション11)

results.csv を読み、
  機能1: JSON抽出タスクの評価 (JSON有効率, アクションアイテムF1, 属性一致率)
  機能2: 要約タスクの評価 (ROUGE-L, オプションでLLM-judge)
  機能3: モデル×量子化×コンテキスト長 別の集計レポート生成
を行い、summary_report.csv / summary_report.md を出力する。

使用例:
  python evaluate.py --results results.csv --gt_dir . --data_dir data
  python evaluate.py --results results.csv --use_llm_judge   # 要API設定
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# JSON 抽出 (benchmark.py と同一ロジック)
# ---------------------------------------------------------------------------
def strip_json(text):
    t = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
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


def parse_action_items(text):
    cand = strip_json(text)
    if cand is None:
        return None
    try:
        obj = json.loads(cand)
    except Exception:
        return None
    items = obj.get("action_items")
    if not isinstance(items, list):
        return None
    return items


# ---------------------------------------------------------------------------
# タスク文字列の正規化と部分一致
# ---------------------------------------------------------------------------
def normalize(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[\s　、。,.\-（）()「」『』]", "", s)
    return s


def task_match(a, b):
    """taskフィールドの部分一致判定 (双方向の部分文字列 or 文字bigram Jaccard)。"""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ga = {na[i:i + 2] for i in range(len(na) - 1)}
    gb = {nb[i:i + 2] for i in range(len(nb) - 1)}
    if not ga or not gb:
        return False
    jac = len(ga & gb) / len(ga | gb)
    return jac >= 0.5


# ---------------------------------------------------------------------------
# 機能1: JSON抽出タスクの F1 と属性一致
# ---------------------------------------------------------------------------
def score_action_items(pred_items, gt_items):
    """貪欲マッチングで TP を数え precision/recall/F1 を返す。
    マッチしたペアについて assignee/due_date/priority の一致率も集計。"""
    used_gt = set()
    tp = 0
    attr = {"assignee": [0, 0], "due_date": [0, 0], "priority": [0, 0]}  # [hit, total]
    for p in pred_items:
        for gi, g in enumerate(gt_items):
            if gi in used_gt:
                continue
            if task_match(p.get("task"), g.get("task")):
                used_gt.add(gi)
                tp += 1
                for key in attr:
                    attr[key][1] += 1
                    if normalize(p.get(key)) == normalize(g.get(key)):
                        attr[key][0] += 1
                break
    n_pred = len(pred_items)
    n_gt = len(gt_items)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp, "n_pred": n_pred, "n_gt": n_gt,
        "precision": precision, "recall": recall, "f1": f1,
        "attr": attr,
    }


# ---------------------------------------------------------------------------
# 機能2: ROUGE-L
# ---------------------------------------------------------------------------
def get_rouge_scorer():
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    except Exception as e:
        print(f"[warn] rouge-score 利用不可: {e}", file=sys.stderr)
        return None


def rouge_l(scorer, reference, hypothesis):
    if scorer is None or not reference:
        return None
    # 日本語は空白分かち書きが無いため文字単位でスコア (簡易)
    ref = " ".join(list(reference))
    hyp = " ".join(list(hypothesis or ""))
    return scorer.score(ref, hyp)["rougeL"].fmeasure


def llm_judge(reference, hypothesis):
    """オプション: Claude API で要約を採点 (1-5)。APIキー未設定なら None。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[warn] ANTHROPIC_API_KEY 未設定のため LLM-judge をスキップ", file=sys.stderr)
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "あなたは日本語要約の評価者です。以下の参照要約と候補要約を比べ、"
            "候補要約の品質を1〜5の整数で採点してください。数字のみ出力。\n\n"
            f"## 参照要約\n{reference}\n\n## 候補要約\n{hypothesis}\n\n## 採点(1-5)\n"
        )
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        m = re.search(r"[1-5]", msg.content[0].text)
        return int(m.group()) if m else None
    except Exception as e:
        print(f"[warn] LLM-judge 失敗: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# データロード
# ---------------------------------------------------------------------------
def load_ground_truth(gt_dir):
    gts = {}
    for fn in os.listdir(gt_dir):
        m = re.match(r"ground_truth_(T\d+)\.json$", fn)
        if m:
            with open(os.path.join(gt_dir, fn), encoding="utf-8") as f:
                obj = json.load(f)
            gts[m.group(1)] = obj.get("action_items", [])
    return gts


def load_reference_summaries(data_dir):
    refs = {}
    if not data_dir or not os.path.isdir(data_dir):
        return refs
    for fn in os.listdir(data_dir):
        m = re.match(r"reference_summary_(T\d+)\.txt$", fn)
        if m:
            with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
                refs[m.group(1)] = f.read().strip()
    return refs


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="量子化比較 評価スクリプト")
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--gt_dir", default=".", help="ground_truth_T*.json の場所")
    ap.add_argument("--data_dir", default="data", help="reference_summary_T*.txt の場所")
    ap.add_argument("--out_prefix", default="summary_report")
    ap.add_argument("--use_llm_judge", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"[error] results が見つかりません: {args.results}", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(open(args.results, encoding="utf-8")))
    gts = load_ground_truth(args.gt_dir)
    refs = load_reference_summaries(args.data_dir)
    scorer = get_rouge_scorer()
    print(f"[info] results={len(rows)}行, ground_truth={list(gts)}, 参照要約={list(refs)}")

    # group key: (model_name, quant_type, context_len)
    groups = defaultdict(lambda: {
        "json_total": 0, "json_valid": 0,
        "f1": [], "assignee": [0, 0], "due_date": [0, 0], "priority": [0, 0],
        "rouge": [], "judge": [],
        "tps": [], "vram": [], "ttft": [],
    })

    for r in rows:
        key = (r["model_name"], r["quant_type"], r["context_len"])
        g = groups[key]
        # 速度・メモリ (数値化できる行のみ)
        for col, dst in (("gen_speed_tps", "tps"), ("peak_vram_gb", "vram"), ("ttft_sec", "ttft")):
            try:
                g[dst].append(float(r[col]))
            except (ValueError, TypeError):
                pass

        task = r["task_type"]
        if task == "json_extract":
            g["json_total"] += 1
            items = parse_action_items(r["output_text"])
            if items is not None:
                g["json_valid"] += 1
                gt = gts.get(r["data_id"])
                if gt is not None:
                    s = score_action_items(items, gt)
                    g["f1"].append(s["f1"])
                    for k in ("assignee", "due_date", "priority"):
                        g[k][0] += s["attr"][k][0]
                        g[k][1] += s["attr"][k][1]
        elif task == "summary":
            ref = refs.get(r["data_id"])
            if ref:
                rl = rouge_l(scorer, ref, r["output_text"])
                if rl is not None:
                    g["rouge"].append(rl)
                if args.use_llm_judge:
                    j = llm_judge(ref, r["output_text"])
                    if j is not None:
                        g["judge"].append(j)

    # 集計
    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    def rate(pair):
        return round(pair[0] / pair[1], 3) if pair[1] else None

    summary_rows = []
    for (model, quant, ctx), g in sorted(groups.items()):
        summary_rows.append({
            "model_name": model, "quant_type": quant, "context_len": ctx,
            "n_runs": g["json_total"] + len(g["rouge"]) if False else None,
            "tok_s": avg(g["tps"]), "ttft_sec": avg(g["ttft"]),
            "peak_vram_gb": avg(g["vram"]),
            "json_valid_rate": rate([g["json_valid"], g["json_total"]]),
            "action_f1": avg(g["f1"]),
            "assignee_acc": rate(g["assignee"]),
            "due_date_acc": rate(g["due_date"]),
            "priority_acc": rate(g["priority"]),
            "rouge_l": avg(g["rouge"]),
            "llm_judge": avg(g["judge"]),
        })

    # CSV 出力
    cols = ["model_name", "quant_type", "context_len", "tok_s", "ttft_sec",
            "peak_vram_gb", "json_valid_rate", "action_f1", "assignee_acc",
            "due_date_acc", "priority_acc", "rouge_l", "llm_judge"]
    with open(f"{args.out_prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)

    # Markdown 出力
    with open(f"{args.out_prefix}.md", "w", encoding="utf-8") as f:
        f.write("## 結果サマリー\n\n")
        f.write("| モデル | 量子化 | ctx | tok/s | TTFT | VRAM | JSON有効率 | F1 | ROUGE-L | judge |\n")
        f.write("|--------|--------|-----|-------|------|------|------------|-----|---------|-------|\n")
        for r in summary_rows:
            def fmt(x, suf=""):
                return f"{x}{suf}" if x is not None else "-"
            f.write("| {m} | {q} | {c} | {tps} | {ttft} | {vram} | {jv} | {f1} | {rl} | {jg} |\n".format(
                m=r["model_name"], q=r["quant_type"], c=r["context_len"],
                tps=fmt(r["tok_s"]), ttft=fmt(r["ttft_sec"]),
                vram=fmt(r["peak_vram_gb"], "GB"),
                jv=fmt(r["json_valid_rate"]), f1=fmt(r["action_f1"]),
                rl=fmt(r["rouge_l"]), jg=fmt(r["llm_judge"]),
            ))

    print(f"[done] {args.out_prefix}.csv / {args.out_prefix}.md を出力")
    for r in summary_rows:
        print(" ", r)


if __name__ == "__main__":
    main()
