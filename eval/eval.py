"""
eval/eval.py - 检索效果评测台（HitRate@K / MRR / P95 延迟）

为什么需要这个脚本（也是简历上所有百分比的来源）：
没有基线数字，一切「优化」都无法证明有效。这个脚本逐条调用 /api/search（只测检索，
不经过大模型，结果可复现、便宜、快），用标注集算三个指标：
- HitRate@K：Top-K 里至少有一条命中正确答案的题目占比；
- MRR：命中位置倒数的平均值（第 1 名命中=1.0，第 2 名=0.5……，衡量排序质量）；
- P95 延迟：95% 的请求快于这个值（比平均值更能暴露长尾卡顿）。

用法：
    python eval/eval.py --tag local                        # 默认打 8010 端口
    python eval/eval.py --tag after --k 10 --base-url http://127.0.0.1:8000

前提：服务已启动，且索引已有语料（POST /api/reindex）。
输出：eval_results/<tag>.md（人看）、eval_results/latest.json（前端 /api/eval/summary 读）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from report import (  # noqa: E402  （直接跑脚本时 eval/ 就是 sys.path[0]）
    DEFAULT_EVAL_SET,
    DEFAULT_OUT_DIR,
    evaluate_case,
    load_eval_set,
    percentile,
    write_markdown,
)


def call_search(client: httpx.Client, base_url: str, question: str, k: int) -> dict:
    """调一次 /api/search。只传 query 和 k，服务端用固定的本地 bge 索引。"""
    payload: dict[str, Any] = {"query": question, "k": k}
    response = client.post(f"{base_url.rstrip('/')}/api/search", json=payload)
    response.raise_for_status()
    return response.json()


def run_case(client: httpx.Client, base_url: str, case: dict, k: int) -> dict:
    """跑一道题，返回主指标（强命中）/辅指标（kw 命中）/耗时/错误信息。单题失败不中断整轮评测。"""
    started = time.perf_counter()
    try:
        response = call_search(client, base_url, case["question"], k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        results = response.get("results") or []
        judged = evaluate_case(case, results)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            **case,
            "hit": False,
            "hit_rank": None,
            "rr": 0.0,
            "kw_hit": False,
            "kw_rank": None,
            "kw_rr": 0.0,
            "source_hit": False,
            "source_rank": None,
            "latency_ms": round(elapsed_ms, 1),
            "top_sources": [],
            "error": str(exc),
        }
    return {
        **case,
        **judged,
        "latency_ms": round(elapsed_ms, 1),
        "top_sources": [str(item.get("source")) for item in results[:3]],
        "error": "",
    }


def _aggregate(records: list[dict]) -> dict:
    """把逐题记录聚合成 HitRate/MRR（主+辅），分题型时复用同一段逻辑。"""
    total = len(records)
    hits = sum(1 for record in records if record["hit"])
    kw_hits = sum(1 for record in records if record["kw_hit"])
    return {
        "total": total,
        "hits": hits,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "mrr": round(sum(record["rr"] for record in records) / total, 4) if total else 0.0,
        "kw_hits": kw_hits,
        "kw_hit_rate": round(kw_hits / total, 4) if total else 0.0,
        "kw_mrr": round(sum(record["kw_rr"] for record in records) / total, 4) if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评测台（HitRate@K / MRR / P95）")
    parser.add_argument("--tag", default="latest", help="结果文件名前缀，如 before / after / local")
    parser.add_argument("--k", type=int, default=5, help="HitRate@K 的 K，默认 5（与 TOP_K 对齐）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="服务地址，自测端口用 8010")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    eval_set_path = Path(args.eval_set)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, placeholder = load_eval_set(eval_set_path)
    if not cases:
        print("[eval] 没有可评测的题目，退出")
        return 2

    base_url = args.base_url.rstrip("/")
    with httpx.Client(timeout=args.timeout) as client:
        try:
            status = client.get(f"{base_url}/api/status").json()
        except Exception as exc:
            print(f"[eval] 连不上服务 {base_url}: {exc}")
            print("[eval] 请先启动服务：python -m uvicorn main:app --port 8010")
            return 2
        active = status.get("embedding") or {}
        chunk_count = int(status.get("chunk_count") or 0)
        if chunk_count == 0:
            print(f"[eval] 索引为空，请先导入语料：POST {base_url}/api/reindex")
            return 2
        print(
            f"[eval] 服务就绪：model={active.get('label') or active.get('model')}，维度={active.get('dimension')}，"
            f"collection={active.get('collection')}，块数={chunk_count}"
        )

        records = [run_case(client, base_url, case, args.k) for case in cases]

    total = len(records)
    latencies = [record["latency_ms"] for record in records]
    metrics = _aggregate(records)
    metrics["p50_latency_ms"] = percentile(latencies, 0.50)
    metrics["p95_latency_ms"] = percentile(latencies, 0.95)
    # 按标注集的 type 分组：HitRate 饱和时（小语料常见），分类型才能看出语义题是否短板
    by_type: dict[str, dict] = {}
    for type_name in dict.fromkeys(str(record.get("type") or "") for record in records):
        subset = [record for record in records if str(record.get("type") or "") == type_name]
        by_type[type_name] = _aggregate(subset)
    report = {
        "tag": args.tag,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": base_url,
        "embedding": active.get("label") or active.get("model") or "bge-small-zh-v1.5",
        "collection": active.get("collection", ""),
        "k": args.k,
        "eval_set": str(eval_set_path),
        "placeholder": placeholder,
        "chunk_count": chunk_count,
        "metrics": metrics,
        "by_type": by_type,
        "cases": records,
    }

    markdown_path = out_dir / f"{args.tag}.md"
    latest_path = out_dir / "latest.json"
    write_markdown(markdown_path, report)
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print(
        f"[eval] 完成（主口径=来源+关键词强命中）：HitRate@{args.k}={metrics['hit_rate']:.2%}，"
        f"MRR={metrics['mrr']:.4f}，P95={metrics['p95_latency_ms']:.1f}ms（{metrics['hits']}/{metrics['total']} 命中）"
    )
    print(
        f"[eval] 辅口径（仅关键词）：HitRate@{args.k}={metrics['kw_hit_rate']:.2%}，MRR={metrics['kw_mrr']:.4f}"
    )
    print(f"[eval] 报告: {markdown_path}")
    print(f"[eval] 结构化结果: {latest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())