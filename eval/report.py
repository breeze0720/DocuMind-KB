"""
eval/report.py - 评测的纯函数部分：读标注集、命中判定、分位数、Markdown 报告

把「数据与计算」和「HTTP 调用」分开，好处是这部分能被 pytest 直接覆盖
（构造几条假检索结果就能验证命中判定和指标算法），也不需要起服务。

指标口径（面试被追问过，所以写清楚）：
- 主指标（HitRate@5 / MRR）= **强命中**：检索结果的 source 与标注 source 一致，
  且该块正文里出现了标注关键词。只有来源对、关键词也对，才算「真的检索到了答案」。
- 辅指标（kw_hit）= 只要正文包含关键词就算，不管来源。它只用于 debug：
  排查「关键词找到了但来源排错」这类情况，不参与主指标计算。
为什么必须分开：如果主指标用「来源 OR 关键词」的并集，会出现「Top1 是别的文档、
但关键词碰巧命中」也被算作命中，指标虚高、口径说不清（这正是之前的坑）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SET = PROJECT_DIR / "eval" / "eval_set.jsonl"
DEFAULT_OUT_DIR = PROJECT_DIR / "eval_results"

# 语料还没生成时的占位题：只为让脚本能跑通流程，数字没有任何意义（报告里会标记 placeholder）
PLACEHOLDER_CASES = [
    {"id": "placeholder-1", "question": "年假有多少天？", "source": "", "keywords": []},
    {"id": "placeholder-2", "question": "报销流程需要哪些材料？", "source": "", "keywords": []},
    {"id": "placeholder-3", "question": "IT 设备申请找谁？", "source": "", "keywords": []},
    {"id": "placeholder-4", "question": "产品支持哪些文件格式？", "source": "", "keywords": []},
    {"id": "placeholder-5", "question": "新员工入职培训多久？", "source": "", "keywords": []},
]


def load_eval_set(path: Path) -> tuple[list[dict], bool]:
    """
    读 JSONL 标注集，返回 (题目列表, 是否占位集)。
    字段名做兼容（question/query/q、source/sources、keywords/kw、type/category），
    因为标注集可能由不同同学用不同字段名生成，评测台不该因此跑不起来。
    """
    if not path.is_file():
        print(f"[eval] 未找到标注集 {path}，先写入 5 条占位题（结果无意义，仅用于跑通流程）")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for case in PLACEHOLDER_CASES:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
        return [dict(case) for case in PLACEHOLDER_CASES], True

    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            print(f"[eval] 第 {line_no} 行不是合法 JSON，跳过")
            continue
        question = str(raw.get("question") or raw.get("query") or raw.get("q") or "").strip()
        if not question:
            continue
        sources = raw.get("sources") or raw.get("source") or raw.get("expected_source") or []
        if isinstance(sources, str):
            sources = [sources]
        sources = [Path(str(item)).name for item in sources if str(item).strip()]
        keywords = raw.get("keywords") or raw.get("kw") or raw.get("expected_keywords") or []
        if isinstance(keywords, str):
            keywords = [item for item in keywords.replace("，", ",").split(",") if item.strip()]
        cases.append(
            {
                "id": str(raw.get("id") or f"case-{line_no}"),
                "question": question,
                "sources": sources,
                "keywords": [str(item) for item in keywords if str(item).strip()],
                "type": str(raw.get("type") or raw.get("category") or "").strip(),
                "note": str(raw.get("note") or ""),
            }
        )
    placeholder = bool(cases) and all(str(case["id"]).startswith("placeholder") for case in cases)
    if not cases:
        print(f"[eval] 标注集为空: {path}")
    return cases, placeholder


def _source_match(case: dict, result: dict) -> bool:
    expected_sources = [source.lower() for source in case.get("sources") or []]
    source = str(result.get("source") or "").lower()
    return bool(expected_sources) and any(
        source == expected or expected in source or source.endswith(expected) for expected in expected_sources
    )


def _keyword_match(case: dict, result: dict) -> bool:
    keywords = [keyword.lower() for keyword in case.get("keywords") or []]
    content = str(result.get("content") or "").lower()
    return bool(keywords) and any(keyword in content for keyword in keywords)


def match_case(case: dict, results: list[dict]) -> tuple[bool, int | None]:
    """
    主指标判定：在 Top-K 里找第一个「强命中」项，返回 (是否命中, 1 基排名)。
    强命中 = 来源文件名匹配且该块正文含标注关键词；没有标关键词的题退化为来源命中。
    """
    has_keywords = bool(case.get("keywords"))
    for result in results:
        if not _source_match(case, result):
            continue
        if not has_keywords or _keyword_match(case, result):
            return True, int(result.get("rank") or 0)
    return False, None


def keyword_match(case: dict, results: list[dict]) -> tuple[bool, int | None]:
    """辅指标判定：只看正文是否含关键词（不管来源），用于 debug 口径对比。"""
    for result in results:
        if _keyword_match(case, result):
            return True, int(result.get("rank") or 0)
    return False, None


def evaluate_case(case: dict, results: list[dict]) -> dict[str, Any]:
    """同时算主指标（强命中）与辅指标（kw 命中），供报告和逐题明细使用。"""
    hit, hit_rank = match_case(case, results)
    kw_hit, kw_rank = keyword_match(case, results)
    source_rank = next(
        (int(result.get("rank") or 0) for result in results if _source_match(case, result)),
        None,
    )
    return {
        "hit": hit,
        "hit_rank": hit_rank,
        "rr": round(1.0 / hit_rank, 4) if hit and hit_rank else 0.0,
        "kw_hit": kw_hit,
        "kw_rank": kw_rank,
        "kw_rr": round(1.0 / kw_rank, 4) if kw_hit and kw_rank else 0.0,
        "source_hit": source_rank is not None,
        "source_rank": source_rank,
    }


def percentile(values: list[float], p: float) -> float:
    """最近秩分位数（nearest-rank）：P95 取排序后第 ceil(0.95*n) 个，n 小时等于最大值，正常。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return round(ordered[index], 1)


TYPE_LABELS = {
    "literal": "literal（原词）",
    "paraphrase": "paraphrase（同义改写）",
    "short": "short（短问题）",
}


def write_markdown(path: Path, report: dict) -> None:
    """输出人类可读报告：天花板说明 → 主/辅总览 → 分题型 → 逐题明细。"""
    metrics = report["metrics"]
    k = report.get("k", 5)
    total = metrics["total"]
    lines = [
        f"# 检索评测报告：{report['tag']}",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 服务地址：{report['base_url']}",
        f"- embedding 后端：{report['embedding']}（collection: {report['collection']}）",
        f"- 标注集：{report['eval_set']}（{total} 条）",
        f"- 知识库块数：{report['chunk_count']}",
    ]
    if report.get("placeholder"):
        lines.append("- ⚠️ 当前使用占位标注集，指标无参考意义；拿到真实语料后请重新评测")
    lines += [
        "",
        "> **天花板效应**：本套语料仅 "
        f"{report['chunk_count']} 个块、{total} 道题，检索指标已接近上限；",
        "> 真实场景需扩大语料、加入同主题干扰文档后再复测（见 README Roadmap）。",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 | 口径 |",
        "| --- | --- | --- |",
        f"| HitRate@{k}（主） | {metrics['hit_rate']:.2%} | 来源命中且该块含标注关键词 |",
        f"| MRR（主） | {metrics['mrr']:.4f} | 同上，按强命中排名计算 |",
        f"| HitRate@{k}（辅） | {metrics.get('kw_hit_rate', 0):.2%} | 仅关键词命中，debug 参考 |",
        f"| MRR（辅） | {metrics.get('kw_mrr', 0):.4f} | 仅关键词命中排名 |",
        f"| P50 延迟 | {metrics['p50_latency_ms']:.1f} ms | 检索接口耗时 |",
        f"| P95 延迟 | {metrics['p95_latency_ms']:.1f} ms | 检索接口长尾耗时 |",
        f"| 主命中/总数 | {metrics['hits']}/{total} | 强命中题目数 |",
        "",
        "## 分题型（主指标）",
        "",
        f"| 题型 | 题数 | HitRate@{k} | MRR |",
        "| --- | --- | --- | --- |",
    ]
    for type_name, stats in (report.get("by_type") or {}).items():
        label = TYPE_LABELS.get(type_name, type_name or "未标注")
        lines.append(f"| {label} | {stats['total']} | {stats['hit_rate']:.2%} | {stats['mrr']:.4f} |")

    lines += [
        "",
        "## 逐题明细",
        "",
        "| 题号 | 题型 | 问题 | 主命中 | 排名 | kw命中 | kw排名 | RR | 延迟(ms) | Top1 来源 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        question = case["question"].replace("|", " ")[:40]
        top_source = (case.get("top_sources") or ["—"])[0] or "—"
        type_label = case.get("type") or "—"
        lines.append(
            f"| {case['id']} | {type_label} | {question} | {'✅' if case['hit'] else '❌'} | "
            f"{case.get('hit_rank') or '—'} | {'✅' if case.get('kw_hit') else '❌'} | "
            f"{case.get('kw_rank') or '—'} | {case['rr']:.3f} | {case['latency_ms']:.1f} | {top_source} |"
        )

    errors = [case for case in report["cases"] if case.get("error")]
    if errors:
        lines += ["", "## 失败题目", ""]
        lines += [f"- {case['id']}：{case['error']}" for case in errors]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")