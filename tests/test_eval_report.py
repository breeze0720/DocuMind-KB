"""tests/test_eval_report.py - 评测台纯函数测试：命中判定（主/辅口径）、分位数、报告输出。"""
import json

from eval.report import evaluate_case, load_eval_set, match_case, percentile, write_markdown


def test_match_case_requires_source_and_keyword():
    """主指标是强命中：来源对 + 关键词也对。只有来源对（块里没关键词）不算。"""
    case = {"sources": ["员工手册.docx"], "keywords": ["十天"]}
    results = [
        {"rank": 1, "source": "员工手册.docx", "content": "这里没提年假"},
        {"rank": 2, "source": "员工手册.docx", "content": "年假为每年十天"},
    ]
    assert match_case(case, results) == (True, 2)


def test_match_case_ignores_keyword_only_hit():
    """关键词命中了但来源不同 → 主指标不算命中（这就是之前口径虚高的坑）。"""
    case = {"sources": ["员工手册.docx"], "keywords": ["十天"]}
    results = [{"rank": 1, "source": "其他文档.md", "content": "年假为每年十天"}]
    assert match_case(case, results) == (False, None)


def test_evaluate_case_reports_both_metrics():
    case = {"sources": ["员工手册.docx"], "keywords": ["十天"]}
    results = [
        {"rank": 1, "source": "其他文档.md", "content": "年假为每年十天"},
        {"rank": 2, "source": "员工手册.docx", "content": "年假为每年十天"},
    ]
    judged = evaluate_case(case, results)
    assert judged["hit"] is True and judged["hit_rank"] == 2
    assert judged["kw_hit"] is True and judged["kw_rank"] == 1
    assert judged["source_rank"] == 2 and judged["rr"] == 0.5


def test_match_case_misses():
    case = {"sources": ["不存在.pdf"], "keywords": ["查无此词"]}
    results = [{"rank": 1, "source": "其他.md", "content": "别的内容"}]
    assert match_case(case, results) == (False, None)


def test_percentile():
    assert percentile([], 0.95) == 0.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.50) == 20.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == 40.0


def test_load_eval_set_compatible_schema(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "question": "年假几天", "source": "员工手册.docx", "kw": "十天"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"query": "报销流程", "expected_source": ["制度.pdf"], "keywords": ["发票"]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    cases, placeholder = load_eval_set(path)
    assert placeholder is False
    assert len(cases) == 2
    assert cases[0]["sources"] == ["员工手册.docx"]
    assert cases[0]["keywords"] == ["十天"]
    assert cases[1]["question"] == "报销流程"


def test_write_markdown_contains_metrics(tmp_path):
    report = {
        "tag": "before",
        "generated_at": "2026-01-01 00:00:00",
        "base_url": "http://127.0.0.1:8010",
        "embedding": "local",
        "collection": "kb_bge_512",
        "eval_set": "eval_set.jsonl",
        "chunk_count": 12,
        "placeholder": False,
        "k": 5,
        "metrics": {
            "total": 1,
            "hits": 1,
            "hit_rate": 1.0,
            "mrr": 1.0,
            "kw_hits": 1,
            "kw_hit_rate": 1.0,
            "kw_mrr": 1.0,
            "p50_latency_ms": 12.0,
            "p95_latency_ms": 15.0,
        },
        "by_type": {"literal": {"total": 1, "hits": 1, "hit_rate": 1.0, "mrr": 1.0}},
        "cases": [
            {
                "id": "q1",
                "type": "literal",
                "question": "年假几天",
                "hit": True,
                "hit_rank": 1,
                "kw_hit": True,
                "kw_rank": 1,
                "rr": 1.0,
                "latency_ms": 12.0,
                "top_sources": ["员工手册.docx"],
                "error": "",
            }
        ],
    }
    path = tmp_path / "before.md"
    write_markdown(path, report)
    text = path.read_text(encoding="utf-8")
    assert "HitRate@5" in text
    assert "100.00%" in text
    assert "天花板效应" in text
    assert "分题型" in text
    assert "q1" in text