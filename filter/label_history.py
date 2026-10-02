# coding=utf-8
"""
阶段 0：从历史 db 导出热榜标题，用本机 AI 标注「是否娱乐圈明星」。

用法（在 trendradar 目录下）：
    uv run python filter/label_history.py --export-only   # 只导标题，看数量
    uv run python filter/label_history.py                 # 导出 + 标注（可反复重入，已标过的跳过）
"""
import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
TITLES_CSV = DATA / "titles.csv"
LABELED_CSV = DATA / "labeled.csv"

DAYS = ["2025-12-25", "2025-12-26", "2025-12-27"]
BATCH_SIZE = 50
TOPICS_CSV = DATA / "labeled_topics.csv"

# 导出顺序：娱乐/社会话题密集的平台先跑，财经时政排后面
PLATFORM_ORDER = [
    "weibo", "douyin", "bilibili-hot-search", "tieba", "hupu",
    "toutiao", "zhihu", "baidu",
]

INTERESTS_FILE = ROOT / "config" / "ai_interests.txt"

SYSTEM_PROMPT = """你是短视频选题编辑，负责从热榜标题里挑出「值得做成社会热点汇报视频」的选题。

我们的选题标准如下：

{interests}

逐条判断每条热榜标题是否属于我们要做的选题。
凡是「一律排除」清单里的（娱乐圈与明星、体育、财经股市、时政与国际关系、科技数码与游戏、低质量标题）一律判 false，哪怕它是当天最热的新闻。
拿不准时判 false。

只输出 JSON，不要任何解释文字。"""

USER_TEMPLATE = """共 {count} 条标题，编号即 id：

{news_list}

输出格式（严格）：
{{"results":[{{"id":1,"keep":true}},{{"id":2,"keep":false}}]}}"""


def export_titles() -> list:
    """从三天的 db 导出标题，按标题去重。"""
    DATA.mkdir(parents=True, exist_ok=True)
    seen = {}
    for day in DAYS:
        db = ROOT / "output" / "news" / f"{day}.db"
        if not db.exists():
            print(f"[跳过] 库不存在: {db}")
            continue
        conn = sqlite3.connect(db)
        for title, plat, rank in conn.execute(
            "select title, platform_id, rank from news_items order by rank"
        ):
            title = (title or "").strip()
            if title and title not in seen:
                seen[title] = {"title": title, "platform": plat, "rank": rank, "day": day}
        conn.close()

    rows = list(seen.values())
    rows.sort(key=lambda r: (PLATFORM_ORDER.index(r["platform"]) if r["platform"] in PLATFORM_ORDER else 99,
                             r["rank"]))
    with TITLES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["title", "platform", "rank", "day"])
        w.writeheader()
        w.writerows(rows)
    print(f"[导出] {len(rows)} 条去重标题 → {TITLES_CSV}")
    return rows


def load_labeled() -> dict:
    """已标注结果：title -> 'true'/'false'（keep）"""
    if not TOPICS_CSV.exists():
        return {}
    out = {}
    with TOPICS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["title"]] = row["keep"]
    return out


def parse_keeps(text: str, ids: list) -> dict:
    """从 AI 响应里抠出 {id: bool}。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"响应里没有 JSON: {text[:200]}")
    payload = json.loads(m.group(0))
    out = {}
    for item in payload.get("results", []):
        try:
            out[int(item["id"])] = bool(item["keep"])
        except (KeyError, TypeError, ValueError):
            continue
    missing = [i for i in ids if i not in out]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 条结果: {missing[:10]}")
    return out


def label(rows: list, max_batches: int = 0) -> None:
    from trendradar.ai.client import AIClient
    from trendradar.core.loader import load_config

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    client = AIClient(cfg["AI"])

    labeled = load_labeled()
    todo = [r for r in rows if r["title"] not in labeled]
    print(f"[标注] 已标 {len(labeled)} 条，待标 {len(todo)} 条，每批 {BATCH_SIZE}")

    interests = INTERESTS_FILE.read_text(encoding="utf-8").strip()
    system_prompt = SYSTEM_PROMPT.format(interests=interests)

    write_header = not TOPICS_CSV.exists()
    with TOPICS_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "keep", "platform", "rank", "day"])
        if write_header:
            writer.writeheader()

        for start in range(0, len(todo), BATCH_SIZE):
            chunk = todo[start:start + BATCH_SIZE]
            numbered = {i: r for i, r in enumerate(chunk, 1)}
            news_list = "\n".join(f"{i}. {r['title']}" for i, r in numbered.items())
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TEMPLATE.format(
                    count=len(chunk), news_list=news_list)},
            ]
            batch_no = start // BATCH_SIZE + 1
            total_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
            if max_batches and batch_no > max_batches:
                print(f"[停止] 已达到 --max-batches {max_batches}，剩下的下次继续")
                break
            try:
                resp = client.chat(messages)
                keeps = parse_keeps(resp, list(numbered.keys()))
            except Exception as e:
                print(f"[批次 {batch_no}/{total_batches}] 失败，跳过: {type(e).__name__}: {e}")
                continue

            for i, row in numbered.items():
                writer.writerow({**row, "keep": "true" if keeps[i] else "false"})
            f.flush()
            n_keep = sum(1 for v in keeps.values() if v)
            print(f"[批次 {batch_no}/{total_batches}] {len(chunk)} 条 → 选题 {n_keep} / 噪音 {len(chunk) - n_keep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-only", action="store_true", help="只导出标题，不调 AI")
    ap.add_argument("--max-batches", type=int, default=0, help="本次最多跑几批（0=不限，用于分段跑）")
    args = ap.parse_args()

    rows = export_titles()
    if args.export_only:
        return
    label(rows, max_batches=args.max_batches)

    labeled = load_labeled()
    n_keep = sum(1 for v in labeled.values() if v == "true")
    print(f"\n[完成] 已标注 {len(labeled)} 条：选题 {n_keep} / 噪音 {len(labeled) - n_keep}")
    print(f"        → {TOPICS_CSV}")


if __name__ == "__main__":
    sys.exit(main())
