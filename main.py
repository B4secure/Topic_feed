import os
import re
import glob
import json
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import feedparser
import pandas as pd
import numpy as np
from dateutil import parser as dateparser

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# CONFIG (can be overridden by GitHub Actions env vars)
# ---------------------------
RUN_MODE = os.getenv("RUN_MODE", "daily").lower()

if RUN_MODE == "weekly":
    PAST_DAYS = int(os.getenv("PAST_DAYS_WEEKLY", "7"))
else:
    PAST_DAYS = int(os.getenv("PAST_DAYS_DAILY", "1"))

MAX_ITEMS = int(os.getenv("MAX_ITEMS", "50"))
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.60"))
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

HL, GL, CEID = "en-GB", "GB", "GB:en"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DAILY_DIR = DATA_DIR / "daily"
WEEKLY_DIR = DATA_DIR / "weekly"
DAILY_DIR.mkdir(exist_ok=True)
WEEKLY_DIR.mkdir(exist_ok=True)

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)


# ---------------------------
# Search library
# ---------------------------
SEARCH_LIBRARY_TEXT = r"""
Current_Affairs	(geopolitics OR migration OR economy OR conflict OR terrorism OR disinformation OR "government policy" OR legislation OR sanctions) AND (Europe OR UK OR "United States" OR global OR international)
Supply_Chain    ("supply chain" OR logistics OR shipping OR freight OR port OR tariffs OR sanctions OR embargoes OR reshoring OR nearshoring OR compliance OR "supply chain disruption" OR fragmentation OR instability) AND (company OR manufacturer OR factory OR supplier OR export OR import OR production)
Protest	(protest OR demonstration OR unrest OR boycott OR strike OR march OR "civil unrest") AND (retail OR "shopping centre" OR "town centre" OR luxury OR brand OR police OR arrested OR violence OR clashes)
Technology    (technology OR cybersecurity OR ransomware OR hacking OR AI OR "artificial intelligence" OR "machine learning" OR automation OR quantum OR semiconductor OR software OR cloud OR 5G OR robotics) AND (launch OR update OR vulnerability OR breach OR research OR earnings OR partnership OR regulation) -(rumor OR review OR gaming OR crypto OR podcast OR live OR blog)
Insider_Risk	("convicted" OR "sentenced" OR "pleaded guilty") AND ("employee" OR "worker" OR "staff") AND ("data" OR "theft" OR "fraud" OR "stolen")
Fraud_Scam	(fraud OR scam OR phishing OR "identity theft" OR "payment fraud" OR "money laundering" OR "cyber fraud" OR "bank fraud" OR "invoice fraud" OR "romance scam" OR "courier fraud" OR "retail fraud") AND (UK OR Britain OR Europe OR warning OR arrested OR convicted OR victim)
""".strip()


def parse_published_dt(published_str: str):
    if not published_str:
        return None
    try:
        dt = dateparser.parse(published_str)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def filter_last_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
    df = df.copy()
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)
    return df


def parse_search_library(text: str) -> pd.DataFrame:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if "\t" in line:
            name, query = line.split("\t", 1)
        else:
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) == 2:
                name, query = parts[0], parts[1]
            elif "(" in line:
                left, right = line.split("(", 1)
                name = left.strip()
                query = "(" + right.strip()
            else:
                rows.append({"search_name": "UNMAPPED_LINE", "raw_query": line})
                continue

        rows.append({"search_name": name.strip(), "raw_query": query.strip()})

    return pd.DataFrame(rows)


def is_google_news_compatible(q: str) -> bool:
    q = (q or "").strip().lower()
    if not q:
        return False
    if q.startswith("http://") or q.startswith("https://"):
        return False
    if "to:" in q or q.startswith("@") or " @" in q:
        return False
    return True


def google_news_rss_url(query: str, past_days: int) -> str:
    full = f"{query} when:{past_days}d"
    q = urllib.parse.quote(full)
    return f"https://news.google.com/rss/search?q={q}&hl={HL}&gl={GL}&ceid={CEID}"


def collect_google_news(df_searches: pd.DataFrame, past_days: int, max_items: int) -> pd.DataFrame:
    out_rows = []
    for _, r in df_searches.iterrows():
        name = r["search_name"]
        q = r["raw_query"]
        rss = google_news_rss_url(q, past_days)
        feed = feedparser.parse(rss)

        print(f"  {name}: Found {len(feed.entries)} raw entries")

        for entry in feed.entries[:max_items]:
            out_rows.append(
                {
                    "search_name": name,
                    "search_query": q,
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "past_days": past_days,
                }
            )

    if out_rows:
        temp_df = pd.DataFrame(out_rows)
        print("\nRaw collection summary:")
        for name in df_searches["search_name"].unique():
            count = len(temp_df[temp_df["search_name"] == name])
            print(f"  {name}: {count} articles")

    return pd.DataFrame(out_rows)


def remap_legacy_unmapped(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["search_name"].astype(str).eq("UNMAPPED_LINE")
    sq = df.loc[m, "raw_query"].astype(str)

    df.loc[m & sq.str.startswith("Insider Risk"), "search_name"] = "Insider_Risk"
    df.loc[m & sq.str.startswith("Fraud/Scam"), "search_name"] = "Fraud_Scam"

    df.loc[df["search_name"].eq("Insider_Risk") & df["raw_query"].astype(str).str.startswith("Insider Risk"),
           "raw_query"] = df["raw_query"].astype(str).str.replace(r"^Insider Risk\s*", "", regex=True)

    df.loc[df["search_name"].eq("Fraud_Scam") & df["raw_query"].astype(str).str.startswith("Fraud/Scam"),
           "raw_query"] = df["raw_query"].astype(str).str.replace(r"^Fraud/Scam\s*", "", regex=True)

    return df


def semantic_dedupe_excel(infile: str, out_clean: str, out_audit: str,
                          threshold: float, model_name: str) -> tuple[int, int]:
    df = pd.read_excel(infile)
    df["title"] = df["title"].fillna("").astype(str)

    if df.empty:
        df.to_excel(out_clean, index=False, engine="openpyxl")
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return 0, 0

    model = SentenceTransformer(model_name)

    keep_global_idx = set()
    audit_rows = []

    for topic, gdf in df.groupby("search_name", dropna=False):
        gdf = gdf.copy()
        gdf["compare_text"] = gdf["title"]
        mask = gdf["compare_text"].str.len() > 0
        gwork = gdf[mask].copy()

        if gwork.empty:
            keep_global_idx.update(gdf.index.tolist())
            continue

        emb = model.encode(
            gwork["compare_text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim = cosine_similarity(emb, emb)
        n = sim.shape[0]

        parent = list(range(n))
        rank = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    union(i, j)

        groups = {}
        gwork_idx = gwork.index.to_list()

        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        for members in groups.values():
            keep_i = min(members, key=lambda k: gwork_idx[k])
            keep_row = gwork_idx[keep_i]
            keep_global_idx.add(keep_row)

            for drop_i in members:
                drop_row = gwork_idx[drop_i]
                if drop_row == keep_row:
                    continue
                audit_rows.append({
                    "search_name": topic,
                    "kept_original_row": int(keep_row),
                    "dropped_original_row": int(drop_row),
                    "similarity": float(sim[keep_i, drop_i]),
                    "kept_title": df.loc[keep_row, "title"],
                    "dropped_title": df.loc[drop_row, "title"],
                })

    df_clean = df.loc[sorted(keep_global_idx)].reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)

    df_clean.to_excel(out_clean, index=False, engine="openpyxl")
    audit.to_excel(out_audit, index=False, engine="openpyxl")
    return len(df), len(df_clean)


def update_master_excel_rolling(new_df: pd.DataFrame, master_path: Path, keep_days: int):
    if master_path.exists():
        old_df = pd.read_excel(master_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined["past_days"] = keep_days
    combined = combined.drop_duplicates(subset=["link"]).reset_index(drop=True)
    combined["published_dt_utc"] = combined["published"].apply(parse_published_dt)
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    combined = combined[combined["published_dt_utc"].notna()]
    combined = combined[combined["published_dt_utc"] >= cutoff].copy()

    if 'published_dt_utc' in combined.columns:
        combined = combined.drop(columns=["published_dt_utc"], errors="ignore")

    combined.to_excel(master_path, index=False, engine="openpyxl")


# ---------------------------
# NEW: Export feed.json for dashboard
# ---------------------------
def export_feed_json(df: pd.DataFrame, past_days: int, run_mode: str):
    """Export deduplicated articles to docs/feed.json for the GitHub Pages dashboard."""
    run_type = "Weekly run" if run_mode == "weekly" else "Daily run"

    articles = []
    for _, row in df.iterrows():
        articles.append({
            "search_name":  str(row.get("search_name", "")),
            "search_query": str(row.get("search_query", "")),
            "title":        str(row.get("title", "")),
            "published":    str(row.get("published", "")),
            "link":         str(row.get("link", "")),
            "past_days":    int(row.get("past_days", past_days)),
        })

    payload = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "lookback_days": past_days,
        "run_type":      run_type,
        "articles":      articles,
    }

    output_path = DOCS_DIR / "feed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"feed.json written → {len(articles)} articles → {output_path}")


def main():
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    print(f"Parsed search library: {len(search_df)} rows")
    print(f"Search names found: {search_df['search_name'].unique()}")

    search_df = remap_legacy_unmapped(search_df)

    bad = search_df[search_df["search_name"] == "UNMAPPED_LINE"]
    if not bad.empty:
        print("UNMAPPED_LINE entries found:")
        print(bad["raw_query"].tolist())

    print(f"After remapping, search names: {search_df['search_name'].unique()}")

    search_df["google_news_compatible"] = search_df["raw_query"].apply(is_google_news_compatible)
    to_run = search_df[search_df["google_news_compatible"]].copy()

    print(f"\nRunning {len(to_run)} compatible searches:")
    for _, row in to_run.iterrows():
        print(f"  - {row['search_name']}: {row['raw_query'][:60]}...")

    results = collect_google_news(to_run, past_days=PAST_DAYS, max_items=MAX_ITEMS)
    results = filter_last_n_days(results, n_days=PAST_DAYS)

    if not results.empty:
        results = results.drop_duplicates(subset=["link"]).reset_index(drop=True)

    raw_results_file = DATA_DIR / f"google_news_raw_{ts}_past{PAST_DAYS}d.xlsx"
    audit_search_file = DATA_DIR / f"search_audit_{ts}.xlsx"

    if 'published_dt_utc' in results.columns:
        results = results.drop(columns=["published_dt_utc"], errors="ignore")

    results.to_excel(raw_results_file, index=False, engine="openpyxl")
    search_df.to_excel(audit_search_file, index=False, engine="openpyxl")

    print(f"\nResults by category:")
    if not results.empty:
        category_counts = results["search_name"].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} articles")
    else:
        print("  No results found")

    dedup_file = DATA_DIR / f"google_news_dedup_{ts}_past{PAST_DAYS}d.xlsx"
    dedup_audit = DATA_DIR / f"google_news_dedup_audit_{ts}.xlsx"

    orig, cleaned = semantic_dedupe_excel(
        infile=str(raw_results_file),
        out_clean=str(dedup_file),
        out_audit=str(dedup_audit),
        threshold=DUP_THRESHOLD,
        model_name=MODEL_NAME,
    )

    df_final = pd.read_excel(dedup_file)

    master = DATA_DIR / "topic_feeds.xlsx"
    update_master_excel_rolling(df_final, master, keep_days=7)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_file = DAILY_DIR / f"topic_feeds_daily.xlsx"
    df_final.to_excel(daily_file, index=False, engine="openpyxl")

    if RUN_MODE == "weekly":
        week = datetime.now(timezone.utc).strftime("%G-W%V")
        weekly_file = WEEKLY_DIR / f"topic_feeds_week.xlsx"
        df_final.to_excel(weekly_file, index=False, engine="openpyxl")
        weekly_latest = DATA_DIR / "topic_feeds_weekly_latest.xlsx"
        df_final.to_excel(weekly_latest, index=False, engine="openpyxl")

    # Export dashboard feed — use 7-day rolling master
    df_master = pd.read_excel(master)
    export_feed_json(df_master, 7, RUN_MODE)

    print(f"\nRUN_MODE={RUN_MODE}")
    print(f"Master updated: {master}")
    print(f"Daily snapshot: {daily_file}")
    if RUN_MODE == "weekly":
        print(f"Weekly snapshot: {weekly_file}")


if __name__ == "__main__":
    main()
