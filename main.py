import os
import re
import glob
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
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.60"))  # FIXED env var name
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

HL, GL, CEID = "en-GB", "GB", "GB:en"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DAILY_DIR = DATA_DIR / "daily"
WEEKLY_DIR = DATA_DIR / "weekly"
DAILY_DIR.mkdir(exist_ok=True)
WEEKLY_DIR.mkdir(exist_ok=True)



# ---------------------------
# Search library (FIXED: Insider/Fraud now tab-delimited)
# ---------------------------
SEARCH_LIBRARY_TEXT = r"""
Current_Affairs    (geopolitics OR migration OR economy OR conflict OR legislation OR terrorism OR disinformation OR misinformation OR government OR policy OR regulation) AND (briefing OR analysis OR update OR report)
Supply_Chain    ("supply chain" OR logistics OR shipping OR freight OR port OR tariffs OR sanctions OR embargoes OR reshoring OR nearshoring OR compliance OR "supply chain disruption" OR fragmentation OR instability) AND (company OR manufacturer OR factory OR supplier OR export OR import OR production)
Protest    (protest OR protests OR demonstration OR demonstrations OR unrest OR "civil resistance" OR "civil disobedience" OR boycott OR boycotts OR march OR marches OR strike OR strikes) AND (police OR capital OR city OR arrests OR arrested OR union OR workers OR students OR rally OR tactics OR targets OR groups)
Technology    (technology OR cybersecurity OR ransomware OR hacking OR AI OR "artificial intelligence" OR "machine learning" OR automation OR quantum OR semiconductor OR software OR cloud OR 5G OR robotics) AND (launch OR update OR vulnerability OR breach OR research OR earnings OR partnership OR regulation) -(rumor OR review OR gaming OR crypto OR podcast OR live OR blog)
Insider_Risk    ("insider risk" OR "insider threat" OR "internal threat" OR "employee misconduct" OR "data exfiltration" OR "privileged access abuse" OR "malicious insider" OR "negligent insider" OR "insider attack" OR "corporate espionage" OR "information leakage" OR "unauthorised access" OR "policy violation" OR "security breach" OR "insider vulnerability")
Fraud_Scam    ("fraud" OR "scam" OR "phishing" OR "identity theft" OR "account takeover" OR "payment fraud" OR "credit card fraud" OR "wire fraud" OR "business email compromise" OR "fake invoice" OR "social engineering" OR "advance fee fraud")

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

        # Prefer tab, but accept 2+ spaces as delimiter
        if "\t" in line:
            name, query = line.split("\t", 1)
        else:
            # split into 2 parts on 2+ spaces
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
        
        # DEBUG: Print how many entries found
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
    
    # DEBUG: Show summary
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

    sq = df.loc[m, "raw_query"].astype(str)  # FIXED: use raw_query not search_query

    df.loc[m & sq.str.startswith("Insider Risk"), "search_name"] = "Insider_Risk"
    df.loc[m & sq.str.startswith("Fraud/Scam"), "search_name"] = "Fraud_Scam"

    # optional: strip the legacy prefix from raw_query (keeps query cleaner)
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

    # Dedupe within each topic (search_name)
    for topic, gdf in df.groupby("search_name", dropna=False):
        gdf = gdf.copy()

        # compare only titles WITHIN this topic
        gdf["compare_text"] = gdf["title"]
        mask = gdf["compare_text"].str.len() > 0
        gwork = gdf[mask].copy()

        # if nothing to compare, keep all rows in this topic
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
            # Keep earliest original row index within this topic
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

    combined["past_days"] = keep_days  # <-- forces consistent value

    combined = combined.drop_duplicates(subset=["link"]).reset_index(drop=True)

    combined["published_dt_utc"] = combined["published"].apply(parse_published_dt)
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    combined = combined[combined["published_dt_utc"].notna()]
    combined = combined[combined["published_dt_utc"] >= cutoff].copy()
    
    # REMOVE timezone info before writing to Excel
    if 'published_dt_utc' in combined.columns:
        combined = combined.drop(columns=["published_dt_utc"], errors="ignore")

    combined.to_excel(
        master_path, index=False, engine="openpyxl"
    )



def main():
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    print(f"Parsed search library: {len(search_df)} rows")
    print(f"Search names found: {search_df['search_name'].unique()}")
    
    # FIX: Apply the remapping function
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

    # FIX: Remove timezone info before writing to Excel
    # Don't use the apply method you had, just handle datetime columns properly
    if 'published_dt_utc' in results.columns:
        results = results.drop(columns=["published_dt_utc"], errors="ignore")

    # Write to Excel WITHOUT datetime timezone info
    results.to_excel(raw_results_file, index=False, engine="openpyxl")
    search_df.to_excel(audit_search_file, index=False, engine="openpyxl")
    
    print(f"\nResults by category:")
    if not results.empty:
        category_counts = results["search_name"].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} articles")
    else:
        print("  No results found")

    # Dedupe the raw file we just created
    dedup_file = DATA_DIR / f"google_news_dedup_{ts}_past{PAST_DAYS}d.xlsx"
    dedup_audit = DATA_DIR / f"google_news_dedup_audit_{ts}.xlsx"

    orig, cleaned = semantic_dedupe_excel(
        infile=str(raw_results_file),
        out_clean=str(dedup_file),
        out_audit=str(dedup_audit),
        threshold=DUP_THRESHOLD,
        model_name=MODEL_NAME,
    )

    # Update rolling master
    # Final deduped dataset from this run
    df_final = pd.read_excel(dedup_file)

    # 1) Update rolling master (recommended: always keep 7 days)
    master = DATA_DIR / "topic_feeds.xlsx"
    update_master_excel_rolling(df_final, master, keep_days=7)

    # 2) Save daily snapshot (always)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_file = DAILY_DIR / f"topic_feeds_daily.xlsx"
    df_final.to_excel(daily_file, index=False, engine="openpyxl")

    # 3) Save weekly snapshot (only on weekly runs)
    if RUN_MODE == "weekly":
        week = datetime.now(timezone.utc).strftime("%G-W%V")  # e.g., 2026-W05
        weekly_file = WEEKLY_DIR / f"topic_feeds_week.xlsx"
        df_final.to_excel(weekly_file, index=False, engine="openpyxl")
        weekly_latest = DATA_DIR / "topic_feeds_weekly_latest.xlsx"
        df_final.to_excel(weekly_latest, index=False, engine="openpyxl")

    print(f"\nRUN_MODE={RUN_MODE}")
    print(f"Master updated: {master}")
    print(f"Daily snapshot: {daily_file}")
    if RUN_MODE == "weekly":
        print(f"Weekly snapshot: {weekly_file}")



if __name__ == "__main__":
    main()
