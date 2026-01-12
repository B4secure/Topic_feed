import os
import re
import glob
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import shutil

import feedparser
import pandas as pd
import numpy as np
from dateutil import parser as dateparser

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# CONFIG (can be overridden by GitHub Actions env vars)
# ---------------------------
PAST_DAYS = int(os.getenv("PAST_DAYS", "7"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "50"))
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

HL, GL, CEID = "en-GB", "GB", "GB:en"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------
# search library 
# ---------------------------
SEARCH_LIBRARY_TEXT = r"""
Emissions Monitoring | All Sectors | UK	(UK OR "United Kingdom" OR Britain OR England OR Scotland OR Wales OR "Northern Ireland") AND (carbon emissions OR CO2 emissions OR "greenhouse gas emissions")
Net Zero Strategy | All Sectors | UK	(UK OR "United Kingdom") AND (net zero OR decarbonisation OR "carbon reduction targets" OR "emissions target")
Climate Policy | All Sectors | UK	(UK OR "United Kingdom") AND ("climate policy" OR "carbon regulation" OR "emission standards" OR "carbon tax" OR "ETS")

Emissions Monitoring | Power | UK	(UK OR "United Kingdom") AND ("power sector emissions" OR "electricity generation emissions" OR "power station CO2" OR "grid emissions")
Energy Transition | Power | UK	(UK OR "United Kingdom") AND ("renewable energy transition" OR "clean energy investment" OR "energy transition" OR "renewable rollout")
Fossil Fuels | Power | UK	(UK OR "United Kingdom") AND ("coal power emissions" OR "gas power emissions" OR "fossil fuel power stations")

Emissions Monitoring | Transport | UK	(UK OR "United Kingdom") AND ("transport emissions" OR "vehicle emissions" OR "road transport emissions")
Transport Policy | Transport | UK	(UK OR "United Kingdom") AND ("electric vehicles policy" OR "EV mandate" OR "ULEZ" OR "clean air zone" OR "transport decarbonisation")
Aviation & Shipping | Transport | UK	(UK OR "United Kingdom") AND ("aviation emissions" OR "shipping emissions" OR "maritime emissions" OR "SAF")

Emissions Monitoring | Industry | UK	(UK OR "United Kingdom") AND ("industrial emissions" OR "manufacturing CO2" OR "factory emissions")
Heavy Industry | Industry | UK	(UK OR "United Kingdom") AND ("cement emissions" OR "steel emissions" OR "heavy industry emissions")
Industrial Technology | Industry | UK	(UK OR "United Kingdom") AND ("carbon capture" OR CCS OR "industrial decarbonisation" OR "clean industrial")

Emissions Monitoring | Buildings | UK	(UK OR "United Kingdom") AND ("building emissions" OR "heating emissions" OR "home insulation" OR "energy efficiency buildings")
Agriculture & Methane | Agriculture | UK	(UK OR "United Kingdom") AND ("agriculture emissions" OR "methane emissions" OR "farming emissions")
Waste & Landfill | Waste | UK	(UK OR "United Kingdom") AND ("waste emissions" OR "landfill emissions" OR "waste management emissions")

Emissions Monitoring | All Sectors | Europe	(Europe OR "European Union" OR EU) AND (carbon emissions OR CO2 emissions OR "greenhouse gas emissions")
Net Zero Strategy | All Sectors | Europe	(Europe OR "European Union" OR EU) AND (net zero OR decarbonisation OR "emissions target" OR "fit for 55")
Climate Policy | All Sectors | Europe	(Europe OR "European Union" OR EU) AND ("climate policy" OR ETS OR "carbon border adjustment" OR CBAM OR "emission standards")

Emissions Monitoring | Power | Europe	(Europe OR "European Union" OR EU) AND ("power sector emissions" OR "electricity emissions" OR "grid emissions")
Energy Transition | Power | Europe	(Europe OR "European Union" OR EU) AND ("renewable energy transition" OR "clean energy investment" OR "energy transition")
Fossil Fuels | Power | Europe	(Europe OR "European Union" OR EU) AND ("coal power emissions" OR "gas power emissions")

Emissions Monitoring | Transport | Europe	(Europe OR "European Union" OR EU) AND ("transport emissions" OR "vehicle emissions")
Transport Policy | Transport | Europe	(Europe OR "European Union" OR EU) AND ("electric vehicles policy" OR "transport decarbonisation" OR "Euro 7" OR "CO2 standards cars")
Aviation & Shipping | Transport | Europe	(Europe OR "European Union" OR EU) AND ("aviation emissions" OR "shipping emissions" OR "ReFuelEU" OR "FuelEU Maritime")

Emissions Monitoring | Industry | Europe	(Europe OR "European Union" OR EU) AND ("industrial emissions" OR "manufacturing CO2")
Heavy Industry | Industry | Europe	(Europe OR "European Union" OR EU) AND ("cement emissions" OR "steel emissions")
Industrial Technology | Industry | Europe	(Europe OR "European Union" OR EU) AND ("carbon capture" OR CCS OR "industrial decarbonisation")

Emissions Monitoring | Buildings | Europe	(Europe OR "European Union" OR EU) AND ("building emissions" OR "heating emissions" OR "energy efficiency buildings")
Agriculture & Methane | Agriculture | Europe	(Europe OR "European Union" OR EU) AND ("agriculture emissions" OR "methane emissions")
Waste & Landfill | Waste | Europe	(Europe OR "European Union" OR EU) AND ("waste emissions" OR "landfill emissions")

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


def filter_last_n_hours(df, hours: int):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    df = df.copy()
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)
    return df


def parse_search_library(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            rows.append({"search_name": "UNMAPPED_LINE", "raw_query": line})
            continue
        name, query = line.split("\t", 1)
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
    return pd.DataFrame(out_rows)


def latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]


def semantic_dedupe_csv(infile: str, out_clean: str, out_audit: str,
                        threshold: float, model_name: str) -> tuple[int, int]:
    df = pd.read_excel(infile)
    df["compare_text"] = df["title"].fillna("").astype(str)

    mask = df["compare_text"].str.len() > 0
    df_work = df[mask].copy().reset_index(drop=True)
    orig_idx = df.index[mask].to_numpy()

    if df_work.empty:
        df.drop(columns=["compare_text"], errors="ignore").to_excel(out_clean, index=False)
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return len(df), len(df)

    model = SentenceTransformer(model_name)
    emb = model.encode(
        df_work["compare_text"].tolist(),
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
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    keep_work = set()
    audit_rows = []

    for g in groups.values():
        if len(g) == 1:
            keep_work.add(g[0])
            continue

        g_map = [(int(orig_idx[i]), i) for i in g]
        g_map.sort(key=lambda x: x[0])

        keep_orig, keep_i = g_map[0]
        keep_work.add(keep_i)

        for drop_orig, drop_i in g_map[1:]:
            audit_rows.append(
                {
                    "kept_original_row": keep_orig,
                    "dropped_original_row": int(drop_orig),
                    "similarity": float(sim[keep_i, drop_i]),
                    "kept_title": df.loc[keep_orig, "title"],
                    "dropped_title": df.loc[int(drop_orig), "title"],
                }
            )

    kept_orig_rows = {int(orig_idx[i]) for i in keep_work}
    drop_orig_rows = set(map(int, orig_idx.tolist())) - kept_orig_rows

    keep_mask = np.ones(len(df), dtype=bool)
    for r in drop_orig_rows:
        keep_mask[r] = False

    df_clean = df.loc[keep_mask].drop(columns=["compare_text"], errors="ignore").reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)

    df_clean.to_excel(out_clean, index=False, engine = "openpyxl")
    audit.to_excel(out_audit, index=False, engine = "openpyxl")
    return len(df), len(df_clean)

def split_search_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly split search_name into theme/sector/location.
    Expected format: Theme | Sector | Location (spaces optional).
    If malformed, fills Unknown.
    """
    s = df.get("search_name", pd.Series([""] * len(df))).astype(str)

    # Split on '|' with optional surrounding whitespace
    parts = s.str.split(r"\s*\|\s*", n=2, expand=True)

    # Ensure 3 columns exist
    for i in range(3):
        if i not in parts.columns:
            parts[i] = ""

    df["theme"] = parts[0].replace("", np.nan).fillna("Unknown").str.strip()
    df["sector"] = parts[1].replace("", np.nan).fillna("Unknown").str.strip()
    df["location"] = parts[2].replace("", np.nan).fillna("Unknown").str.strip()
    return df




def main():
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    search_df["google_news_compatible"] = search_df["raw_query"].apply(is_google_news_compatible)
    to_run = search_df[search_df["google_news_compatible"]].copy()

    results = collect_google_news(to_run, past_days=PAST_DAYS, max_items=MAX_ITEMS)
    

    if not results.empty:
        results = results.drop_duplicates(subset=["link"]).reset_index(drop=True)
        results = split_search_name(results)

        results = results.rename(columns={
    "published": "published_date",
    "link": "url",
    "search_query": "keyword_query"
})



    raw_results_file = DATA_DIR / f"google_news_raw_{ts}_past{PAST_DAYS}d.xlsx"
    audit_search_file = DATA_DIR / f"search_audit_{ts}.xlsx"

    
    results = results.apply
    (lambda s: s.dt.tz_localize(None) if hasattr(s, "dt") and getattr(s.dt, "tz", None) is not None else s)

    results.to_excel(raw_results_file, index=False, engine = "openpyxl")
    search_df.to_excel(audit_search_file, index=False, engine = "openpyxl")

    # Dedupe the raw file we just created
    dedup_file = DATA_DIR / f"google_news_dedup_{ts}_past{PAST_DAYS}d.xlsx"
    dedup_audit = DATA_DIR / f"google_news_dedup_audit_{ts}.xlsx"

    orig, cleaned = semantic_dedupe_csv(
        infile=str(raw_results_file),
        out_clean=str(dedup_file),
        out_audit=str(dedup_audit),
        threshold=DUP_THRESHOLD,
        model_name=MODEL_NAME,
    )


    # Always keep a stable single file for automation
    latest = DATA_DIR / "latest_deduped.xlsx"
    shutil.copyfile(dedup_file, latest)
    print(f"Saved latest: {latest}")


    print(f"Saved raw:   {raw_results_file} | rows={len(results)}")
    print(f"Saved audit: {audit_search_file} | searches={len(search_df)}")
    print(f"Dedupe: original={orig} cleaned={cleaned}")
    print(f"Saved dedup: {dedup_file}")
    print(f"Saved dedup audit: {dedup_audit}")
   


if __name__ == "__main__":
    main()


# %%











