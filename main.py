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
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException


# ---------------------------
# CONFIG
# ---------------------------

RUN_MODE = os.getenv("RUN_MODE", "daily").lower()

if RUN_MODE == "weekly":
    PAST_DAYS = int(os.getenv("PAST_DAYS_WEEKLY", "7"))
else:
    PAST_DAYS = int(os.getenv("PAST_DAYS_DAILY", "1"))

MAX_ITEMS     = int(os.getenv("MAX_ITEMS", "50"))
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.60"))
MODEL_NAME    = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

HL, GL, CEID = "en-GB", "GB", "GB:en"

DATA_DIR   = Path("data")
DAILY_DIR  = DATA_DIR / "daily"
WEEKLY_DIR = DATA_DIR / "weekly"
DOCS_DIR   = Path("docs")

for d in [DATA_DIR, DAILY_DIR, WEEKLY_DIR, DOCS_DIR]:
    d.mkdir(exist_ok=True)


# ---------------------------
# EVENT TAXONOMY THEMES (4 categories for semantic sweep)
# ---------------------------

EVENT_TAXONOMY_THEMES = [
    # Current Affairs (includes protest and civil unrest)
    "government policy legislation regulation international relations geopolitics",
    "conflict war sanctions trade restrictions political crisis",
    "migration refugee crisis border policy",
    "disinformation propaganda fake news media manipulation",
    "protest demonstration march rally strike civil unrest",
    "riot disorder looting anti-social behaviour dispersal order",
    "fuel protest farmers protest general strike national walkout",
    "flash mob gathering crowd disorder public disturbance",
    "far right far left extremism hate crime antisemitic attack",
    "boycott brand campaign consumer backlash",
    # Supply Chain
    "supply chain disruption logistics shipping delay freight",
    "port strike transport blockade haulage road closure disruption",
    "fuel protest farmers blockade lorry driver trucker strike",
    "trade war tariffs import export ban sanctions embargo",
    "raw material shortage component shortage factory closure",
    "border delay customs backlog reshoring nearshoring",
    "freight disruption delivery delay warehouse strike",
    # Technology and Cyber
    "cybersecurity data breach ransomware hack malware attack",
    "phishing zero day vulnerability software exploit",
    "cyber attack infrastructure critical systems outage",
    "artificial intelligence automation technology regulation",
    "surveillance facial recognition biometrics privacy",
    "semiconductor chip shortage technology supply disruption",
    "deepfake identity fraud technology crime",
    # Fraud and Insider Risk
    "fraud scam phishing identity theft financial crime money laundering",
    "payment fraud bank fraud card fraud APP fraud mandate fraud",
    "investment scam crypto fraud Ponzi scheme romance scam",
    "employee theft staff fraud workplace misconduct insider threat",
    "data theft stolen data leaked confidential information sabotage",
    "convicted sentenced pleaded guilty employee fraud embezzlement",
    "bribery corruption insider dealing trade secret theft",
]


# ---------------------------
# SEMANTIC SWEEP SEARCHES (4 categories only)
# ---------------------------

SEMANTIC_SWEEP_SEARCHES = [
    # Current Affairs
    {"search_name": "Current_Affairs", "query": "Europe politics government policy protest civil unrest"},
    {"search_name": "Current_Affairs", "query": "UK politics government policy protest demonstration"},
    {"search_name": "Current_Affairs", "query": "conflict sanctions geopolitics international relations"},
    {"search_name": "Current_Affairs", "query": "protest strike riot disorder Europe UK civil unrest"},
    {"search_name": "Current_Affairs", "query": "fuel protest farmers strike general strike blockade Europe"},
    {"search_name": "Current_Affairs", "query": "migration refugees border crisis Europe UK"},
    {"search_name": "Current_Affairs", "query": "hate crime antisemitic extremism far right attack Europe"},
    # Supply Chain
    {"search_name": "Supply_Chain", "query": "supply chain disruption logistics shipping freight delay"},
    {"search_name": "Supply_Chain", "query": "transport strike blockade road closure haulage disruption"},
    {"search_name": "Supply_Chain", "query": "fuel protest farmers blockade lorry driver motorway"},
    {"search_name": "Supply_Chain", "query": "trade war tariffs import export ban sanctions"},
    {"search_name": "Supply_Chain", "query": "port strike shipping delay freight disruption"},
    {"search_name": "Supply_Chain", "query": "raw material shortage component shortage factory closure"},
    {"search_name": "Supply_Chain", "query": "border delay customs backlog supply disruption"},
    # Technology
    {"search_name": "Technology", "query": "cybersecurity data breach ransomware hack malware"},
    {"search_name": "Technology", "query": "cyber attack phishing zero day vulnerability exploit"},
    {"search_name": "Technology", "query": "artificial intelligence technology regulation policy"},
    {"search_name": "Technology", "query": "surveillance biometrics facial recognition privacy"},
    {"search_name": "Technology", "query": "semiconductor chip shortage technology supply"},
    # Fraud and Insider Risk
    {"search_name": "Fraud_Scam", "query": "fraud scam financial crime money laundering UK Europe"},
    {"search_name": "Fraud_Scam", "query": "payment fraud bank fraud card fraud APP fraud mandate"},
    {"search_name": "Fraud_Scam", "query": "investment scam crypto fraud Ponzi scheme"},
    {"search_name": "Fraud_Scam", "query": "employee arrested convicted fraud theft data stolen workplace"},
    {"search_name": "Fraud_Scam", "query": "insider threat staff fraud embezzlement bribery corruption"},
]


# ---------------------------
# SEARCH LIBRARY (4 categories)
# ---------------------------

SEARCH_LIBRARY_TEXT = r"""
Current_Affairs	(geopolitics OR migration OR economy OR conflict OR terrorism OR disinformation OR "government policy" OR legislation OR sanctions OR protest OR demonstration OR "civil unrest" OR riot OR strike OR blockade OR "fuel protest" OR "farmers protest" OR "general strike" OR disorder OR "far right" OR "far left" OR extremism OR "hate crime" OR "antisemitic" OR "Islamophobic" OR "dispersal order" OR "anti-social behaviour" OR "flash mob") AND (Europe OR UK OR "United States" OR global OR international OR France OR Germany OR Spain OR Italy OR Belgium OR Netherlands OR Ireland)
Supply_Chain	("supply chain" OR logistics OR shipping OR freight OR port OR tariffs OR sanctions OR embargoes OR reshoring OR nearshoring OR "supply chain disruption" OR fragmentation OR instability OR "road closure" OR "port strike" OR "transport strike" OR "freight disruption" OR "fuel protest" OR "farmers protest" OR "lorry driver" OR "trucker strike" OR haulage OR "border delay" OR "customs delay" OR "import ban" OR "export ban" OR "trade war" OR "trade disruption" OR "raw material shortage" OR "component shortage") AND (company OR manufacturer OR factory OR supplier OR export OR import OR production OR delivery OR route OR network OR Europe OR UK OR global)
Technology	(technology OR cybersecurity OR ransomware OR hacking OR "data breach" OR malware OR phishing OR "cyber attack" OR "zero day" OR vulnerability OR "artificial intelligence" OR automation OR semiconductor OR software OR cloud OR "5G" OR robotics OR "quantum computing" OR IoT OR deepfake OR surveillance OR biometrics OR "facial recognition") AND (launch OR update OR breach OR attack OR warning OR research OR regulation OR arrested OR convicted OR victim OR company OR government) AND -(rumor OR gaming OR podcast OR "live blog" OR "product review")
Tech   (technology OR cybersecurity OR "zero day" OR vulnerability OR malware OR software OR cloud OR IoT OR automation OR robotics OR "artificial intelligence" OR AI OR "machine learning" OR "deep learning" OR "generative AI" OR LLM OR semiconductor OR chip OR "edge computing" OR "quantum computing" OR "quantum tech" OR "5G" OR "6G" OR "digital twin" OR "blockchain" OR "web3" OR "extended reality" OR XR OR AR OR VR OR metaverse OR "autonomous systems" OR drones OR "smart cities" OR "facial recognition" OR biometrics OR surveillance OR "computer vision" OR "synthetic media" OR deepfake OR "cyber-physical systems"OR biotech OR "biotechnology" OR "synthetic biology" OR genomics OR "health tech" OR medtech OR "neurotechnology"OR "climate tech" OR greentech OR "clean technology" OR "energy storage" OR battery OR hydrogen OR "carbon capture"OR "advanced materials" OR nanotechnology OR "space tech" OR satellite OR "aerospace technology")AND(launch OR update OR release OR develop OR innovation OR breakthrough OR research OR study OR pilot OR trial OR partnership OR investment OR funding OR acquisition OR merger OR expansion OR deployment OR adoption OR rollout OR regulation OR policy OR compliance OR warning OR risk OR breach OR attack OR incident OR arrest OR conviction OR lawsuit)AND-(rumor OR gaming OR podcast OR "live blog" OR "product review" OR "stock price" OR "celebrity")
Fraud_Scam	(fraud OR scam OR phishing OR "identity theft" OR "payment fraud" OR "money laundering" OR "cyber fraud" OR "bank fraud" OR "invoice fraud" OR "romance scam" OR "courier fraud" OR "retail fraud" OR "card fraud" OR "mandate fraud" OR "authorised push payment" OR "APP fraud" OR "investment fraud" OR "crypto fraud" OR counterfeit OR "false accounting" OR embezzlement OR bribery OR corruption OR "insider dealing" OR "convicted" OR "sentenced" OR "pleaded guilty" OR "employee theft" OR "staff fraud" OR "workplace fraud" OR "data theft" OR "stolen data" OR "leaked data" OR "rogue employee" OR "disgruntled employee" OR sabotage OR "intellectual property theft" OR "trade secret" OR "charged with" OR "arrested for" OR "dismissed for") AND (UK OR Britain OR Europe OR warning OR arrested OR convicted OR victim OR company OR employee OR worker OR staff OR manager OR director)
""".strip()


# ---------------------------
# HELPERS
# ---------------------------

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
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            name, query = line.split("\t", 1)
        else:
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) == 2:
                name, query = parts[0], parts[1]
            elif "(" in line:
                left, right = line.split("(", 1)
                name  = left.strip()
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


def google_news_rss_url(query: str, past_days: int,
                        hl: str = "en-GB", gl: str = "GB", ceid: str = "GB:en") -> str:
    full = f"{query} when:{past_days}d"
    q = urllib.parse.quote(full)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def remap_legacy_unmapped(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m  = df["search_name"].astype(str).eq("UNMAPPED_LINE")
    sq = df.loc[m, "raw_query"].astype(str)
    df.loc[m & sq.str.startswith("Insider Risk"), "search_name"] = "Fraud_Scam"
    df.loc[m & sq.str.startswith("Fraud/Scam"),   "search_name"] = "Fraud_Scam"
    df.loc[df["search_name"].eq("Fraud_Scam") & df["raw_query"].astype(str).str.startswith("Fraud/Scam"),
           "raw_query"] = df["raw_query"].astype(str).str.replace(r"^Fraud/Scam\s*", "", regex=True)
    return df


# ---------------------------
# KEYWORD COLLECTION
# ---------------------------

def collect_google_news(df_searches: pd.DataFrame, past_days: int, max_items: int) -> pd.DataFrame:
    out_rows = []
    for _, r in df_searches.iterrows():
        name = r["search_name"]
        q    = r["raw_query"]
        rss  = google_news_rss_url(q, past_days)
        feed = feedparser.parse(rss)
        print(f"  {name}: {len(feed.entries)} raw entries")
        for entry in feed.entries[:max_items]:
            out_rows.append({
                "search_name":  name,
                "search_query": q,
                "title":        entry.get("title", ""),
                "published":    entry.get("published", ""),
                "link":         entry.get("link", ""),
                "past_days":    past_days,
                "source":       "google_rss",
            })

    if out_rows:
        temp_df = pd.DataFrame(out_rows)
        print("\nRaw collection summary:")
        for name in df_searches["search_name"].unique():
            count = len(temp_df[temp_df["search_name"] == name])
            print(f"  {name}: {count} articles")

    return pd.DataFrame(out_rows)


# ---------------------------
# SEMANTIC DEDUPE
# ---------------------------

def semantic_dedupe_excel(infile: str, out_clean: str, out_audit: str,
                          threshold: float, model_name: str) -> tuple:
    df = pd.read_excel(infile)
    df["title"] = df["title"].fillna("").astype(str)

    if df.empty:
        df.to_excel(out_clean, index=False, engine="openpyxl")
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return 0, 0

    model           = SentenceTransformer(model_name)
    keep_global_idx = set()
    audit_rows      = []

    for topic, gdf in df.groupby("search_name", dropna=False):
        gdf  = gdf.copy()
        gdf["compare_text"] = gdf["title"]
        mask  = gdf["compare_text"].str.len() > 0
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
        n   = sim.shape[0]

        parent = list(range(n))
        rank   = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return
            if rank[ra] < rank[rb]:   parent[ra] = rb
            elif rank[ra] > rank[rb]: parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    union(i, j)

        groups    = {}
        gwork_idx = gwork.index.to_list()
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        for members in groups.values():
            keep_i   = min(members, key=lambda k: gwork_idx[k])
            keep_row = gwork_idx[keep_i]
            keep_global_idx.add(keep_row)
            for drop_i in members:
                drop_row = gwork_idx[drop_i]
                if drop_row == keep_row:
                    continue
                audit_rows.append({
                    "search_name":          topic,
                    "kept_original_row":    int(keep_row),
                    "dropped_original_row": int(drop_row),
                    "similarity":           float(sim[keep_i, drop_i]),
                    "kept_title":           df.loc[keep_row, "title"],
                    "dropped_title":        df.loc[drop_row, "title"],
                })

    df_clean = df.loc[sorted(keep_global_idx)].reset_index(drop=True)
    audit    = pd.DataFrame(audit_rows)
    df_clean.to_excel(out_clean, index=False, engine="openpyxl")
    audit.to_excel(out_audit,   index=False, engine="openpyxl")
    return len(df), len(df_clean)


# ---------------------------
# ROLLING MASTER EXCEL
# ---------------------------

def update_master_excel_rolling(new_df: pd.DataFrame, master_path: Path, keep_days: int):
    if master_path.exists():
        old_df   = pd.read_excel(master_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined["past_days"] = keep_days
    combined = combined.drop_duplicates(subset=["link"]).reset_index(drop=True)
    combined["published_dt_utc"] = combined["published"].apply(parse_published_dt)
    cutoff   = datetime.now(timezone.utc) - timedelta(days=keep_days)
    combined = combined[combined["published_dt_utc"].notna()]
    combined = combined[combined["published_dt_utc"] >= cutoff].copy()
    combined = combined.drop(columns=["published_dt_utc"], errors="ignore")
    combined.to_excel(master_path, index=False, engine="openpyxl")


# ---------------------------
# SEMANTIC SWEEP
# Safety net — finds relevant articles keyword searches missed.
# Uses broad topic queries + semantic similarity scoring.
# Results are mapped back to the 4 categories so your team
# sees clean category labels, not "Semantic — X" names.
# ---------------------------

def semantic_sweep(existing_links: set, past_days: int,
                   max_items: int, model_name: str) -> pd.DataFrame:

    print(f"\n{'='*60}")
    print(f"SEMANTIC SWEEP — broad topic searches + relevance scoring")
    print(f"  Searches: {len(SEMANTIC_SWEEP_SEARCHES)}  |  Threshold: 0.30")
    print(f"{'='*60}\n")

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    theme_vectors = model.encode(
        EVENT_TAXONOMY_THEMES,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"✅ {len(EVENT_TAXONOMY_THEMES)} theme vectors built\n")

    all_articles = []

    for search in SEMANTIC_SWEEP_SEARCHES:
        url = google_news_rss_url(search["query"], past_days)
        try:
            feed      = feedparser.parse(url)
            new_count = 0
            for entry in feed.entries[:max_items]:
                link = entry.get("link", "")
                if link in existing_links:
                    continue
                all_articles.append({
                    "search_name":  search["search_name"],  # already one of the 4 categories
                    "search_query": search["query"],
                    "title":        entry.get("title", ""),
                    "published":    entry.get("published", ""),
                    "link":         link,
                    "past_days":    past_days,
                    "source":       "semantic_sweep",
                })
                new_count += 1
            print(f"  [{search['search_name']}] {search['query'][:55]}: {new_count} new")
        except Exception as e:
            print(f"  ⚠️  Error '{search['query'][:40]}': {e}")

    print(f"\nTotal new articles from sweep: {len(all_articles)}")

    if not all_articles:
        print("  Nothing new to score")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    # Time filter
    cutoff = datetime.now(timezone.utc) - timedelta(days=past_days)
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)

    if df.empty:
        print("  No new articles within time window")
        return pd.DataFrame()

    print(f"After time filter: {len(df)} articles to score")

    # Translate to English for semantic scoring
    print("Translating for semantic scoring...")
    translator = GoogleTranslator(source="auto", target="en")
    titles_en  = []
    for title in df["title"].fillna("").astype(str):
        if not title.strip():
            titles_en.append("")
            continue
        try:
            lang = detect(title)
        except LangDetectException:
            lang = "en"
        if lang == "en":
            titles_en.append(title)
            continue
        try:
            result = translator.translate(title)
            titles_en.append(result if result else title)
        except Exception:
            titles_en.append(title)
    df["title_en"] = titles_en

    # Semantic relevance scoring
    print("Scoring semantic relevance...")
    texts = df["title_en"].fillna(df["title"]).fillna("").tolist()
    article_vectors = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    sim_matrix  = cosine_similarity(article_vectors, theme_vectors)
    max_scores  = sim_matrix.max(axis=1)
    best_themes = sim_matrix.argmax(axis=1)

    df["semantic_score"] = max_scores
    df["matched_theme"]  = [EVENT_TAXONOMY_THEMES[i] for i in best_themes]

    RELEVANCE_THRESHOLD = 0.30
    before = len(df)
    df = df[df["semantic_score"] >= RELEVANCE_THRESHOLD].reset_index(drop=True)
    print(f"Semantic filter (≥{RELEVANCE_THRESHOLD}): {before} → {len(df)} relevant articles")

    if not df.empty:
        print(f"\nTop matches by category:")
        for cat in ["Current_Affairs", "Supply_Chain", "Technology", "Fraud_Scam"]:
            cat_df = df[df["search_name"] == cat]
            if not cat_df.empty:
                top = cat_df.nlargest(3, "semantic_score")
                print(f"  {cat}:")
                for _, row in top.iterrows():
                    print(f"    [{row['semantic_score']:.2f}] {str(row.get('title_en', row['title']))[:75]}")

    # Remove internal scoring columns before merging
    df = df.drop(columns=["published_dt_utc", "semantic_score", "matched_theme", "title_en"],
                 errors="ignore")

    return df


# ---------------------------
# FEED.JSON EXPORTS
# ---------------------------

def _build_articles(df: pd.DataFrame, past_days: int) -> list:
    articles = []
    for _, row in df.iterrows():
        articles.append({
            "search_name":  str(row.get("search_name", "")),
            "search_query": str(row.get("search_query", "")),
            "title":        str(row.get("title", "")),
            "published":    str(row.get("published", "")),
            "link":         str(row.get("link", "")),
            "past_days":    int(row.get("past_days", past_days)),
            "source":       str(row.get("source", "google_rss")),
        })
    return articles


def export_daily_feed_json(df_today: pd.DataFrame, df_master: pd.DataFrame, past_days: int):
    articles    = _build_articles(df_master, past_days)
    payload     = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "lookback_days": past_days,
        "run_type":      "Daily run",
        "feed_type":     "daily",
        "articles":      articles,
    }
    output_path = DOCS_DIR / "daily_feed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"daily_feed.json written → {len(articles)} articles → {output_path}")


def export_weekly_feed_json(df: pd.DataFrame):
    articles    = _build_articles(df, 7)
    payload     = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "lookback_days": 7,
        "run_type":      "Weekly run",
        "feed_type":     "weekly",
        "articles":      articles,
    }
    output_path = DOCS_DIR / "weekly_feed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"weekly_feed.json written → {len(articles)} articles → {output_path}")


def export_feed_json(df: pd.DataFrame, past_days: int, run_mode: str):
    articles    = _build_articles(df, past_days)
    payload     = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "lookback_days": past_days,
        "run_type":      "Weekly run" if run_mode == "weekly" else "Daily run",
        "articles":      articles,
    }
    output_path = DOCS_DIR / "feed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"feed.json written → {len(articles)} articles → {output_path}")


# ---------------------------
# MAIN
# ---------------------------

def main():
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    print(f"Parsed search library: {len(search_df)} rows")
    print(f"Categories: {search_df['search_name'].unique()}")

    search_df = remap_legacy_unmapped(search_df)

    bad = search_df[search_df["search_name"] == "UNMAPPED_LINE"]
    if not bad.empty:
        print("UNMAPPED_LINE entries:")
        print(bad["raw_query"].tolist())

    search_df["google_news_compatible"] = search_df["raw_query"].apply(is_google_news_compatible)
    to_run = search_df[search_df["google_news_compatible"]].copy()

    print(f"\nRunning {len(to_run)} keyword searches across 4 categories")

    results = collect_google_news(to_run, past_days=PAST_DAYS, max_items=MAX_ITEMS)
    results = filter_last_n_days(results, n_days=PAST_DAYS)

    if not results.empty:
        results = results.drop_duplicates(subset=["link"]).reset_index(drop=True)

    raw_results_file  = DATA_DIR / f"google_news_raw_{ts}_past{PAST_DAYS}d.xlsx"
    audit_search_file = DATA_DIR / f"search_audit_{ts}.xlsx"

    if "published_dt_utc" in results.columns:
        results = results.drop(columns=["published_dt_utc"], errors="ignore")

    results.to_excel(raw_results_file, index=False, engine="openpyxl")
    search_df.to_excel(audit_search_file, index=False, engine="openpyxl")

    print(f"\nResults by category:")
    if not results.empty:
        for category, count in results["search_name"].value_counts().items():
            print(f"  {category}: {count} articles")
    else:
        print("  No results found")

    dedup_file  = DATA_DIR / f"google_news_dedup_{ts}_past{PAST_DAYS}d.xlsx"
    dedup_audit = DATA_DIR / f"google_news_dedup_audit_{ts}.xlsx"

    orig, cleaned = semantic_dedupe_excel(
        infile=str(raw_results_file),
        out_clean=str(dedup_file),
        out_audit=str(dedup_audit),
        threshold=DUP_THRESHOLD,
        model_name=MODEL_NAME,
    )

    df_final = pd.read_excel(dedup_file)

    master     = DATA_DIR / "topic_feeds.xlsx"
    daily_file = DAILY_DIR / "topic_feeds_daily.xlsx"

    update_master_excel_rolling(df_final, master, keep_days=7)
    df_final.to_excel(daily_file, index=False, engine="openpyxl")

    if RUN_MODE == "weekly":
        weekly_file   = WEEKLY_DIR / "topic_feeds_week.xlsx"
        weekly_latest = DATA_DIR / "topic_feeds_weekly_latest.xlsx"
        df_final.to_excel(weekly_file,   index=False, engine="openpyxl")
        df_final.to_excel(weekly_latest, index=False, engine="openpyxl")

    # ── SEMANTIC SWEEP ──
    # Finds relevant articles keyword searches missed.
    # Results already mapped to the 4 category names.
    existing_links = set(df_final["link"].dropna().tolist())

    df_semantic = semantic_sweep(
        existing_links=existing_links,
        past_days=PAST_DAYS,
        max_items=MAX_ITEMS,
        model_name=MODEL_NAME,
    )

    if not df_semantic.empty:
        df_final = pd.concat([df_final, df_semantic], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=["link"]).reset_index(drop=True)
        df_final.to_excel(daily_file, index=False, engine="openpyxl")
        update_master_excel_rolling(df_semantic, master, keep_days=7)
        print(f"\n✓ Semantic sweep added {len(df_semantic)} new articles")
        print(f"✓ Final feed total: {len(df_final)} articles")
        print(f"\nFinal breakdown by category:")
        for cat, count in df_final["search_name"].value_counts().items():
            print(f"  {cat}: {count}")
    else:
        print("\n  Semantic sweep: no additional articles found")

    # Export dashboard feeds
    df_master = pd.read_excel(master)

    daily_json_path = DOCS_DIR / "daily_feed.json"
    if daily_json_path.exists() and RUN_MODE == "daily":
        try:
            with open(daily_json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_articles = existing.get("articles", [])
            if existing_articles:
                df_existing  = pd.DataFrame(existing_articles)
                common_cols  = [c for c in df_final.columns if c in df_existing.columns]
                df_existing  = df_existing[common_cols]
                df_merged    = pd.concat([df_existing, df_final[common_cols]], ignore_index=True)
                df_merged    = df_merged.drop_duplicates(subset=["link"]).reset_index(drop=True)
                df_merged["published_dt_utc"] = df_merged["published"].apply(parse_published_dt)
                cutoff       = datetime.now(timezone.utc) - timedelta(days=7)
                df_merged    = df_merged[df_merged["published_dt_utc"].notna()]
                df_merged    = df_merged[df_merged["published_dt_utc"] >= cutoff]
                df_merged    = df_merged.drop(columns=["published_dt_utc"], errors="ignore").reset_index(drop=True)
                export_daily_feed_json(df_final, df_merged, PAST_DAYS)
            else:
                export_daily_feed_json(df_final, df_final, PAST_DAYS)
        except Exception as e:
            print(f"Daily feed merge warning: {e}")
            export_daily_feed_json(df_final, df_final, PAST_DAYS)
    else:
        export_daily_feed_json(df_final, df_final, PAST_DAYS)

    export_feed_json(df_final, PAST_DAYS, RUN_MODE)

    if RUN_MODE == "weekly":
        export_weekly_feed_json(df_master)

    print(f"\nRUN_MODE  = {RUN_MODE}")
    print(f"PAST_DAYS = {PAST_DAYS}")
    print(f"Master    : {master}")
    print(f"Daily     : {daily_file}")
    if RUN_MODE == "weekly":
        print(f"Weekly    : {weekly_file}")


if __name__ == "__main__":
    main()
