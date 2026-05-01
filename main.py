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

MAX_ITEMS     = int(os.getenv("MAX_ITEMS", "30"))
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.70"))
MODEL_NAME    = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

HL, GL, CEID = "en-GB", "GB", "GB:en"

DATA_DIR   = Path("data")
DAILY_DIR  = DATA_DIR / "daily"
WEEKLY_DIR = DATA_DIR / "weekly"
DOCS_DIR   = Path("docs")

for d in [DATA_DIR, DAILY_DIR, WEEKLY_DIR, DOCS_DIR]:
    d.mkdir(exist_ok=True)


# ---------------------------
# CANONICAL CATEGORY NAMES
# These are the final saved names used throughout the pipeline.
# A/B query pairs are collected under A/B names then normalised
# to the canonical name before deduplication and saving.
# ---------------------------

CANONICAL_CATEGORIES = [
    "PESTLE_Political",
    "PESTLE_Economic",
    "PESTLE_Social",
    "PESTLE_Technological",
    "PESTLE_Environmental",
    "PESTLE_Legal",
    "Supply_Chain",
    "Technology_Cyber",
    "Technology_AI",
    "Fraud_Financial",
    "Fraud_Insider",
]

# Maps A/B split query names back to their canonical category name
AB_NAME_MAP = {
    "PESTLE_Political_A":     "PESTLE_Political",
    "PESTLE_Political_B":     "PESTLE_Political",
    "PESTLE_Economic_A":      "PESTLE_Economic",
    "PESTLE_Economic_B":      "PESTLE_Economic",
    "PESTLE_Social_A":        "PESTLE_Social",
    "PESTLE_Social_B":        "PESTLE_Social",
    "PESTLE_Environmental_A": "PESTLE_Environmental",
    "PESTLE_Environmental_B": "PESTLE_Environmental",
    "PESTLE_Legal_A":         "PESTLE_Legal",
    "PESTLE_Legal_B":         "PESTLE_Legal",
    "Supply_Chain_A":         "Supply_Chain",
    "Supply_Chain_B":         "Supply_Chain",
    "Technology_AI_A":        "Technology_AI",
    "Technology_AI_B":        "Technology_AI",
    "Fraud_Insider_A":        "Fraud_Insider",
    "Fraud_Insider_B":        "Fraud_Insider",
}


def normalise_category(name: str) -> str:
    """Resolve A/B split names to their canonical category name."""
    return AB_NAME_MAP.get(name, name)


# ---------------------------
# SEARCH LIBRARY
# All queries are RSS-safe: under 700 encoded chars, max 25 OR terms,
# no broken - exclusion operators.
# A/B pairs are used where a topic needs more terms than one query allows.
# After collection, A/B results are merged under the canonical name.
# ---------------------------

SEARCH_LIBRARY_TEXT = r"""
PESTLE_Political_A   ("trade war" OR tariffs OR sanctions OR "political instability" OR "geopolitical risk" OR "diplomatic tension" OR "regime change" OR "political crisis" OR "coup" OR "civil war" OR "foreign policy" OR "NATO" OR "G7" OR "G20" OR "trade deal") AND (threat OR risk OR warning OR emerging OR escalation OR impact) AND (Europe OR UK OR "United States" OR global OR China OR Russia OR "Middle East")
PESTLE_Political_B   ("trade policy" OR "government policy" OR "energy politics" OR "food security" OR "water security" OR "soft power" OR "UN resolution" OR "bilateral agreement" OR "supply chain politics" OR "economic bloc" OR "election result") AND (threat OR risk OR opportunity OR impact OR warning OR shift OR agreement) AND (Europe OR UK OR "United States" OR global OR China OR Russia)
PESTLE_Economic_A   ("inflation" OR "interest rates" OR "recession" OR "economic slowdown" OR "GDP" OR "unemployment" OR "wage growth" OR "fiscal policy" OR "monetary policy" OR "central bank" OR "debt crisis" OR "currency fluctuation" OR "exchange rate") AND (threat OR risk OR warning OR forecast OR outlook OR decline OR crisis) AND (Europe OR UK OR "United States" OR global)
PESTLE_Economic_B   ("cost of living" OR "consumer spending" OR "energy prices" OR "oil prices" OR "commodity prices" OR "supply chain costs" OR "consumer confidence" OR "retail sales" OR "economic outlook" OR "market volatility") AND (threat OR risk OR opportunity OR impact OR warning OR forecast OR growth OR decline) AND (Europe OR UK OR "United States" OR global OR retail)
PESTLE_Social_A   ("demographic change" OR "ageing population" OR "migration" OR "urbanisation" OR "cost of living" OR "housing crisis" OR "social inequality" OR "workforce diversity" OR "ethical consumption" OR "sustainability demand") AND (Europe OR UK OR "United States" OR global OR retail OR consumer OR brand OR business)
PESTLE_Social_B   ("Gen Z" OR "millennial" OR "luxury demand" OR "experiential retail" OR "anti-consumerism" OR "protest movement" OR "social unrest" OR "trust in institutions" OR "polarisation" OR "mental health" OR "brand authenticity") AND (Europe OR UK OR "United States" OR global OR retail OR luxury OR consumer OR trend OR brand)
PESTLE_Technological   (semiconductor OR "chip shortage" OR quantum OR "5G" OR "6G" OR robotics OR automation OR drone OR satellite OR hydrogen OR battery OR "energy storage" OR biotech OR "autonomous vehicle") AND (threat OR ban OR shortage OR "supply chain" OR regulation OR "national security" OR disruption OR breakthrough) AND (Europe OR UK OR "United States" OR global OR government OR industry)
PESTLE_Environmental_A   ("climate change" OR "extreme weather" OR "flood risk" OR "drought" OR "wildfire" OR "heatwave" OR "carbon emissions" OR "net zero" OR "water scarcity" OR "energy transition" OR "renewable energy") AND (threat OR risk OR warning OR impact OR disruption OR emerging) AND (Europe OR UK OR "United States" OR global OR company OR industry)
PESTLE_Environmental_B   ("sustainability regulation" OR "ESG" OR "green policy" OR "carbon tax" OR "plastic ban" OR "deforestation" OR "biodiversity" OR "greenwashing" OR "climate litigation" OR "environmental protest" OR "scope 3 emissions") AND (threat OR risk OR opportunity OR warning OR regulation OR fine OR ban OR litigation) AND (Europe OR UK OR "United States" OR global OR retail OR company)
PESTLE_Legal_A   ("data privacy" OR "GDPR" OR "AI Act" OR "EU AI Act" OR "data protection" OR "competition law" OR "antitrust" OR "consumer protection" OR "product liability" OR "intellectual property" OR "trademark" OR "FCA" OR "ICO" OR "FTC") AND (warning OR fine OR ban OR ruling OR compliance OR enforcement OR lawsuit) AND (Europe OR UK OR "United States" OR global OR company OR brand)
PESTLE_Legal_B   ("employment law" OR "unfair dismissal" OR "sanctions compliance" OR "money laundering" OR "AML" OR "financial regulation" OR "tax law" OR "corporate governance" OR "whistleblower" OR "class action" OR "regulatory fine" OR "modern slavery" OR "supply chain due diligence") AND (threat OR risk OR warning OR fine OR ruling OR compliance OR enforcement OR lawsuit) AND (Europe OR UK OR "United States" OR global OR company OR employer)
Supply_Chain_A   ("supply chain disruption" OR "port strike" OR "transport strike" OR "freight disruption" OR "fuel protest" OR "farmers protest" OR "lorry driver" OR "trucker strike" OR haulage OR "border delay" OR "customs delay" OR "road closure" OR "Suez" OR "Dover" OR "Channel Tunnel") AND (disruption OR delay OR closed OR strike OR collapsed OR warning OR shortage OR Europe OR UK)
Supply_Chain_B   ("supply chain" OR logistics OR shipping OR freight OR tariffs OR reshoring OR nearshoring OR "import ban" OR "export ban" OR "raw material shortage" OR "component shortage") AND (disruption OR shortage OR delay OR collapsed OR strike OR blocked OR warning) AND (Europe OR UK OR global OR manufacturer OR supplier)
Technology_Cyber   (cybersecurity OR ransomware OR "data breach" OR malware OR "cyber attack" OR "zero day" OR vulnerability OR hacking OR phishing OR "cyber incident" OR "critical infrastructure" OR "cyber espionage" OR "nation state" OR "supply chain attack" OR "LLMjacking" OR "AI security") AND (attack OR breach OR warning OR arrested OR convicted OR threat OR incident OR advisory OR disclosed OR government OR company)
Technology_AI_A    ("ChatGPT" OR "Claude" OR "Gemini" OR "Copilot" OR "GPT-4" OR "GPT-5" OR "Grok" OR "Llama" OR "Mistral" OR "agentic AI" OR "AI agent" OR "AI assistant" OR "foundation model" OR "large language model" OR "generative AI") AND (launched OR released OR updated OR announced OR threat OR opportunity OR business OR enterprise OR productivity OR workforce OR jobs OR disruption) AND (Europe OR UK OR "United States" OR global OR company OR industry)
Technology_AI_B   ("AI adoption" OR "AI investment" OR "AI strategy" OR "AI transformation" OR "AI in business" OR "AI tools" OR "AI automation" OR "AI productivity" OR "AI workforce" OR "AI jobs" OR "AI opportunity" OR "AI risk" OR "AI misuse" OR "deepfake" OR "synthetic media") AND (company OR business OR enterprise OR organisation OR threat OR opportunity OR warning OR deployed OR implemented OR growing OR declining) AND (Europe OR UK OR "United States" OR global)
Fraud_Financial   (fraud OR scam OR "identity theft" OR "payment fraud" OR "money laundering" OR "bank fraud" OR "invoice fraud" OR "romance scam" OR "courier fraud" OR "card fraud" OR "mandate fraud" OR "APP fraud" OR "investment fraud" OR "crypto fraud" OR counterfeit OR embezzlement OR bribery OR corruption) AND (UK OR Britain OR Europe OR warning OR arrested OR convicted OR victim OR sentenced OR charged)
Fraud_Insider_A   ("employee theft" OR "staff fraud" OR "workplace fraud" OR "insider threat" OR "insider risk" OR "rogue employee" OR "disgruntled employee" OR sabotage OR "data theft" OR "stolen data" OR "leaked data") AND (UK OR Britain OR Europe OR company OR employer OR arrested OR convicted OR dismissed OR charged OR sentenced)
Fraud_Insider_B   ("intellectual property theft" OR "trade secret" OR "corporate espionage" OR "unauthorised access" OR "misuse of data" OR "internal fraud" OR "payroll fraud" OR "expense fraud" OR "HR fraud" OR "director fraud" OR "manager charged" OR "worker convicted") AND (UK OR Britain OR Europe OR company OR employer OR staff OR manager OR director OR arrested OR convicted OR charged)
""".strip()


# ---------------------------
# RELEVANCE THEMES
# Short phrases used to score whether a sweep article is relevant
# to its assigned category. One set of phrases per category.
# The semantic model compares article titles against these phrases
# and keeps only articles that score above the threshold.
# ---------------------------

RELEVANCE_THEMES = {
    "PESTLE_Political": [
        "trade war sanctions geopolitics political instability government policy",
        "diplomatic tension foreign policy NATO international relations conflict",
        "election regime change coup civil war political crisis",
        "energy politics food security water security bilateral agreement",
    ],
    "PESTLE_Economic": [
        "inflation interest rates recession consumer spending cost of living",
        "currency exchange rate GDP unemployment wage growth central bank",
        "energy prices commodity prices retail sales economic outlook",
        "market volatility fiscal policy debt crisis economic slowdown",
    ],
    "PESTLE_Social": [
        "consumer values ethical consumption sustainability luxury brand retail",
        "demographic change ageing population migration social inequality",
        "Gen Z millennial experiential retail anti-consumerism protest",
        "housing crisis mental health polarisation trust institutions",
    ],
    "PESTLE_Technological": [
        "semiconductor chip shortage quantum 5G robotics automation supply",
        "hydrogen battery energy storage carbon capture biotech autonomous vehicle",
        "satellite drone national security disruption breakthrough regulation",
    ],
    "PESTLE_Environmental": [
        "climate change extreme weather flood drought wildfire heatwave",
        "net zero sustainability ESG regulation carbon tax greenwashing",
        "energy transition renewable hydrogen climate litigation protest",
        "biodiversity deforestation water scarcity scope 3 emissions",
    ],
    "PESTLE_Legal": [
        "GDPR data privacy AI Act EU AI Act data protection regulation",
        "antitrust competition law regulatory fine FCA ICO FTC lawsuit",
        "employment law sanctions compliance money laundering AML financial regulation",
        "modern slavery supply chain due diligence corporate governance whistleblower",
    ],
    "Supply_Chain": [
        "supply chain disruption logistics shipping freight delay port strike",
        "transport blockade haulage road closure lorry driver trucker strike",
        "trade war tariffs import export ban sanctions raw material shortage",
        "border delay customs backlog reshoring nearshoring Suez Dover",
    ],
    "Technology_Cyber": [
        "cyber attack data breach ransomware malware hacking phishing zero day",
        "critical infrastructure cyber espionage nation state attack vulnerability",
        "LLMjacking AI security supply chain attack incident advisory",
    ],
    "Technology_AI": [
        "artificial intelligence generative AI large language model foundation model",
        "AI regulation AI Act governance safety risk misuse deepfake",
        "AI automation productivity adoption investment breakthrough workforce",
        "AI phishing AI-enabled fraud synthetic media machine learning",
    ],
    "Fraud_Financial": [
        "fraud scam financial crime money laundering payment fraud bank fraud",
        "investment scam crypto fraud romance scam card fraud APP fraud mandate",
        "bribery corruption embezzlement counterfeit false accounting convicted sentenced",
    ],
    "Fraud_Insider": [
        "employee theft staff fraud workplace misconduct insider threat rogue employee",
        "data theft stolen data leaked confidential sabotage disgruntled employee",
        "intellectual property theft trade secret corporate espionage unauthorised access",
        "payroll fraud expense fraud HR fraud director fraud worker convicted",
    ],
}

# Flatten into parallel lists for batch encoding
RELEVANCE_THEME_LIST = []
RELEVANCE_THEME_CATS = []
for cat, phrases in RELEVANCE_THEMES.items():
    for phrase in phrases:
        RELEVANCE_THEME_LIST.append(phrase)
        RELEVANCE_THEME_CATS.append(cat)


# ---------------------------
# SEMANTIC SWEEP SEARCHES
# Plain natural-language queries — no boolean syntax.
# Results scored against RELEVANCE_THEMES to filter off-topic articles.
# ---------------------------

SEMANTIC_SWEEP_SEARCHES = [
    # PESTLE Political
    {"search_name": "PESTLE_Political", "query": "trade war sanctions geopolitics political instability Europe UK"},
    {"search_name": "PESTLE_Political", "query": "government policy foreign policy diplomatic tension international"},
    {"search_name": "PESTLE_Political", "query": "conflict regime change political crisis election Europe"},
    {"search_name": "PESTLE_Political", "query": "energy politics food security water security global threat"},
    # PESTLE Economic
    {"search_name": "PESTLE_Economic", "query": "inflation interest rates recession consumer spending UK Europe"},
    {"search_name": "PESTLE_Economic", "query": "currency exchange rate economic outlook GDP unemployment"},
    {"search_name": "PESTLE_Economic", "query": "energy prices commodity prices retail sales market volatility"},
    {"search_name": "PESTLE_Economic", "query": "central bank fiscal policy debt crisis economic slowdown"},
    # PESTLE Social
    {"search_name": "PESTLE_Social", "query": "consumer values ethical consumption sustainability luxury brand"},
    {"search_name": "PESTLE_Social", "query": "demographic change ageing population migration social inequality"},
    {"search_name": "PESTLE_Social", "query": "Gen Z millennial retail trend anti-consumerism protest"},
    {"search_name": "PESTLE_Social", "query": "housing crisis mental health polarisation trust institutions"},
    # PESTLE Technological
    {"search_name": "PESTLE_Technological", "query": "semiconductor chip shortage quantum 5G robotics automation"},
    {"search_name": "PESTLE_Technological", "query": "hydrogen battery energy storage biotech autonomous vehicle"},
    {"search_name": "PESTLE_Technological", "query": "satellite drone national security disruption breakthrough"},
    # PESTLE Environmental
    {"search_name": "PESTLE_Environmental", "query": "climate change extreme weather flood drought wildfire Europe"},
    {"search_name": "PESTLE_Environmental", "query": "net zero sustainability ESG regulation carbon tax greenwashing"},
    {"search_name": "PESTLE_Environmental", "query": "energy transition renewable hydrogen climate litigation"},
    {"search_name": "PESTLE_Environmental", "query": "biodiversity deforestation water scarcity scope 3 emissions"},
    # PESTLE Legal
    {"search_name": "PESTLE_Legal", "query": "GDPR AI Act data protection regulation Europe UK fine"},
    {"search_name": "PESTLE_Legal", "query": "antitrust competition law regulatory fine FCA ICO lawsuit"},
    {"search_name": "PESTLE_Legal", "query": "sanctions compliance money laundering AML financial regulation"},
    {"search_name": "PESTLE_Legal", "query": "modern slavery supply chain due diligence corporate governance"},
    # Supply Chain
    {"search_name": "Supply_Chain", "query": "supply chain disruption logistics shipping freight delay"},
    {"search_name": "Supply_Chain", "query": "transport strike blockade road closure haulage disruption"},
    {"search_name": "Supply_Chain", "query": "trade war tariffs import export ban raw material shortage"},
    {"search_name": "Supply_Chain", "query": "port strike border delay customs backlog Suez Dover"},
    # Technology Cyber
    {"search_name": "Technology_Cyber", "query": "cyber attack data breach ransomware malware hacking"},
    {"search_name": "Technology_Cyber", "query": "cyber espionage nation state attack critical infrastructure"},
    {"search_name": "Technology_Cyber", "query": "zero day vulnerability phishing LLMjacking AI security"},
    # Technology AI
    {"search_name": "Technology_AI", "query": "artificial intelligence generative AI regulation risk businesses"},
    {"search_name": "Technology_AI", "query": "AI model launched regulation governance safety misuse deepfake"},
    {"search_name": "Technology_AI", "query": "AI automation productivity adoption workforce disruption"},
    {"search_name": "Technology_AI", "query": "AI phishing fraud synthetic media machine learning threat"},
    # Fraud Financial
    {"search_name": "Fraud_Financial", "query": "fraud scam financial crime money laundering UK Europe"},
    {"search_name": "Fraud_Financial", "query": "payment fraud bank fraud card fraud APP fraud mandate"},
    {"search_name": "Fraud_Financial", "query": "investment scam crypto fraud romance scam convicted sentenced"},
    # Fraud Insider
    {"search_name": "Fraud_Insider", "query": "employee arrested convicted fraud theft workplace insider"},
    {"search_name": "Fraud_Insider", "query": "insider threat staff fraud embezzlement bribery data stolen"},
    {"search_name": "Fraud_Insider", "query": "intellectual property theft trade secret corporate espionage rogue employee"},
]


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
    """Handle old category names from previous pipeline versions."""
    df = df.copy()
    remap = {
        "Current_Affairs":     "PESTLE_Political",
        "Technology":          "Technology_Cyber",
        "Tech":                "PESTLE_Technological",
        "Insider_Risk":        "Fraud_Insider",
        "Fraud_Scam":          "Fraud_Financial",
        "Technology_Emerging": "PESTLE_Technological",
    }
    df["search_name"] = df["search_name"].replace(remap)
    df["search_name"] = df["search_name"].apply(normalise_category)
    return df


# ---------------------------
# KEYWORD COLLECTION
# Parses SEARCH_LIBRARY_TEXT, runs each query as a Google News RSS fetch,
# and immediately normalises A/B names to their canonical category name.
# ---------------------------

def collect_google_news(df_searches: pd.DataFrame, past_days: int, max_items: int) -> pd.DataFrame:
    out_rows = []
    for _, r in df_searches.iterrows():
        raw_name = r["search_name"]
        q        = r["raw_query"]
        rss      = google_news_rss_url(q, past_days)
        feed     = feedparser.parse(rss)
        canon    = normalise_category(raw_name)
        print(f"  {raw_name} -> {canon}: {len(feed.entries)} raw entries")
        for entry in feed.entries[:max_items]:
            out_rows.append({
                "search_name":  canon,
                "search_query": q,
                "title":        entry.get("title", ""),
                "published":    entry.get("published", ""),
                "link":         entry.get("link", ""),
                "past_days":    past_days,
                "source":       "google_rss",
            })

    if out_rows:
        temp_df = pd.DataFrame(out_rows)
        print("\nRaw collection summary (canonical categories):")
        for name in sorted(temp_df["search_name"].unique()):
            count = len(temp_df[temp_df["search_name"] == name])
            print(f"  {name}: {count} articles")

    return pd.DataFrame(out_rows)


# ---------------------------
# SEMANTIC DEDUPE — INTRA-TOPIC ONLY
#
# KEY BEHAVIOUR:
# - Deduplication runs independently per canonical category
# - An article appearing in both PESTLE_Political AND Supply_Chain
#   is kept in BOTH — cross-topic duplicates are intentional and preserved
# - Only near-identical articles within the SAME category are removed
# - The duplicate check key is (link, search_name) not just link
# ---------------------------

def semantic_dedupe_within_topic(df: pd.DataFrame, threshold: float, model_name: str) -> tuple:
    if df.empty:
        return df.copy(), pd.DataFrame()

    df["title"] = df["title"].fillna("").astype(str)
    model      = SentenceTransformer(model_name)
    keep_rows  = []
    audit_rows = []

    for topic in sorted(df["search_name"].unique()):
        topic_df = df[df["search_name"] == topic].copy().reset_index(drop=True)

        if topic_df.empty:
            continue

        valid_mask = topic_df["title"].str.strip().str.len() > 0
        work_df    = topic_df[valid_mask].reset_index(drop=True)
        empty_df   = topic_df[~valid_mask].reset_index(drop=True)

        if work_df.empty:
            keep_rows.append(topic_df)
            continue

        emb = model.encode(
            work_df["title"].tolist(),
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
            groups.setdefault(find(i), []).append(i)

        kept_indices = []
        for members in groups.values():
            keep_i = min(members)
            kept_indices.append(keep_i)
            for drop_i in members:
                if drop_i == keep_i:
                    continue
                audit_rows.append({
                    "search_name":   topic,
                    "kept_index":    keep_i,
                    "dropped_index": drop_i,
                    "similarity":    float(sim[keep_i, drop_i]),
                    "kept_title":    work_df.loc[keep_i, "title"],
                    "dropped_title": work_df.loc[drop_i, "title"],
                })

        kept_df = work_df.loc[sorted(kept_indices)].reset_index(drop=True)
        if not empty_df.empty:
            kept_df = pd.concat([kept_df, empty_df], ignore_index=True)

        keep_rows.append(kept_df)
        removed = len(topic_df) - len(kept_df)
        print(f"  [{topic}] {len(topic_df)} -> {len(kept_df)} (removed {removed} intra-topic duplicates)")

    df_clean = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame()
    df_audit = pd.DataFrame(audit_rows)
    return df_clean, df_audit


def semantic_dedupe_excel(infile: str, out_clean: str, out_audit: str,
                          threshold: float, model_name: str) -> tuple:
    df = pd.read_excel(infile)

    if df.empty:
        df.to_excel(out_clean, index=False, engine="openpyxl")
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return 0, 0

    orig_count          = len(df)
    df_clean, df_audit  = semantic_dedupe_within_topic(df, threshold, model_name)

    df_clean.to_excel(out_clean, index=False, engine="openpyxl")
    df_audit.to_excel(out_audit, index=False, engine="openpyxl")
    return orig_count, len(df_clean)


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
    # Drop exact same article in exact same category — keep cross-topic copies
    combined = combined.drop_duplicates(subset=["link", "search_name"]).reset_index(drop=True)
    combined["published_dt_utc"] = combined["published"].apply(parse_published_dt)
    cutoff   = datetime.now(timezone.utc) - timedelta(days=keep_days)
    combined = combined[combined["published_dt_utc"].notna()]
    combined = combined[combined["published_dt_utc"] >= cutoff].copy()
    combined = combined.drop(columns=["published_dt_utc"], errors="ignore")
    combined.to_excel(master_path, index=False, engine="openpyxl")


# ---------------------------
# SEMANTIC SWEEP
# Runs broad natural-language searches to catch articles that the
# boolean keyword queries may have missed. Each article is scored
# against the RELEVANCE_THEMES for its assigned category.
# Articles scoring below 0.30 are dropped as off-topic.
# ---------------------------

def semantic_sweep(existing_links: set, past_days: int,
                   max_items: int, model_name: str) -> pd.DataFrame:

    print(f"\n{'='*60}")
    print(f"SEMANTIC SWEEP — broad searches + relevance scoring")
    print(f"  Searches: {len(SEMANTIC_SWEEP_SEARCHES)}  |  Relevance threshold: 0.30")
    print(f"{'='*60}\n")

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    theme_vectors = model.encode(
        RELEVANCE_THEME_LIST,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"Relevance themes loaded: {len(RELEVANCE_THEME_LIST)} phrases across {len(RELEVANCE_THEMES)} categories\n")

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
                    "search_name":  search["search_name"],
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
            print(f"  Warning '{search['query'][:40]}': {e}")

    print(f"\nTotal new articles from sweep: {len(all_articles)}")

    if not all_articles:
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    cutoff = datetime.now(timezone.utc) - timedelta(days=past_days)
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)

    if df.empty:
        print("  No new articles within time window")
        return pd.DataFrame()

    print(f"After time filter: {len(df)} articles to score")

    print("Translating non-English titles...")
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

    print("Scoring relevance against category themes...")
    texts           = df["title_en"].fillna(df["title"]).fillna("").tolist()
    article_vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    sim_matrix      = cosine_similarity(article_vectors, theme_vectors)

    scores       = []
    matched_cats = []
    for i, row in df.iterrows():
        assigned_cat = row["search_name"]
        # Score against themes for the article's assigned category only
        cat_indices = [j for j, c in enumerate(RELEVANCE_THEME_CATS) if c == assigned_cat]
        if cat_indices:
            cat_scores  = sim_matrix[i, cat_indices]
            best_score  = float(cat_scores.max())
            best_theme  = RELEVANCE_THEME_LIST[cat_indices[int(cat_scores.argmax())]]
        else:
            best_score = float(sim_matrix[i].max())
            best_theme = RELEVANCE_THEME_LIST[int(sim_matrix[i].argmax())]
        scores.append(best_score)
        matched_cats.append(best_theme)

    df["relevance_score"] = scores
    df["matched_theme"]   = matched_cats

    RELEVANCE_THRESHOLD = 0.30
    before = len(df)
    df = df[df["relevance_score"] >= RELEVANCE_THRESHOLD].reset_index(drop=True)
    print(f"Relevance filter (>={RELEVANCE_THRESHOLD}): {before} -> {len(df)} kept")

    if not df.empty:
        print(f"\nTop matches by category:")
        for cat in CANONICAL_CATEGORIES:
            cat_df = df[df["search_name"] == cat]
            if not cat_df.empty:
                top = cat_df.nlargest(2, "relevance_score")
                print(f"  {cat}:")
                for _, row in top.iterrows():
                    print(f"    [{row['relevance_score']:.2f}] {str(row.get('title_en', row['title']))[:70]}")

    df = df.drop(columns=["published_dt_utc", "relevance_score", "matched_theme", "title_en"],
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
    print(f"daily_feed.json written -> {len(articles)} articles -> {output_path}")


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
    print(f"weekly_feed.json written -> {len(articles)} articles -> {output_path}")


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
    print(f"feed.json written -> {len(articles)} articles -> {output_path}")


# ---------------------------
# MAIN
# ---------------------------

def main():
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    # Parse search library — A/B names normalised to canonical inside collect_google_news
    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    print(f"Parsed search library: {len(search_df)} rows")

    search_df = remap_legacy_unmapped(search_df)
    print(f"Categories: {sorted(search_df['search_name'].unique())}")

    bad = search_df[search_df["search_name"] == "UNMAPPED_LINE"]
    if not bad.empty:
        print("UNMAPPED_LINE entries:")
        print(bad["raw_query"].tolist())

    search_df["google_news_compatible"] = search_df["raw_query"].apply(is_google_news_compatible)
    to_run = search_df[search_df["google_news_compatible"]].copy()

    print(f"\nRunning {len(to_run)} RSS queries across {len(CANONICAL_CATEGORIES)} canonical categories")

    results = collect_google_news(to_run, past_days=PAST_DAYS, max_items=MAX_ITEMS)
    results = filter_last_n_days(results, n_days=PAST_DAYS)

    if not results.empty:
        # Drop exact same URL in exact same category — keep cross-topic copies
        results = results.drop_duplicates(subset=["link", "search_name"]).reset_index(drop=True)

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

    print(f"\nDedup: {orig} -> {cleaned} articles (intra-topic only, cross-topic preserved)")

    df_final   = pd.read_excel(dedup_file)
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
    existing_links = set(df_final["link"].dropna().tolist())

    df_semantic = semantic_sweep(
        existing_links=existing_links,
        past_days=PAST_DAYS,
        max_items=MAX_ITEMS,
        model_name=MODEL_NAME,
    )

    if not df_semantic.empty:
        df_final = pd.concat([df_final, df_semantic], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=["link", "search_name"]).reset_index(drop=True)
        df_final.to_excel(daily_file, index=False, engine="openpyxl")
        update_master_excel_rolling(df_semantic, master, keep_days=7)
        print(f"\nSemantic sweep added {len(df_semantic)} new articles")
        print(f"Final feed total: {len(df_final)} articles")
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
                df_merged    = df_merged.drop_duplicates(subset=["link", "search_name"]).reset_index(drop=True)
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
