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
# EVENT TAXONOMY THEMES
# 6 PESTLE categories + Supply Chain + Technology + Fraud
# ---------------------------

EVENT_TAXONOMY_THEMES = [
    # PESTLE_Political
    "government policy trade war sanctions geopolitical risk political instability",
    "diplomatic tension international relations trade deal bilateral agreement",
    "regime change political crisis coup civil war conflict",
    "election result political shift foreign policy NATO G7 G20",
    "soft power energy politics food security water security",
    # PESTLE_Economic
    "inflation interest rates consumer spending cost of living recession",
    "currency fluctuation exchange rate economic slowdown GDP unemployment",
    "energy prices commodity prices supply chain costs import costs",
    "consumer confidence retail sales economic outlook fiscal policy",
    "central bank debt crisis market volatility stock market bond market",
    # PESTLE_Social
    "demographic change ageing population migration urbanisation lifestyle trend",
    "consumer values brand authenticity ethical consumption sustainability demand",
    "social inequality housing crisis mental health wellbeing workforce diversity",
    "trust institutions polarisation culture war Gen Z millennial luxury demand",
    "anti-consumerism experiential retail social media influencer cancel culture",
    # PESTLE_Technological
    "cybersecurity data breach ransomware hack malware attack zero day",
    "agentic AI autonomous systems artificial intelligence regulation disruption",
    "semiconductor chip shortage quantum computing technology supply",
    "surveillance facial recognition biometrics deepfake privacy",
    "R&D breakthrough robotics automation IoT digital transformation",
    # PESTLE_Environmental
    "climate change extreme weather flood drought wildfire heatwave",
    "carbon emissions net zero sustainability regulation ESG green policy",
    "carbon tax plastic ban deforestation biodiversity water scarcity",
    "energy transition renewable energy hydrogen carbon capture greenwashing",
    "climate litigation environmental protest supply chain sustainability scope 3",
    # PESTLE_Legal
    "employment law unfair dismissal data privacy GDPR AI regulation",
    "EU AI Act data protection cyber law digital regulation competition law",
    "antitrust consumer protection product liability intellectual property trademark",
    "sanctions compliance money laundering AML KYC financial regulation",
    "regulatory fine class action litigation modern slavery supply chain due diligence",
    # Supply Chain
    "supply chain disruption logistics shipping delay freight port strike",
    "transport blockade haulage road closure lorry driver trucker strike",
    "trade war tariffs import export ban sanctions embargo",
    "raw material shortage component shortage factory closure border delay",
    "freight disruption delivery delay customs backlog reshoring nearshoring",
    # Technology Cyber
    "cyber attack infrastructure critical systems outage nation state attack",
    "phishing zero day vulnerability software exploit supply chain attack",
    "LLMjacking AI threat cyber fraud cyber espionage",
    # Technology Emerging
    "chip shortage semiconductor supply quantum 5G 6G robotics automation",
    "hydrogen battery energy storage carbon capture biotech genomics",
    "satellite space tech autonomous vehicle digital twin smart city",
    # Fraud and Insider Risk
    "fraud scam financial crime money laundering payment fraud bank fraud",
    "investment scam crypto fraud Ponzi scheme romance scam card fraud",
    "employee theft staff fraud workplace misconduct insider threat embezzlement",
    "data theft stolen data leaked confidential information sabotage bribery",
    "convicted sentenced pleaded guilty employee fraud corruption trade secret",
]


# ---------------------------
# SEMANTIC SWEEP SEARCHES
# ---------------------------

SEMANTIC_SWEEP_SEARCHES = [
    # PESTLE Political
    {"search_name": "PESTLE_Political", "query": "trade war sanctions geopolitics political instability Europe UK"},
    {"search_name": "PESTLE_Political", "query": "government policy foreign policy diplomatic tension international"},
    {"search_name": "PESTLE_Political", "query": "conflict regime change political crisis election Europe"},
    {"search_name": "PESTLE_Political", "query": "energy politics food security water security global threat"},
    # PESTLE Economic
    {"search_name": "PESTLE_Economic", "query": "inflation interest rates recession consumer spending UK Europe"},
    {"search_name": "PESTLE_Economic", "query": "currency fluctuation exchange rate economic outlook GDP"},
    {"search_name": "PESTLE_Economic", "query": "energy prices commodity prices retail sales market volatility"},
    {"search_name": "PESTLE_Economic", "query": "central bank fiscal policy debt crisis economic slowdown"},
    # PESTLE Social
    {"search_name": "PESTLE_Social", "query": "consumer values ethical consumption sustainability luxury brand"},
    {"search_name": "PESTLE_Social", "query": "demographic change ageing population migration social inequality"},
    {"search_name": "PESTLE_Social", "query": "brand authenticity cancel culture Gen Z millennial retail trend"},
    {"search_name": "PESTLE_Social", "query": "housing crisis mental health polarisation trust institutions"},
    # PESTLE Technological
    {"search_name": "PESTLE_Technological", "query": "cybersecurity data breach ransomware hack malware attack"},
    {"search_name": "PESTLE_Technological", "query": "agentic AI autonomous systems technology regulation disruption"},
    {"search_name": "PESTLE_Technological", "query": "semiconductor chip shortage quantum computing supply"},
    {"search_name": "PESTLE_Technological", "query": "surveillance biometrics deepfake AI threat privacy"},
    # PESTLE Environmental
    {"search_name": "PESTLE_Environmental", "query": "climate change extreme weather flood drought wildfire Europe"},
    {"search_name": "PESTLE_Environmental", "query": "net zero sustainability ESG regulation carbon tax greenwashing"},
    {"search_name": "PESTLE_Environmental", "query": "energy transition renewable hydrogen carbon capture"},
    {"search_name": "PESTLE_Environmental", "query": "climate litigation environmental protest supply chain sustainability"},
    # PESTLE Legal
    {"search_name": "PESTLE_Legal", "query": "employment law GDPR AI regulation data protection Europe UK"},
    {"search_name": "PESTLE_Legal", "query": "EU AI Act antitrust competition law regulatory fine lawsuit"},
    {"search_name": "PESTLE_Legal", "query": "sanctions compliance money laundering AML financial regulation"},
    {"search_name": "PESTLE_Legal", "query": "modern slavery supply chain due diligence intellectual property"},
    # Supply Chain
    {"search_name": "Supply_Chain", "query": "supply chain disruption logistics shipping freight delay"},
    {"search_name": "Supply_Chain", "query": "transport strike blockade road closure haulage disruption"},
    {"search_name": "Supply_Chain", "query": "trade war tariffs import export ban sanctions"},
    {"search_name": "Supply_Chain", "query": "port strike shipping delay raw material shortage factory closure"},
    {"search_name": "Supply_Chain", "query": "border delay customs backlog supply disruption reshoring"},
    # Technology Cyber
    {"search_name": "Technology_Cyber", "query": "cybersecurity data breach ransomware hack malware"},
    {"search_name": "Technology_Cyber", "query": "cyber attack phishing zero day vulnerability exploit"},
    {"search_name": "Technology_Cyber", "query": "nation state attack critical infrastructure cyber espionage"},
    {"search_name": "Technology_Cyber", "query": "LLMjacking AI security threat supply chain attack"},
    # Technology Emerging
    {"search_name": "Technology_Emerging", "query": "semiconductor chip shortage quantum 5G robotics automation"},
    {"search_name": "Technology_Emerging", "query": "hydrogen battery energy storage carbon capture biotech"},
    {"search_name": "Technology_Emerging", "query": "satellite space tech autonomous vehicle digital twin"},
    # Fraud
    {"search_name": "Fraud_Scam", "query": "fraud scam financial crime money laundering UK Europe"},
    {"search_name": "Fraud_Scam", "query": "payment fraud bank fraud card fraud APP fraud mandate"},
    {"search_name": "Fraud_Scam", "query": "investment scam crypto fraud Ponzi scheme romance scam"},
    {"search_name": "Fraud_Scam", "query": "employee arrested convicted fraud theft data stolen workplace"},
    {"search_name": "Fraud_Scam", "query": "insider threat staff fraud embezzlement bribery corruption"},
]


# ---------------------------
# SEARCH LIBRARY
# Categories: PESTLE x6 + Supply_Chain + Technology_Cyber + Technology_Emerging + Fraud_Scam
# ---------------------------

SEARCH_LIBRARY_TEXT = r"""
PESTLE_Political	("trade policy" OR "trade war" OR tariffs OR sanctions OR "government policy" OR "political instability" OR "geopolitical risk" OR "soft power" OR "diplomatic tension" OR "regime change" OR "political crisis" OR "election result" OR "coup" OR "civil war" OR "international relations" OR "foreign policy" OR "NATO" OR "UN resolution" OR "G7" OR "G20" OR "bilateral agreement" OR "trade deal" OR "economic bloc" OR "supply chain politics" OR "energy politics" OR "food security" OR "water security") AND (threat OR risk OR opportunity OR impact OR warning OR emerging OR shift OR change OR tension OR escalation OR agreement) AND (Europe OR UK OR "United States" OR global OR international OR China OR Russia OR "Middle East")
PESTLE_Economic	("inflation" OR "interest rates" OR "consumer spending" OR "cost of living" OR "currency fluctuation" OR "exchange rate" OR "recession" OR "economic slowdown" OR "GDP" OR "unemployment" OR "labour market" OR "wage growth" OR "energy prices" OR "oil prices" OR "commodity prices" OR "supply chain costs" OR "import costs" OR "export decline" OR "consumer confidence" OR "retail sales" OR "economic outlook" OR "fiscal policy" OR "monetary policy" OR "central bank" OR "debt crisis" OR "market volatility" OR "stock market" OR "bond market") AND (threat OR risk OR opportunity OR impact OR warning OR emerging OR forecast OR outlook OR decline OR growth OR crisis) AND (Europe OR UK OR "United States" OR global OR international)
PESTLE_Social	("demographic change" OR "ageing population" OR "birth rate" OR "migration" OR "urbanisation" OR "lifestyle trend" OR "consumer values" OR "brand authenticity" OR "ethical consumption" OR "sustainability demand" OR "social media" OR "influencer" OR "mental health" OR "wellbeing" OR "workforce diversity" OR "gender equality" OR "cost of living" OR "housing crisis" OR "social inequality" OR "class divide" OR "trust in institutions" OR "polarisation" OR "culture war" OR "cancel culture" OR "Gen Z" OR "millennial" OR "luxury demand" OR "experiential retail" OR "anti-consumerism") AND (threat OR risk OR opportunity OR trend OR shift OR emerging OR change OR impact OR warning OR growing OR declining) AND (Europe OR UK OR "United States" OR global OR retail OR luxury OR brand OR consumer)
PESTLE_Technological	(cybersecurity OR ransomware OR "data breach" OR malware OR "cyber attack" OR "zero day" OR vulnerability OR hacking OR phishing OR "agentic AI" OR "autonomous systems" OR "AI regulation" OR "AI risk" OR "LLMjacking" OR "AI security" OR semiconductor OR "chip shortage" OR quantum OR "5G" OR robotics OR automation OR deepfake OR surveillance OR biometrics OR "facial recognition" OR "R&D breakthrough" OR "tech regulation" OR "digital transformation") AND (threat OR risk OR opportunity OR warning OR breach OR attack OR disruption OR regulation OR emerging OR breakthrough OR ban OR sanction) AND (Europe OR UK OR "United States" OR global OR government OR company OR industry) AND -(rumor OR gaming OR podcast OR "live blog" OR "product review" OR review)
PESTLE_Environmental	("climate change" OR "global warming" OR "extreme weather" OR "flood risk" OR "drought" OR "wildfire" OR "storm" OR "heatwave" OR "carbon emissions" OR "net zero" OR "sustainability regulation" OR "ESG" OR "green policy" OR "carbon tax" OR "carbon border" OR "plastic ban" OR "deforestation" OR "biodiversity" OR "water scarcity" OR "energy transition" OR "renewable energy" OR "hydrogen" OR "carbon capture" OR "supply chain sustainability" OR "scope 3 emissions" OR "greenwashing" OR "climate litigation" OR "environmental protest") AND (threat OR risk OR opportunity OR warning OR regulation OR fine OR ban OR emerging OR impact OR disruption OR litigation) AND (Europe OR UK OR "United States" OR global OR retail OR luxury OR brand OR company)
PESTLE_Legal	("employment law" OR "unfair dismissal" OR "data privacy" OR "GDPR" OR "AI regulation" OR "AI Act" OR "EU AI Act" OR "data protection" OR "cyber law" OR "digital regulation" OR "competition law" OR "antitrust" OR "consumer protection" OR "product liability" OR "intellectual property" OR "trademark" OR "copyright" OR "sanctions compliance" OR "money laundering regulation" OR "AML" OR "KYC" OR "financial regulation" OR "tax law" OR "corporate governance" OR "whistleblower" OR "class action" OR "litigation" OR "regulatory fine" OR "ICO" OR "FCA" OR "FTC" OR "right to repair" OR "modern slavery" OR "supply chain due diligence") AND (threat OR risk OR opportunity OR warning OR fine OR ban OR ruling OR emerging OR change OR compliance OR enforcement OR lawsuit OR regulation) AND (Europe OR UK OR "United States" OR global OR company OR employer OR brand OR retail)
Supply_Chain	("supply chain" OR logistics OR shipping OR freight OR port OR tariffs OR sanctions OR embargoes OR reshoring OR nearshoring OR "supply chain disruption" OR fragmentation OR instability OR "road closure" OR "port strike" OR "transport strike" OR "freight disruption" OR "fuel protest" OR "farmers protest" OR "lorry driver" OR "trucker strike" OR haulage OR "border delay" OR "customs delay" OR "import ban" OR "export ban" OR "trade war" OR "trade disruption" OR "raw material shortage" OR "component shortage") AND (company OR manufacturer OR factory OR supplier OR export OR import OR production OR delivery OR route OR network OR Europe OR UK OR global)
Technology_Cyber	(cybersecurity OR ransomware OR "data breach" OR malware OR "cyber attack" OR "zero day" OR vulnerability OR hacking OR phishing OR "cyber incident" OR "critical infrastructure" OR "cyber espionage" OR "nation state attack" OR "supply chain attack" OR deepfake OR surveillance OR biometrics OR "facial recognition" OR "cyber fraud" OR "LLMjacking" OR "AI threat" OR "AI security") AND (attack OR breach OR warning OR arrested OR convicted OR victim OR threat OR vulnerability OR incident OR regulation OR government OR company) AND -(rumor OR gaming OR podcast OR "live blog" OR review)
Technology_Emerging	(semiconductor OR chip OR "chip shortage" OR quantum OR "5G" OR "6G" OR robotics OR automation OR drone OR satellite OR "space tech" OR hydrogen OR battery OR "energy storage" OR "carbon capture" OR biotech OR genomics OR nanotechnology OR "digital twin" OR blockchain OR "smart city" OR "autonomous vehicle" OR "edge computing") AND (attack OR breach OR warning OR threat OR ban OR sanction OR shortage OR "supply chain" OR regulation OR legislation OR "national security" OR espionage OR vulnerability OR incident OR disruption) AND -(AI OR "artificial intelligence" OR "machine learning") AND -(investment OR funding OR IPO OR shares OR stock OR shareholder OR "annual meeting" OR "quarterly results" OR "private placement" OR review OR podcast OR gaming)
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
    # Remap old category names to new ones
    remap = {
        "Current_Affairs":    "PESTLE_Political",
        "Technology":         "Technology_Cyber",
        "Tech":               "Technology_Emerging",
        "Insider_Risk":       "Fraud_Scam",
    }
    df["search_name"] = df["search_name"].replace(remap)
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
# ---------------------------

def semantic_sweep(existing_links: set, past_days: int,
                   max_items: int, model_name: str) -> pd.DataFrame:

    all_categories = [
        "PESTLE_Political", "PESTLE_Economic", "PESTLE_Social",
        "PESTLE_Technological", "PESTLE_Environmental", "PESTLE_Legal",
        "Supply_Chain", "Technology_Cyber", "Technology_Emerging", "Fraud_Scam",
    ]

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
                    "search_name":  search["search_name"],
                    "search_query": search["query"],
                    "title":        entry.get("title", ""),
                    "published":    entry.get("published", ""),
                    "link":         link,
                    "past_days":    past_days,
                    "source":       "semantic_sweep",
                })
                new_count += 1
            print(f"  [{search['search_name']}] {search['query'][:50]}: {new_count} new")
        except Exception as e:
            print(f"  ⚠️  Error '{search['query'][:40]}': {e}")

    print(f"\nTotal new articles from sweep: {len(all_articles)}")

    if not all_articles:
        print("  Nothing new to score")
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

    print("Scoring semantic relevance...")
    texts = df["title_en"].fillna(df["title"]).fillna("").tolist()
    article_vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
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
        for cat in all_categories:
            cat_df = df[df["search_name"] == cat]
            if not cat_df.empty:
                top = cat_df.nlargest(2, "semantic_score")
                print(f"  {cat}:")
                for _, row in top.iterrows():
                    print(f"    [{row['semantic_score']:.2f}] {str(row.get('title_en', row['title']))[:70]}")

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

    search_df = remap_legacy_unmapped(search_df)
    print(f"Categories: {sorted(search_df['search_name'].unique())}")

    bad = search_df[search_df["search_name"] == "UNMAPPED_LINE"]
    if not bad.empty:
        print("UNMAPPED_LINE entries:")
        print(bad["raw_query"].tolist())

    search_df["google_news_compatible"] = search_df["raw_query"].apply(is_google_news_compatible)
    to_run = search_df[search_df["google_news_compatible"]].copy()

    print(f"\nRunning {len(to_run)} keyword searches across 10 categories")

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
