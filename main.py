# %%
pip install pandas numpy feedparser requests trafilatura rapidfuzz sentence-transformers scikit-learn


# %%
pip install python-dateutil


# %%
import re
import urllib.parse
from datetime import datetime, timezone

import feedparser
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
PAST_DAYS = 1         # change to 1 / 3 / 14 etc.
MAX_ITEMS = 50          # per search
HL, GL, CEID = "en-GB", "GB", "GB:en"

from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

def parse_published_dt(published_str: str):
    if not published_str:
        return None
    try:
        dt = dateparser.parse(published_str)
        if dt is None:
            return None
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def filter_last_n_days(df, n_days: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
    df = df.copy()
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    # keep rows with a valid date AND within window
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)
    return df


# ---------------------------
# ALL SEARCHES (from your pasted library)  :contentReference[oaicite:1]{index=1}
# Paste your full list here exactly as-is if you update it later.
# ---------------------------
SEARCH_LIBRARY_TEXT = r"""
Brand search Google News and X - PROTEST	("Gucci" OR "Ralph Lauren" OR "Dior" OR "Armani" OR "louis Vuitton" OR "Valentino" OR "Burberry" OR "dolce & gabbana" OR "Polo Ralph Lauren" OR "Cartier" OR "Calvin Klein"  OR "Hilfiger" OR "Tommy H" OR "Timberland" OR "Prada" OR "Nike" OR "Balenciaga" OR "Canada Goose" ) AND ("protest")
Brand search Google News and X (can't have 2 "And" terms in twitter searches) - BOYCOTT	("Gucci" OR "Ralph Lauren" OR "Dior" OR "Armani" OR "louis Vuitton" OR "Valentino" OR "Burberry" OR "dolce & gabbana" OR "Polo Ralph Lauren" OR "Cartier" OR "Calvin Klein"  OR "Hilfiger" OR "Tommy H" OR "Timberland" OR "Prada" OR "Nike" OR "Balenciaga" OR "Canada Goose" ) AND ("boycott")
Cotswold Designer Outlet	("Cotswold Designer Outlet") 
Value Retail	("valueretail" OR "Value retail" OR "Bicester Collection")
London Marylebone	("marylebone station" OR "london Marylebone") AND ("incident" OR "delay" OR "closed" OR "delays" OR "protest" OR "boycott" OR "bomb" OR "Bomb threat" OR "shooting" OR "explosion" OR "stabbing"  OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat")
Roermond outlet	("Roermond Outlet" AND (incident OR protest OR closure OR police))
Roermond outlet conversations	("@RoermondOutlet" OR "to:roermondoutlet" OR "Roermond outlet")
Searches for any tweets directed at BV Accounts	(to:bicestervillage OR to:KildareVillage OR to:Vallee_Village OR to:MMVillage OR to:WertheimVillage OR to:IngolstadtVillage OR to:FidenzaVillage OR to:LasRozasVillage OR to:LaRocaVillage OR to:bicester_coll)
Searches for any tweets mentioning BV Accounts	(@bicestervillage OR @KildareVillage OR @Vallee_Village OR @MMVillage OR @WertheimVillage OR @IngolstadtVillage OR @FidenzaVillage OR @LasRozasVillage OR @LaRocaVillage OR @bicester_coll)
BV Logistics Companies	("Dropit" OR "DHL" OR "UPS" OR "DPD" OR "Flight Logistics" OR "Fedex" OR "Amazon" OR "Royal Mail" OR "Parcelforce" OR "EVRI" ) AND ( "supply " OR  "delivery")
High Level City searches   Belgium Dutch language	("brussels" OR "Maastricht" OR "Antwerp") AND ("Bom" OR "bommelding" OR "onbeheerd pakket" OR "verdacht pakket" OR "explosieven" OR "explosie" OR "explosief" OR "schieten" OR "schietincident" OR "steken" OR "bommetje" OR "bomba" OR "Geknalt" OR "liquidatie")
High Level City searches   Belgium English language	("brussels" OR "Maastricht" OR "Antwerp") AND ("Bomb" OR "explosion" OR "shooting"  OR "Stabbing" OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat" OR "Retail crime")
High Level City searches Belgium French language	("brussels" OR "Maastricht" OR "Antwerp" ) AND ("Bombe" OR "alerte à la bombe" OR "explosif" OR "colis sans surveillance" OR  "explosion" OR "colis suspect" OR "tournage" OR "poignarder")
Local Town Searches  Maasmechelen Dutch Terms	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND ("Boycot" OR "Boycotten" OR "Protest" OR "Bom" OR "bommelding" OR "onbeheerd pakket" OR "verdacht pakket" OR "explosieven" OR "explosie" OR "explosief" OR "schieten" OR "schietincident" OR "steken" OR "bommetje" OR "bomba" OR "Geknalt" OR "liquidatie" ) -geoffroy
Local Town Searches Maasmechelen French Terms	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND ("manifestation" OR "protestation" OR "boycotter" OR "Bombe" OR "explosif" OR "colis sans surveillance"  OR "alerte à la bombe"   OR "colis suspect" OR "explosion" OR "tournage" OR "poignarder") -geoffroy
Local Town Searches Maasmechelen English	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND ("protest" OR "boycott" OR "bomb"  OR "shooting" OR "explosion" OR "stabbing"  OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat" ) -geoffroy
Village searches Maasmechelen	 (  "Maasmechelen Village" OR "#MaasmechelenVillage"  OR "MaasmechelenVillage" )
High Level City searches   Germany, English Terms	( "Frankfurt" OR  "Cologne"  OR "Munich"  ) AND ("Bomb" OR "explosion" OR "shooting"  OR "Stabbing"  OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat")
High Level City searches   Germany, German language	( "Frankfurt" OR  "Cologne"  OR "Munich"  ) AND ("Bombe" OR "explosion" OR "unbeaufsichtigtes Paket"  OR "verdächtiges Paket" OR "Sprengstoffe" OR "Bombendrohung" OR "Schießen" OR "Stechen")
Local Town Searches Wertheim in English	("Wertheim" OR "Wurzburg" OR "Aschaffenburg"  ) AND ("protest" OR "boycott" OR "bomb"  OR "shooting" OR "explosion" OR "stabbing"  OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat" )
Local Town Searches Wertheim German Terms	( "Wertheim" OR "Wurzburg" OR "Aschaffenburg") AND ("protest" OR "protestieren" OR "boykottieren" OR "Boykott" OR "Bombe" OR "Bombendrohung" OR "verdächtiges Paket" OR "unbeaufsichtigtes Paket" OR "explosion"  OR "Sprengstoffe" OR "Schießen" OR "Stechend" OR "Messerstecherei")
Village search Wertheim, 	("Wertheim Village" OR "WertheimVillage" OR "#WertheimVillage" )
PETA Twitter search	peta (gucci OR dior OR armani OR lv OR valentino OR vuitton OR polo OR ralph OR klein OR hilfiger OR prada OR balenciaga OR goose OR burberry OR lvmh)
Peta Broad Search 	"peta" AND ("Italy" OR "UK" OR "Germany" OR "Belgium" OR "Spain" OR "France") AND ("fashion" OR "designer" OR "retail" OR  "clothing" OR "handbag") AND ("protest" OR "appointment" OR "rally" OR "demonstration")
Peta Village Search	("peta") AND ("Ingolstadt" OR "Kildare" OR "vallee" OR "bicester" OR "Wertheim" OR "Rosas" OR "Roca" OR "Fidenza" OR "Maasmechelen")
XR and JSO Broad search	"Extinction Rebellion" 
XR and JSO village search 	("XR" OR "Extinction Rebellion" OR "JSO" OR "Just Stop Oil") AND ("Ingolstadt" OR "Kildare" OR "vallee" OR "bicester" OR "Wertheim" OR "Rosaz" OR "Roca" OR "Fidenza" OR "Maasmechelen")
Ireland / Kildare Protest Groups - Social Media	"https://www.facebook.com/profile.php?id=61556527004170

https://twitter.com/CelbridgePSC

https://www.facebook.com/profile.php?id=100071885352531

https://twitter.com/ipsc48

https://www.ipsc.ie/"
Shoplifting UK	"shoplifting gang" OR "shoplifting trend" OR "pickpocket gang" OR "theft from person" OR "retail crime" OR "shop theft"
High Level City searches   Spain, English   language	("Madrid" OR  "Barcelona")  AND ("Bomb" OR "explosion" OR "shooting" OR "Stabbing" OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat")
High Level City searches   Spain, Spanish   language	("Madrid" OR  "Barcelona")  AND ("Bomba" OR "amenaza de bomba"  OR "explosión" OR "paquete sospechoso" OR "paquete desatendido" OR "explosivos" OR "explosiva" OR "tiroteo"  OR "puñalada")
Local Town Searches Las Rozas and La Roca English	("Mataro" OR "Badalona" OR  "Las Rozas" OR "Madrid" OR "LasRozas") AND ("protest" OR "strikes" OR "boycott" OR "bomb"  OR "shooting" OR "explosion" OR "stabbing" OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat" )
Local Town Searches Las Rozas and La Roca Spanish terms	("Mataro" OR "Badalona" OR  "Las Rozas" OR "Madrid" OR "LasRozas") AND ("protest" OR "boycott" OR "Bomba" OR "amenaza de bomba"  OR "explosión" OR "paquete sospechoso" OR "paquete desatendido" OR "explosivos" OR "explosiva" OR "tiroteo"  OR "puñalada" )
Village Search Las Rozas and La Roca 	("Las Rozas Village" OR "LasRozasVillage" OR "#Lasrozasvillage" OR "La Roca Village" OR "#LaRocaVillage" OR "LaRocaVillage")
Local Town Searches Bicester and Kildare	(Kildare OR "Newbridge" OR "Bicester") AND ("protest" OR "boycott" OR "bomb" OR "shooting" OR "explosion" OR "stabbing" OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat" )
Village Search Bicester and Kildare 	( "Bicester Village" OR "BicesterVillage"  OR "#BicesterVillage"  OR "Kildare Village" OR "KildareVillage" OR "#KildareVillage" )
High Level City searches   UK & Ireland 	("Dublin" OR "Oxford" OR "London") AND ("Bomb" OR "explosion" OR "shooting"  OR "Stabbing" OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat" )
High Level City searches  France  English language	("Paris") AND ("Bomb" OR "explosion" OR "shooting"  OR "Stabbing"  OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat" )
High Level City searches  France  French language	("Paris" ) AND ("Bombe" OR "alerte à la bombe" OR "explosion" OR "explosifs" OR "explosif" OR "explosive"  OR "colis sans surveillance"  OR "colis suspect" OR "tournage" OR "poignarder")
Local Town Searches La Vallee and  French Terms	("Serris" OR "Bailly-Romainvilliers" OR "Magny-le-Hongre"  OR  "Seine-et-Marne" OR "Chessy" OR "Disneylandparis" Or "Disneyland paris") AND ("manifestation" OR "alerte à la bombe" OR "protestation" OR "boycotter"  OR "greve" OR "colis suspect" OR "Bombe" OR "explosion" OR "colis sans surveillance" OR "explosifs" OR "tournage" OR "poignarder")
Local Town Searches La Vallee and Ingolstadt English	("Serris" OR "Bailly-Romainvilliers" OR "Disneyland Paris" OR "Magny-le-Hongre" OR "Chessy" OR "Seine-et-Marne" OR "Ingolstadt"  OR "Disneylandparis")  AND ("protest" OR "strike" OR "boycott" OR "bomb"  OR "shooting" OR "explosion" OR "stabbing"  OR "suspicious package" OR "unattended Package" OR "explosive" OR "bomb threat" )
Local Town Searches Ingolstadt German Terms	("Ingolstadt") AND ("protest" OR "protestieren" OR "strike" or "schlagen" OR "boykottieren" OR "Boykott" OR "Bombe" OR "Bombendrohung" OR "explosion" OR "verdächtiges Paket"   OR "Sprengstoffe" OR "unbeaufsichtigtes Paket" OR "Schießen" OR "Stechend" OR "Messerstecherei")
Village search La Vallee, Ingolstadt	("La Vallee Village" OR "#LaValleeVillage" OR "LaValleeVillage" OR "Ingolstadt Village" OR "IngolstadtVillage" OR "#IngolstadtVillage") 
High Level City searches   Italy, English   language	("Milan" OR "Bologna") AND ("Bomb" OR "Bomb Threat" OR "explosion" OR "shooting" OR "Stabbing" OR "suspicious package" OR "unattended Package" OR "explosive")
High Level City searches   Italy, Italian  language	("Milan"  OR "Bologna") AND ("Bomba" OR "esplosione" OR "sparatoria"  OR "Accoltellamento" OR "esplosivi" OR "pacco incustodito" OR "pacco sospetto" OR "minaccia bomba")
Local Town Searches Fidenza 	("Fidenza" OR "Parma" OR "Piacenza" OR "Cremona") AND ("protest" OR "boycott" OR "bomb" OR "Bomb threat" OR "shooting" OR "explosion" OR "stabbing"  OR "suspicious package" OR "unattended Package" OR "explosive" )
Local Town Searches Fidenza Italian terms	("Fidenza" OR "Parma" OR "Piacenza" OR "Cremona") AND ("boicottare" OR "protesta" OR "Bomba" OR "esplosione" OR "esplosivi"  OR "sparatoria" OR "pacco incustodito" OR "pacco sospetto" OR "Accoltellamento" OR "minaccia bomba")
Village Search Fidenza	( "Fidenza Village" OR "Fidenzavillage" OR "#fidenzavillage" )
Cotswold Designer Outlet	("Cotswold Designer Outlet") AND (incident OR protest OR boycott OR police OR closure OR evacuation OR crime)

London Marylebone	("London Marylebone" OR "Marylebone station") AND (incident OR disruption OR delay OR delays OR closure OR closed OR police OR protest OR bomb OR "bomb threat" OR explosion OR stabbing OR shooting OR "suspicious package" OR "unattended package")

Roermond outlet	("Roermond Outlet" OR Roermond) AND (incident OR protest OR boycott OR police OR closure OR evacuation OR bomb OR "bomb threat" OR explosion OR stabbing OR shooting OR crime)

Roermond outlet conversations	("Roermond Outlet" OR Roermond) AND (incident OR protest OR boycott OR police OR closure OR evacuation OR crime)

Searches for any tweets directed at BV Accounts	("Bicester Village" OR "Kildare Village" OR "La Vallee Village" OR "Maasmechelen Village" OR "Wertheim Village" OR "Ingolstadt Village" OR "Fidenza Village" OR "Las Rozas Village" OR "La Roca Village") AND (incident OR protest OR boycott OR police OR closure OR evacuation OR bomb OR "bomb threat" OR explosion OR stabbing OR shooting)

High Level City searches   Belgium Dutch language	("Brussels" OR "Maastricht" OR "Antwerp") AND (bomb OR explosion OR shooting OR stabbing OR police OR evacuation OR "suspicious package" OR "unattended package" OR protest OR riot)

Local Town Searches  Maasmechelen Dutch Terms	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (bomb OR explosion OR shooting OR stabbing OR police OR evacuation OR "suspicious package" OR "unattended package" OR protest OR boycott)

Local Town Searches Maasmechelen French Terms	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (bomb OR explosion OR shooting OR stabbing OR police OR evacuation OR "suspicious package" OR "unattended package" OR protest OR boycott)

Local Town Searches Maasmechelen English	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (bomb OR "bomb threat" OR explosion OR shooting OR stabbing OR police OR evacuation OR "suspicious package" OR "unattended package" OR protest OR boycott)

Village searches Maasmechelen	("Maasmechelen Village" OR Maasmechelen) AND (incident OR protest OR police OR security OR evacuation OR closure)

Local Town Searches Wertheim in English	("Wertheim" OR "Wurzburg" OR "Aschaffenburg") AND (bomb OR "bomb threat" OR explosion OR shooting OR stabbing OR police OR evacuation OR "suspicious package" OR "unattended package" OR protest OR boycott)

Local Town Searches Wertheim German Terms	("Wertheim" OR "Wurzburg" OR "Aschaffenburg") AND (bomb OR explosion OR shooting OR stabbing OR police OR evacuation OR "suspicious package" OR "unattended package" OR protest OR boycott)

Village search Wertheim, 	("Wertheim Village" OR Wertheim) AND (incident OR protest OR police OR security OR evacuation OR closure)

Peta Village Search	(PETA OR "People for the Ethical Treatment of Animals") AND ("Ingolstadt" OR "Kildare" OR "La Vallee" OR "Bicester" OR "Wertheim" OR "Las Rozas" OR "La Roca" OR "Fidenza" OR "Maasmechelen") AND (protest OR campaign OR demonstration OR boycott)

Ireland / Kildare Protest Groups - Social Media	("Kildare" OR "Newbridge" OR "Celbridge") AND (protest OR demonstration OR activist OR police OR disruption)

Local Town Searches La Vallee and  French Terms	("Serris" OR "Bailly-Romainvilliers" OR "Magny-le-Hongre" OR "Seine-et-Marne" OR "Chessy" OR "Disneyland Paris" OR "Disneylandparis") AND (protest OR strike OR police OR evacuation OR bomb OR "bomb threat" OR explosion OR stabbing OR shooting OR "suspicious package" OR "unattended package")

Local Town Searches La Vallee and Ingolstadt English	("Serris" OR "Bailly-Romainvilliers" OR "Magny-le-Hongre" OR "Chessy" OR "Seine-et-Marne" OR "Disneyland Paris" OR "Ingolstadt") AND (protest OR strike OR boycott OR police OR evacuation OR bomb OR "bomb threat" OR explosion OR stabbing OR shooting OR "suspicious package" OR "unattended package")

Local Town Searches Ingolstadt German Terms	("Ingolstadt") AND (protest OR strike OR boycott OR police OR evacuation OR bomb OR "bomb threat" OR explosion OR stabbing OR shooting OR "suspicious package" OR "unattended package")

Village search La Vallee, Ingolstadt	("La Vallee Village" OR "Ingolstadt Village" OR "LaValleeVillage" OR "IngolstadtVillage") AND (incident OR protest OR police OR security OR evacuation OR closure)

Local Town Searches Fidenza 	("Fidenza" OR "Parma" OR "Piacenza" OR "Cremona") AND (protest OR boycott OR police OR evacuation OR bomb OR "bomb threat" OR explosion OR stabbing OR shooting OR "suspicious package" OR "unattended package")

Local Town Searches Fidenza Italian terms	("Fidenza" OR "Parma" OR "Piacenza" OR "Cremona") AND (protest OR boycott OR police OR evacuation OR bomb OR "bomb threat" OR explosion OR stabbing OR shooting OR "suspicious package" OR "unattended package")

Village Search Fidenza	("Fidenza Village" OR "Fidenzavillage" OR "FidenzaVillage") AND (incident OR protest OR police OR security OR evacuation OR closure)

""".strip()


# ---------------------------
# Parsing + filtering
# ---------------------------
def parse_search_library(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            # keep orphan lines (often URLs) as "raw" entries
            rows.append({"search_name": "UNMAPPED_LINE", "raw_query": line})
            continue
        name, query = line.split("\t", 1)
        rows.append({"search_name": name.strip(), "raw_query": query.strip()})
    return pd.DataFrame(rows)

def is_google_news_compatible(q: str) -> bool:
    q = (q or "").strip().lower()
    if not q:
        return False
    # exclude plain URLs
    if q.startswith("http://") or q.startswith("https://"):
        return False
    # exclude X/Twitter operators/handles that are generally not useful for Google News RSS
    if "to:" in q or q.startswith("@") or " @" in q:
        return False
    return True

def google_news_rss_url(query: str, past_days: int) -> str:
    # add time window
    full = f"{query} when:{past_days}d"
    q = urllib.parse.quote(full)
    return f"https://news.google.com/rss/search?q={q}&hl=en-GB&gl=GB&ceid=GB:en"

def collect_google_news(df_searches: pd.DataFrame, past_days: int, max_items: int) -> pd.DataFrame:
    collected_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_rows = []

    for _, r in df_searches.iterrows():
        name = r["search_name"]
        q = r["raw_query"]

        rss = google_news_rss_url(q, past_days)
        feed = feedparser.parse(rss)

        for entry in feed.entries[:max_items]:
            out_rows.append({
                "search_name": name,
                "search_query": q,
                "title": entry.get("title", ""),
                
                "published": entry.get("published", ""),
                "link": entry.get("link", ""),
               
                "past_days": past_days,
               
            })

    return pd.DataFrame(out_rows)

# ---------------------------
# Run
# ---------------------------
search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
search_df["google_news_compatible"] = search_df["raw_query"].apply(is_google_news_compatible)

# Keep ALL searches in this table (audit), but only run the compatible ones
to_run = search_df[search_df["google_news_compatible"]].copy()

results = collect_google_news(to_run, past_days=PAST_DAYS, max_items=MAX_ITEMS)

# Strict enforcement (this is the fix)
results = filter_last_n_days(results, n_days=PAST_DAYS)


# de-dupe exact links across searches
if not results.empty:
    results = results.drop_duplicates(subset=["link"]).reset_index(drop=True)

# Save outputs
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
results_file = f"google_news_{stamp}_past{PAST_DAYS}d.csv"
audit_file = f"search_audit_{stamp}.csv"

results.to_csv(results_file, index=False)
search_df.to_csv(audit_file, index=False)

print(f"Saved results: {results_file} | rows={len(results)}")
print(f"Saved audit:   {audit_file}  | searches={len(search_df)}")
print("\nNon-compatible searches (kept in audit, not run):")
print(search_df[~search_df["google_news_compatible"]][["search_name","raw_query"]].head(15).to_string(index=False))


# %%
pip install sentence-transformers scikit-learn pandas


# %%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# -----------------------
# SETTINGS
# -----------------------
INFILE = "google_news_20251218_110818_UTC_past1d.csv"
OUTFILE = None  # auto-name with timestamp if None
AUDITFILE = None

DUP_THRESHOLD = 0.60  # use same as your test (0.90–0.95 typical)
MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------
# LOAD
# -----------------------
df = pd.read_csv(INFILE)

# Build text used for semantic similarity
df["compare_text"] = (
    df["title"].fillna("").astype(str))

# Keep empty-text rows untouched, only dedupe rows with content
mask = df["compare_text"].str.len() > 0
df_work = df[mask].copy().reset_index(drop=True)
orig_idx = df.index[mask].to_numpy()  # original row indices in df

# -----------------------
# EMBED + SIMILARITY
# -----------------------
model = SentenceTransformer(MODEL_NAME)
emb = model.encode(df_work["compare_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
sim = cosine_similarity(emb, emb)

n = sim.shape[0]

# -----------------------
# UNION-FIND to group duplicates (connected components)
# -----------------------
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

# Build edges for near-duplicates
for i in range(n):
    for j in range(i + 1, n):
        if sim[i, j] >= DUP_THRESHOLD:
            union(i, j)

# Group indices by component
groups = {}
for i in range(n):
    r = find(i)
    groups.setdefault(r, []).append(i)

# -----------------------
# KEEP RULE: keep earliest row from the ORIGINAL FILE
# -----------------------
keep_work = set()
audit_rows = []

for g in groups.values():
    if len(g) == 1:
        keep_work.add(g[0])
        continue

    # map to original row indices; keep the earliest in original file order
    g_map = [(int(orig_idx[i]), i) for i in g]
    g_map.sort(key=lambda x: x[0])

    keep_orig, keep_i = g_map[0]
    keep_work.add(keep_i)

    for drop_orig, drop_i in g_map[1:]:
        audit_rows.append({
            "kept_original_row": keep_orig,
            "dropped_original_row": int(drop_orig),
            "similarity": float(sim[keep_i, drop_i]),
            "kept_title": df.loc[keep_orig, "title"],
            "dropped_title": df.loc[int(drop_orig), "title"],
        })

# Determine original rows to drop
kept_orig_rows = {int(orig_idx[i]) for i in keep_work}
drop_orig_rows = set(map(int, orig_idx.tolist())) - kept_orig_rows

keep_mask = np.ones(len(df), dtype=bool)
for r in drop_orig_rows:
    keep_mask[r] = False

df_clean = df.loc[keep_mask].drop(columns=["compare_text"], errors="ignore").reset_index(drop=True)
audit = pd.DataFrame(audit_rows)

# -----------------------
# SAVE
# -----------------------
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_UTC")
if OUTFILE is None:
    OUTFILE = f"google_news_results_dedup_{ts}.csv"
if AUDITFILE is None:
    AUDITFILE = f"google_news_dedup_audit_{ts}.csv"

df_clean.to_csv(OUTFILE, index=False)
audit.to_csv(AUDITFILE, index=False)

print(f"Original rows: {len(df)}")
print(f"Cleaned rows : {len(df_clean)}")
print(f"Saved cleaned file: {OUTFILE}")
print(f"Saved audit file  : {AUDITFILE}")


# %%



