# -*- coding: utf-8 -*-
"""
Streamlit app — PetBot (cards with first photo only, robust media parsing)
(Version: best-model-only = hybrid_raw_facet + ner_light_nom, no MMR)
"""
import streamlit as st
st.set_page_config(page_title="PetBot Search", layout="wide")

import os, time, ast, re, json
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np

from src.config import get_blob_settings, local_ner_dir, local_mr_dir, local_pets_csv_path
from src.azure_io import download_prefix_flat, smart_download_single_blob
from src.models import load_ner_pipeline, load_mr_model, load_faiss_index
from src.retrieval import (
    only_text, BM25,
    parse_facets_from_text, entity_spans_to_facets, sanitize_facets_ner_light,
    filter_with_relaxation, make_boosted_query,
    emb_search, mmr_rerank  # mmr_rerank imported but unused (kept for compatibility)
)
from src.ui import sidebar_controls

# ---- Optional fuzzy breed mapping ----
try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False
    process = fuzz = None



# -------------------------------------------
# CONFIG
# -------------------------------------------
# Keep BREED + AGE hard: do not include in relaxation order
RELAX_ORDER = ["colors_any", "state", "gender"]  # animal kept if present (breed & age fixed if provided)

# Global floors
MIN_CAND_FLOOR_BASE = 300
LEX_POOL = 2000
EMB_POOL = 200

# Hybrid weights (match your eval)
HYBRID_W = {"lex": 0.1, "emb": 0.9}

# -------------------------------------------
# Helpers
# -------------------------------------------
COLOR_MAP = {"golden": "yellow", "gold": "yellow", "cream": "white", "ginger": "orange"}
def normalize_color(c: str) -> str:
    c = (c or "").strip().lower()
    return COLOR_MAP.get(c, c)

def _safe_list_from_cell(x):
    """Parse strings like "['a','b']" or '["a","b"]' or comma strings into list."""
    if isinstance(x, list): return x
    if x is None: return []
    s = str(x).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list): return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list): return obj
        except Exception:
            return []
    if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def parse_colors_cell(x) -> List[str]:
    return [normalize_color(str(t)) for t in _safe_list_from_cell(x) if str(t).strip()]

def resolve_overlaps_longest(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def lab(z: Dict[str, Any]) -> str:
        L = str(z.get("entity_group") or z.get("label") or "").upper()
        L = re.sub(r"^[BI]-", "", L)
        return L
    spans = sorted(spans or [], key=lambda s: (int(s.get("start", 0)), -(int(s.get("end", 0)) - int(s.get("start", 0)))))
    kept: List[Dict[str, Any]] = []
    for s in spans:
        s_start, s_end = int(s.get("start", 0)), int(s.get("end", 0))
        s_lab = lab(s)
        s_len = s_end - s_start
        drop = False
        for t in list(kept):
            t_start, t_end = int(t.get("start", 0)), int(t.get("end", 0))
            t_lab = lab(t)
            t_len = t_end - t_start
            overlaps = not (s_end <= t_start or s_start >= t_end)
            if overlaps:
                if (s_lab == "COLOR" and t_lab == "BREED") or (t_len >= s_len):
                    drop = True
                    break
        if not drop:
            kept = [t for t in kept if not (int(t.get("start", 0)) >= s_start and int(t.get("end", 0)) <= s_end)]
            kept.append(s)
    return kept

def safe_merge(model_v, rule_v):
    if model_v is None: return rule_v
    if isinstance(model_v, str) and model_v.strip() == "": return rule_v
    if isinstance(model_v, (list, tuple)) and len(model_v) == 0: return rule_v
    return model_v

BREED_ALIASES = {
    "husky": ["siberian husky", "alaskan husky"],
    "gsd": ["german shepherd", "german shepherd dog"],
    "gr": ["golden retriever"],
    "grd": ["golden retriever"],
}

def map_breed_to_catalog(breed_text: Optional[str], catalog_breeds: List[str], min_score: int = 87) -> Optional[str]:
    if not breed_text: return None
    b = str(breed_text).strip().lower().rstrip(",.;:")
    if not b: return None
    if b in catalog_breeds: return b
    if b in BREED_ALIASES:
        for cand in BREED_ALIASES[b]:
            c = cand.lower()
            if c in catalog_breeds:
                return c
    if not _HAS_FUZZ or not catalog_breeds: return b
    cand, score, _ = process.extractOne(b, catalog_breeds, scorer=fuzz.token_sort_ratio)
    return cand if score >= min_score else b

def _minmax(d: Dict[int, float]) -> Dict[int, float]:
    if not d: return {}
    vals = np.fromiter(d.values(), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    den = (hi - lo) or 1.0
    return {k: (v - lo) / den for k, v in d.items()}

def expand_kid_friendly(q: str) -> str:
    t = q.lower()
    if "kid" in t or "child" in t or "family" in t:
        extras = " kid-friendly child-friendly family-friendly gentle with kids good with children good with kids family dog"
        return q + " " + extras
    return q

def _age_years_from_months(age_months) -> str:
    try:
        m = float(age_months); y = m/12.0
        if m < 12: return f"{int(round(m))} mo (puppy/kitten)"
        return f"{y:.1f} yrs"
    except Exception:
        return "—"

def debug_print(spans, mf, rf, facets, mapped_breed, inferred_animal, floor, df_prefilter_rows, age_used):
    st.write("**NER raw spans**:", spans)
    st.write("**Model facets (mf)**:", mf)
    st.write("**Rule facets (rf)**:", rf)
    st.write("**Merged facets (pre-map)**:", facets)
    if mapped_breed is not None:
        st.write("**Breed mapped to catalog**:", mapped_breed)
    if inferred_animal is not None:
        st.write("**Animal inferred from breed**:", inferred_animal)
    st.write(f"**Adaptive floor used**: {floor}")
    if df_prefilter_rows is not None:
        st.write(f"**Pre-filter by breed substring rows**: {df_prefilter_rows}")
    st.write(f"**Age facet used**: {age_used}")

# ---------- Cards UI (1 photo only) ----------
def _badge_bool(x, label):
    v = str(x or "").strip().lower()
    if v in {"true","yes","y","1"}: return f"✅ {label}"
    if v in {"false","no","n","0"}: return f"❌ {label}"
    if v in {"unknown", "nan", ""}: return f"➖ {label}"
    return f"ℹ️ {label}: {x}"

def _comma_join_listlike(x):
    if isinstance(x, list):
        return ", ".join([str(t) for t in x if str(t).strip()]) or "—"
    if isinstance(x, str) and x.strip():
        return x
    return "—"

def _first_photo_url(row) -> Optional[str]:
    photos = row.get("photo_links")
    if not isinstance(photos, list):
        photos = _safe_list_from_cell(photos)
    if photos:
        url = str(photos[0]).strip().strip('"').strip("'")
        return url if url else None
    return None

def render_pet_card(row: pd.Series):
    pid = int(row.get("pet_id"))
    name = str(row.get("name") or f"Pet {pid}")
    url  = str(row.get("url") or "")
    animal = (row.get("animal") or "").title()
    breed  = str(row.get("breed") or "—")
    gender = (row.get("gender") or "—").title()
    state  = (row.get("state") or "—").title()
    color  = str(row.get("color") or "—")
    colors_canon = row.get("colors_canonical")
    colors_txt = _comma_join_listlike(colors_canon if isinstance(colors_canon, list) else color)
    age_mo = row.get("age_months"); age_yrs_txt = _age_years_from_months(age_mo)
    size = str(row.get("size") or "—").title()
    fur  = str(row.get("fur_length") or "—").title()
    cond = str(row.get("condition") or "—").title()
    vacc = _badge_bool(row.get("vaccinated"), "vaccinated")
    dewm = _badge_bool(row.get("dewormed"), "dewormed")
    neut = _badge_bool(row.get("neutered"), "neutered")
    spay = _badge_bool(row.get("spayed"),    "spayed")

    if url: st.markdown(f"### [{name}]({url})")
    else:   st.markdown(f"### {name}")

    img_url = _first_photo_url(row)
    if st.session_state.get("debug_media", False):
        st.caption(f"DEBUG photo_links[0]: {repr(img_url)}")

    if img_url:
        try:
            st.image(img_url, use_column_width=True)
        except TypeError:
            st.image(img_url)
        except Exception:
            st.markdown(f"![photo]({img_url})")
    else:
        st.info("No photo available.")

    st.write(
        f"**{animal}** • **Breed:** {breed} • **Gender:** {gender} • "
        f"**Age:** {age_yrs_txt} • **State:** {state}"
    )
    st.write(
        f"**Color(s):** {colors_txt} • **Size:** {size} • **Fur:** {fur} • **Condition:** {cond}"
    )
    st.markdown(" | ".join([vacc, dewm, neut, spay]))

    desc = str(row.get("description_clean") or "").strip()
    if desc:
        excerpt = desc if len(desc) < 300 else (desc[:300].rsplit(" ", 1)[0] + "…")
        with st.expander("Description", expanded=False):
            st.write(excerpt)

def render_results_grid(res_df: pd.DataFrame, max_cols: int = 3):
    if res_df.empty:
        st.warning("No results to show.")
        return
    rows = [r for _, r in res_df.iterrows()]
    n = len(rows)
    col_count = max(1, min(max_cols, n))
    for i in range(0, n, col_count):
        cols = st.columns(col_count, gap="medium")
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= n: continue
            with col:
                with st.container(border=True):
                    render_pet_card(rows[idx])

# -------------------------------------------
@st.cache_resource(show_spinner=True)
def bootstrap_and_load():
    cfg = get_blob_settings()
    conn = cfg["connection_string"]

    st.write(f"Using **ML container**: `{cfg['ml_container']}` | **pets container**: `{cfg['pets_container']}`")

    with st.spinner("Downloading NER model from Azure Blob (flat folder)..."):
        download_prefix_flat(conn, cfg["ml_container"], cfg["ner_prefix"], local_ner_dir())

    with st.spinner("Downloading Matching/Ranking model (flat folder)..."):
        download_prefix_flat(conn, cfg["ml_container"], cfg["mr_prefix"], local_mr_dir())

    with st.spinner("Downloading pet CSV..."):
        st.write(f"Fetching blob `{cfg['pets_csv_blob']}` from container `{cfg['pets_container']}`")
        smart_download_single_blob(conn, cfg["pets_container"], cfg["pets_csv_blob"], local_pets_csv_path())

    ner = load_ner_pipeline(local_ner_dir())
    student, doc_ids, doc_vecs = load_mr_model(local_mr_dir())

    faiss_index = load_faiss_index(local_mr_dir(), dim=doc_vecs.shape[1])
    if faiss_index is not None:
        st.success(f"FAISS index loaded from {local_mr_dir()} (ntotal may display in logs).")
    else:
        st.info("FAISS not found — falling back to fast brute-force similarity.")

    dfp = pd.read_csv(local_pets_csv_path())

    # Expect columns
    for col in ["pet_id","animal","breed","gender","state","colors_canonical","doc",
                "description_clean","age_months","photo_links","video_links","color","size","fur_length","condition","url","name"]:
        if col not in dfp.columns:
            st.warning(f"Column '{col}' missing in CSV; some filters or UI may be less effective.")

    # normalize key text cols to lower for matching
    for c in ("animal", "gender", "state", "breed", "size", "fur_length", "condition"):
        if c in dfp.columns:
            dfp[c] = dfp[c].astype(str).fillna("").str.strip().str.lower()

    # parse list-like cols
    if "colors_canonical" in dfp.columns:
        dfp["colors_canonical"] = dfp["colors_canonical"].apply(parse_colors_cell)

    # *** IMPORTANT: parse media columns to real lists ***
    for media_col in ["photo_links", "video_links"]:
        if media_col in dfp.columns:
            dfp[media_col] = dfp[media_col].apply(_safe_list_from_cell)

    # breed -> animal map (for inference)
    breed_to_animal = {}
    if "breed" in dfp.columns and "animal" in dfp.columns:
        tmp = (dfp[["breed","animal"]]
                 .dropna()
                 .groupby("breed")["animal"]
                 .agg(lambda s: s.value_counts().idxmax()))
        breed_to_animal = tmp.to_dict()

    # BM25 over doc text (use 'doc' if available)
    docs_raw = {int(r["pet_id"]): only_text(str(r.get("doc", ""))) for _, r in dfp.iterrows()}
    bm25 = BM25().fit(docs_raw)

    # Breed catalog
    breed_catalog = []
    if "breed" in dfp.columns:
        breed_catalog = sorted(set([b for b in dfp["breed"].astype(str).str.lower().tolist() if b]))

    return ner, student, doc_ids, doc_vecs, faiss_index, dfp, bm25, breed_catalog, breed_to_animal

ner, student, doc_ids, doc_vecs, faiss_index, dfp, bm25, breed_catalog, breed_to_animal = bootstrap_and_load()

st.title("🐾 PetBot — NER + Hybrid Search (Cards & Media)")

ui = sidebar_controls()
# ---- Lock UI to best model: hybrid (BM25 + Embeddings), no MMR ----
ui["method"] = "hybrid"
ui["use_mmr"] = False
ui["mmr_lambda"] = 0.35  # unused when MMR=False
ui["topk"] = max(1, int(ui.get("topk", 12)))

st.caption("Tip: try “white female **puppy** poodle in Selangor good with kids”")

q = st.text_input("Your search", value="brown female cat in selangor")
if not q.strip():
    st.stop()

# -------------------------------------------
# NER + facets + (hard-ish) filtering — with AGE facet
# -------------------------------------------
t0 = time.time()

raw_spans = ner([q[:300]])[0] if q else []
spans = resolve_overlaps_longest(raw_spans)
mf = entity_spans_to_facets(spans)             # from src.retrieval (ANIMAL/BREED/GENDER/COLOR/STATE)
rf = parse_facets_from_text(q)                 # rule-based (ANIMAL/GENDER/COLORS/STATE)

# AGE facet: numeric or puppy/kitten
age_floor_mo, age_ceil_mo = None, None
tq = only_text(q)
m_age_mo = re.search(r"\b([1-9][0-9]?)\s*mo(nth)?s?\b", tq)
m_age_yr = re.search(r"\b([1-9][0-9]?)\s*y(ear)?s?\b", tq)
if "puppy" in tq or "kitten" in tq:
    age_floor_mo, age_ceil_mo = 0, 12
elif m_age_mo:
    val = int(m_age_mo.group(1)); age_floor_mo, age_ceil_mo = max(0, val-3), val+3
elif m_age_yr:
    val = int(m_age_yr.group(1)); mo = 12*val; age_floor_mo, age_ceil_mo = max(0, mo-6), mo+6

facets = {
    "animal": safe_merge(mf.get("animal"), rf.get("animal")),
    "breed":  safe_merge(mf.get("breed"),  rf.get("breed")),
    "gender": safe_merge(mf.get("gender"), rf.get("gender")),
    "colors_any": safe_merge(mf.get("colors_any"), rf.get("colors_any")),
    "state":  safe_merge(mf.get("state"),  rf.get("state")),
}
if facets.get("colors_any"):
    facets["colors_any"] = sorted({normalize_color(c) for c in facets["colors_any"] if c})

mapped_breed = None
if facets.get("breed"):
    mapped_breed = map_breed_to_catalog(facets["breed"], breed_catalog, min_score=87)
    facets["breed"] = mapped_breed

inferred_animal = None
if not facets.get("animal") and mapped_breed:
    inferred_animal = breed_to_animal.get(mapped_breed)
    if inferred_animal:
        facets["animal"] = inferred_animal

facets = sanitize_facets_ner_light(facets)

# -------- Adaptive floors (breed/age hard) --------
BASE_FLOOR = MIN_CAND_FLOOR_BASE
floor = BASE_FLOOR
has_breed = bool(facets.get("breed"))
has_color = bool(facets.get("colors_any"))
has_state = bool(facets.get("state"))
has_gender = bool(facets.get("gender"))
has_age = age_floor_mo is not None or age_ceil_mo is not None

if has_breed and has_age and has_color:
    floor = 20
elif has_breed and has_age:
    floor = 30
elif has_breed and (has_state or has_gender):
    floor = 50
elif has_breed:
    floor = 80
elif has_age and (has_state or has_gender):
    floor = 120
elif has_age or has_color:
    floor = 160

if ui["strict_mode"]:
    floor = max(10, floor // 2)

# -------- Hard breed pre-filter --------
df_to_filter = dfp
prefilter_rows = None
if facets.get("breed") and "breed" in dfp.columns:
    b = re.escape(facets["breed"])
    df_b = dfp[dfp["breed"].astype(str).str.contains(rf"\b{b}\b", na=False)]
    prefilter_rows = int(len(df_b))
    if len(df_b) >= 1:
        df_to_filter = df_b

with st.expander("🔎 Debug — NER spans & facets"):
    st.write("Your search:", q)
    debug_print(spans, mf, rf, facets, mapped_breed, inferred_animal, floor, prefilter_rows, (age_floor_mo, age_ceil_mo))

# Relaxation (breed & age not relaxed)
cand_df, used = filter_with_relaxation(df_to_filter, facets, order=RELAX_ORDER, min_floor=floor)

# Apply AGE hard filter if present
def _age_ok(mo):
    if mo is None or (isinstance(mo, float) and np.isnan(mo)): return False
    try:
        m = float(mo)
        if age_floor_mo is not None and m < age_floor_mo: return False
        if age_ceil_mo is not None and m > age_ceil_mo: return False
        return True
    except Exception:
        return False

age_used_flag = False
if has_age and "age_months" in cand_df.columns:
    before = len(cand_df)
    cand_df = cand_df[cand_df["age_months"].apply(_age_ok)]
    after = len(cand_df)
    age_used_flag = True
    if after == 0 and not ui["strict_mode"]:
        def _age_soft(mo):
            try:
                m = float(mo)
                lo = 0 if age_floor_mo is None else max(0, age_floor_mo - 6)
                hi = 999 if age_ceil_mo is None else (age_ceil_mo + 6)
                return lo <= m <= hi
            except Exception:
                return False
        cand_df = df_to_filter[df_to_filter["age_months"].apply(_age_soft)]

# Build boosted query (+ kid-friendly expansions help BM25)
boost_q = make_boosted_query(q, used)
boost_q = expand_kid_friendly(boost_q)

# Show used facets (augment with age)
used_aug = dict(used)
if age_used_flag:
    used_aug["age_months_range"] = [age_floor_mo, age_ceil_mo]
st.write(f"**Used facets**: {used_aug} | **candidates**: {len(cand_df)}")
st.caption(f"Boosted query: `{boost_q}`")
st.caption(f"NER+Relax time: {time.time()-t0:.3f}s")

pool_ids = set(cand_df["pet_id"].astype(int).tolist())
by_id = cand_df.set_index("pet_id")

# -------------------------------------------
# Facet-preserving re-ranker (breed & age extra weight)
# -------------------------------------------
orig_intent = {
    "breed": facets.get("breed"),
    "gender": facets.get("gender"),
    "state": facets.get("state"),
    "colors_any": set(facets.get("colors_any") or []),
    "age_floor_mo": age_floor_mo,
    "age_ceil_mo":  age_ceil_mo,
}
def facet_score(pid: int) -> float:
    row = by_id.loc[int(pid)].to_dict()
    s = 0.0
    if orig_intent["breed"] and re.search(rf"\b{re.escape(orig_intent['breed'])}\b", str(row.get("breed","")), flags=re.I):
        s += 3.0
    try:
        m = float(row.get("age_months", np.nan))
        if orig_intent["age_floor_mo"] is not None or orig_intent["age_ceil_mo"] is not None:
            lo = orig_intent["age_floor_mo"] if orig_intent["age_floor_mo"] is not None else 0
            hi = orig_intent["age_ceil_mo"] if orig_intent["age_ceil_mo"] is not None else 999
            if lo <= m <= hi: s += 2.5
    except Exception:
        pass
    if orig_intent["gender"] and str(row.get("gender","")).lower() == str(orig_intent["gender"]).lower():
        s += 1.5
    if orig_intent["state"] and str(row.get("state","")).lower() == str(orig_intent["state"]).lower():
        s += 1.0
    want = orig_intent["colors_any"]
    if want and isinstance(row.get("colors_canonical"), list):
        if any(c in row["colors_canonical"] for c in want):
            s += 0.7
    return s

# -------------------------------------------
# BEST MODEL ONLY: Hybrid (BM25 + Embeddings), no MMR
# -------------------------------------------
topk = ui["topk"]

def rerank_with_facets(hits_list):
    return sorted(hits_list, key=lambda x: (facet_score(x[0]), x[1]), reverse=True)

if not pool_ids:
    hits = []
else:
    # 1) BM25 over filtered pool
    lex_all = bm25.search(boost_q, topk=LEX_POOL)
    s_lex = {pid: s for pid, s in lex_all if pid in pool_ids}

    # 2) Embedding search over filtered pool
    emb_all = emb_search(
        boost_q, student, doc_ids, doc_vecs,
        pool_topn=EMB_POOL, faiss_index=faiss_index
    )
    s_emb = {pid: s for pid, s in emb_all if pid in pool_ids}

    # 3) Normalize and combine with hybrid weights (lex:emb = 0.1:0.9)
    nlex, nemb = _minmax(s_lex), _minmax(s_emb)
    wl, we = HYBRID_W["lex"], HYBRID_W["emb"]
    combo = {pid: wl*nlex.get(pid, 0.0) + we*nemb.get(pid, 0.0) for pid in set(nlex) | set(nemb)}

    # 4) Top-K + facet-preserving re-rank (breed & age heavier)
    hits = sorted(combo.items(), key=lambda x: -x[1])[:max(EMB_POOL, topk)]
    hits = hits[:topk]
    hits = rerank_with_facets(hits)

# -------------------------------------------
# Show results (Cards / Table)
# -------------------------------------------
if not hits:
    st.warning("No results. Try relaxing filters or removing a facet (e.g., color) — or disable Strict Mode.")
else:
    base_cols = [
        "name","animal","breed","gender","state","color","colors_canonical",
        "size","fur_length","condition","age_months","description_clean",
        "url","photo_links","video_links"
    ]
    display_cols = [c for c in base_cols if c in dfp.columns]
    res_df = (
        dfp.set_index("pet_id")
           .loc[[int(pid) for pid, _ in hits], display_cols]
           .reset_index()
    )
    res_df["score"] = [float(s) for _, s in hits]

    if ui["view_mode"] == "Cards":
        # Limit to a reasonable number for performance (top 12 default)
        top_for_cards = min(len(res_df), ui["topk"])
        render_results_grid(res_df.iloc[:top_for_cards].copy(), max_cols=ui["grid_cols"])
    else:
        # add age in years col for table
        def _age_years(x):
            try:
                m = float(x)
                return f"{m/12.0:.1f}"
            except Exception:
                return ""
        res_df["age_years"] = res_df["age_months"].apply(_age_years) if "age_months" in res_df else ""
        cols = ["pet_id","name","animal","breed","gender","state","age_months","age_years","colors_canonical","size","fur_length","condition","url","score"]
        cols = [c for c in cols if c in res_df.columns]
        st.dataframe(res_df[cols], use_container_width=True)
