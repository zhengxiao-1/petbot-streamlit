# src/retrieval.py
import re, math, time, ast, json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# ---------- text clean ----------
def only_text(s: str) -> str:
    s = re.sub(r"http\S+", " ", str(s or ""))
    s = re.sub(r"[^a-zA-Z0-9\s/,+()\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

# ---------- tiny BM25 ----------
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.docs=[]; self.ids=[]
        self.df={}; self.idf={}
        self.doc_lens=None; self.avgdl=0.0
    def _tok(self, text): return re.findall(r"[a-z0-9]+", text.lower())
    def fit(self, doc_map: Dict[int, str]):
        self.ids = np.array(list(doc_map.keys()), dtype=int)
        self.docs = [self._tok(doc_map[i]) for i in self.ids]
        N = len(self.docs)
        self.doc_lens = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(np.mean(self.doc_lens)) if N else 0.0
        from collections import defaultdict
        df = defaultdict(int)
        for d in self.docs:
            for t in set(d): df[t]+=1
        self.df = dict(df)
        self.idf = {t: math.log(1 + (N - df_t + 0.5)/(df_t + 0.5)) for t, df_t in self.df.items()}
        return self
    def search(self, query: str, topk=2000):
        q = self._tok(query)
        scores = np.zeros(len(self.docs), dtype=float)
        for i, d in enumerate(self.docs):
            from collections import Counter
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t not in self.idf: continue
                idf = self.idf[t]; f = tf.get(t, 0)
                denom = f + self.k1*(1 - self.b + self.b*len(d)/(self.avgdl + 1e-9))
                s += idf * (f*(self.k1+1))/(denom + 1e-9)
            scores[i] = s
        if topk < len(scores):
            idx = np.argpartition(-scores, topk)[:topk]
            idx = idx[np.argsort(-scores[idx])]
        else:
            idx = np.argsort(-scores)[::-1]
        return [(int(self.ids[i]), float(scores[i])) for i in idx[:topk]]

# ---------- NER light: parse ----------
def parse_facets_from_text(user_text: str) -> Dict[str, Any]:
    t = only_text(user_text)
    animal = None
    if re.search(r"\b(dog|puppy|canine)\b", t): animal = "dog"
    if re.search(r"\b(cat|kitten|feline)\b", t): animal = "cat"
    gender = None
    if re.search(r"\bmale\b", t): gender = "male"
    elif re.search(r"\bfemale\b", t): gender = "female"
    state = None
    # quick lookup for Malaysian states (extend as needed)
    states = ["selangor","kuala lumpur","putrajaya","johor","penang","pahang","sabah","sarawak","perak","kedah","kelantan","melaka","negeri sembilan","perlis","terengganu","labuan"]
    for s in states:
        if re.search(rf"\b{s}\b", t): state = s; break
    COLORS = ["black","white","brown","beige","tan","golden","cream","gray","grey","silver",
              "orange","ginger","brindle","calico","tortoiseshell","tortie","tabby","yellow"]
    found = [c for c in COLORS if re.search(rf"\b{re.escape(c)}\b", t)]
    colors_any = sorted(set(["gray" if c=="grey" else c for c in found])) if found else None
    return {"animal": animal, "gender": gender, "colors_any": colors_any, "breed": None, "state": state}

def entity_spans_to_facets(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    facets = {"animal": None, "breed": None, "gender": None, "colors_any": [], "state": None}
    for sp in (spans or []):
        label = re.sub(r"^[BI]-", "", str(sp.get("entity_group") or sp.get("label") or "")).upper()
        text  = only_text(sp.get("word") or sp.get("entity") or "")
        if not label or not text:
            continue
        if label == "ANIMAL" and text in {"dog","cat"}:
            facets["animal"] = text
        elif label == "BREED":
            facets["breed"] = text
        elif label == "GENDER" and text in {"male","female"}:
            facets["gender"] = text
        elif label == "COLOR":
            facets["colors_any"].append(text)
        elif label in {"STATE","LOCATION","CITY"}:
            facets["state"] = text
    facets["colors_any"] = sorted(set(facets["colors_any"])) or None
    return facets

def sanitize_facets_ner_light(f: dict) -> dict:
    out = {k: f.get(k) for k in ["animal","breed","gender","colors_any","state"]}
    if out.get("colors_any"):
        cols = []
        for c in out["colors_any"]:
            cols.append("gray" if c == "grey" else c)
        out["colors_any"] = sorted(set(cols)) or None
    return out

# robust list parser for colors_canonical
def safe_list_from_cell(x) -> List[str]:
    if x is None: return []
    if isinstance(x, float) and np.isnan(x): return []
    if isinstance(x, list): return [str(t).strip().lower() for t in x if str(t).strip()]
    s = str(x).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list): return [str(t).strip().lower() for t in obj if str(t).strip()]
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list): return [str(t).strip().lower() for t in obj if str(t).strip()]
        except Exception:
            return []
    if "," in s:
        return [t.strip().lower() for t in s.split(",") if t.strip()]
    return [s.lower()] if s else []

# ---------- candidate filtering (relaxed) ----------
def filter_once(df: pd.DataFrame, facets: Dict[str, Any]) -> pd.DataFrame:
    out = df
    # animal (string)
    if "animal" in out.columns and facets.get("animal"):
        out = out[out["animal"].astype(str).str.lower() == facets["animal"]]
    # gender (string)
    if "gender" in out.columns and facets.get("gender") in {"male","female"}:
        out = out[out["gender"].astype(str).str.lower() == facets["gender"]]
    # state (string, case-insensitive)
    if "state" in out.columns and facets.get("state"):
        st = facets["state"].lower()
        out = out[out["state"].astype(str).str.lower().str.contains(re.escape(st), na=False)]
    # colors (canonical list in column)
    if "colors_canonical" in out.columns and facets.get("colors_any"):
        want = set([c.lower() for c in facets["colors_any"]])
        def _hit(xs):
            xs_list = safe_list_from_cell(xs)
            return any(c in xs_list for c in want)
        out = out[out["colors_canonical"].apply(_hit)]
    # breed (soft substring)
    if "breed" in out.columns and facets.get("breed"):
        b = str(facets["breed"]).lower()
        out = out[out["breed"].astype(str).str.lower().str.contains(re.escape(b), na=False)]
    return out.reset_index(drop=True)

def filter_with_relaxation(df: pd.DataFrame, facets: Dict[str, Any], order: List[str], min_floor: int = 300) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    used = dict(facets or {})
    res = filter_once(df, used)
    for key in order:
        if len(res) >= min_floor:
            break
        if used.get(key) not in (None, [], ""):
            used[key] = None
            res = filter_once(df, used)
    if len(res) == 0:
        # keep only animal if present
        keep = {"animal": used.get("animal"), "breed": None, "gender": None, "colors_any": None, "state": None}
        used = keep
        res = filter_once(df, used)
        if len(res) == 0:
            res = df
    return res, used

def make_boosted_query(user_text: str, used: dict) -> str:
    facet_terms = []
    for key in ["animal","breed","gender","state"]:
        if used.get(key):
            facet_terms.append(str(used[key]).lower())
    if used.get("colors_any"):
        facet_terms.extend([c.lower() for c in used["colors_any"]])
    return only_text(user_text + " " + " ".join(facet_terms))

# ---------- embedding search ----------
def ann_search_faiss(faiss_index, q_vec: np.ndarray, pool_topn: int) -> List[Tuple[int, float]]:
    import numpy as np
    D, I = faiss_index.search(q_vec.reshape(1, -1).astype("float32"), pool_topn)
    hits = []
    for idx, score in zip(I[0], D[0]):
        hits.append((int(idx), float(score)))
    return hits

def emb_search(
    q: str,
    student,
    doc_ids: np.ndarray,
    doc_vecs: np.ndarray,
    pool_topn: int = 200,
    faiss_index = None,
) -> List[Tuple[int, float]]:
    qv = student.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
    if faiss_index is not None:
        hits = ann_search_faiss(faiss_index, qv, pool_topn)
        # FAISS returns positions; convert to pet IDs if your index was built on doc_vecs in same order
        # If index was built on doc_vecs directly: ids = doc_ids[idx]
        return [(int(doc_ids[i]), float(s)) for (i, s) in hits]
    # brute force
    sims = doc_vecs @ qv
    if pool_topn < sims.shape[0]:
        idx = np.argpartition(-sims, pool_topn)[:pool_topn]
        idx = idx[np.argsort(-sims[idx])]
    else:
        idx = np.argsort(-sims)[::-1]
    return [(int(doc_ids[i]), float(sims[i])) for i in idx[:pool_topn]]

# ---------- MMR re-ranking ----------
def mmr_rerank(
    candidates: List[Tuple[int, float]],
    doc_embs: np.ndarray,
    doc_ids: np.ndarray,
    q_vec: np.ndarray,
    k: int = 10,
    lambda_mult: float = 0.7,
) -> List[Tuple[int, float]]:
    if not candidates:
        return []
    # Map pid->idx
    idx_map = {int(pid): i for i, pid in enumerate(doc_ids)}
    # Precompute sim to query
    sel = []
    cand = candidates.copy()
    picked = set()
    while cand and len(sel) < k:
        best_pid, best_score = None, -1e9
        for pid, s in cand:
            i = idx_map.get(int(pid))
            if i is None: 
                continue
            rep = 0.0
            for p_sel, _ in sel:
                j = idx_map.get(int(p_sel))
                if j is None: 
                    continue
                rep = max(rep, float(doc_embs[i] @ doc_embs[j]))
            mmr_score = lambda_mult * s - (1 - lambda_mult) * rep
            if mmr_score > best_score:
                best_score = mmr_score
                best_pid = pid
        if best_pid is None:
            break
        # move best_pid to sel
        best_item = next((x for x in cand if x[0] == best_pid), None)
        if best_item:
            sel.append(best_item)
            picked.add(best_pid)
            cand = [x for x in cand if x[0] != best_pid]
        else:
            break
    return sel
