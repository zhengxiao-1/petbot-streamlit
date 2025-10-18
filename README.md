# PetBot 🐾 — Streamlit Search (NER + Hybrid Ranking)

PetBot is a **Streamlit** app that helps users search Malaysian pet-adoption listings using natural language.  
It combines a **lightweight NER** pipeline for facet extraction with a **hybrid retriever** (BM25 + embeddings)
and a **facet-aware re-ranker** (puts extra weight on **breed** and **age**) for relevant results.

> **Default mode:** *Best-model-only* — Hybrid (**0.1 × BM25 + 0.9 × Embeddings**) with **no MMR**, matching your evaluation choice.

---

## 📦 What’s in this repo

```
.
├─ streamlit_app.py                # Main Streamlit app (calls set_page_config FIRST)
├─ src/
│  ├─ azure_io.py                  # Azure Blob helpers (flat folder downloads, smart single-blob fetch)
│  ├─ config.py                    # Reads Streamlit secrets and resolves local cache paths
│  ├─ models.py                    # Loads NER (HF local folder) + sentence embedding model + FAISS index (optional)
│  ├─ retrieval.py                 # BM25, embedding search, MMR (unused in default), facet parsing & relaxation
│  └─ ui.py                        # Sidebar controls (locked to best model; hides Method/MMR)
├─ requirements.txt
└─ README.md
```

### Key implementation details (from this codebase)
- **Secrets-driven config** (`src/config.py`): expects Streamlit **secrets** keys.
- **Azure downloads** (`src/azure_io.py`): download models/CSV from Azure Blob Storage.
- **Models** (`src/models.py`): loads NER + embeddings + FAISS index.
- **Retrieval** (`src/retrieval.py`): BM25, embeddings, relaxation, MMR (unused by default).
- **UI** (`src/ui.py`): locked to hybrid model; exposes only Top-K & Strict Mode controls.

---

## 🐍 Requirements

See `requirements.txt` (pinned versions):
- `streamlit==1.37.0`, `pandas==2.2.3`, `numpy==2.1.3`
- `transformers==4.44.0`, `sentence-transformers==3.0.1`, `torch>=2.1.0`
- `faiss-cpu==1.11.0`, `azure-storage-blob==12.19.1`

> If FAISS isn’t available, the app **falls back** to a fast brute-force similarity search.

---

## 🔐 Configuration (Streamlit secrets)

Create `.streamlit/secrets.toml`:

```toml
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=..."
ML_ARTIFACTS_CONTAINER = "ml"
PETS_CONTAINER         = "pets"
NER_PREFIX = "ner_model_flat"
MR_PREFIX  = "mr_model_flat"
PETS_CSV_BLOB = "all_pet_details_clean.csv"
```

Local cache paths (from `config.py`):
- NER → `artifacts/ner`
- MR  → `artifacts/mr`
- Pets CSV → `artifacts/pets.csv`

---

## ▶️ Running locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🧠 How it works

1. **NER + rules → facets (animal, breed, gender, color, state)**
2. **Age** detected (keywords or numeric)
3. **Filter + relax** (colors → state → gender; breed & age hard)
4. **Hybrid retrieval** (BM25+Embeddings = 0.1:0.9)
5. **Facet re-rank** (breed>age>gender>state>colors)
6. **Display** cards (photos + info) or table

---

## 📑 CSV schema

| Column | Type | Purpose |
|--------|------|----------|
| pet_id | int | Unique ID |
| name, url | str | Display info |
| animal, breed, gender, state | str | Core facets |
| age_months | int | Age in months |
| description_clean, doc | str | Searchable text |
| color, colors_canonical | list/str | Normalized colors |
| photo_links, video_links | list | Media URLs |
| size, fur_length, condition | str | Extra attributes |

---

## ⚙️ Parameters you can tune

- `HYBRID_W = {"lex": 0.1, "emb": 0.9}` in `streamlit_app.py`
- `RELAX_ORDER = ["colors_any", "state", "gender"]`
- `MIN_CAND_FLOOR_BASE = 300`

---

## 🚀 Deploy

**Streamlit Cloud**
1. Push to GitHub.
2. Create new app → main file: `streamlit_app.py`.
3. Add secrets via web UI.

**Docker**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🛠️ Troubleshooting

- **`set_page_config()` error** → must be called first and once.  
- **Azure path errors** → check prefixes & container names.  
- **Missing embeddings** → ensure `doc_ids.npy` & `doc_embeddings.npy` exist.  
- **FAISS missing** → fallback to brute-force search.  
- **Photos missing** → check `photo_links` format; Markdown fallback used.

---

## 📝 License

MIT (or your choice).
