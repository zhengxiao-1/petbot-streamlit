# src/models.py
import os
import numpy as np
from typing import Optional, Tuple, Any

def load_ner_pipeline(local_ner_dir: str):
    """
    Load NER from a flat local folder (pure HF files). No HF Hub calls.
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    tok = AutoTokenizer.from_pretrained(local_ner_dir, local_files_only=True)
    mdl = AutoModelForTokenClassification.from_pretrained(local_ner_dir, local_files_only=True)
    pipe = pipeline(
        "token-classification",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="simple",
        device=-1  # set to 0 if you want GPU
    )
    return pipe

def _build_sentence_transformer_from_hf_folder(model_dir: str):
    """
    Build a SentenceTransformer from plain HF files in a LOCAL folder:
    config.json + tokenizer + weights (pytorch_model.bin or model.safetensors).
    Uses mean pooling. No HF Hub calls.
    """
    from transformers import AutoConfig
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Transformer as STTransformer, Pooling

    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    if hidden is None:
        raise RuntimeError("Cannot infer hidden size from config.json (no hidden_size/d_model).")

    word_emb = STTransformer(model_dir)  # loads the local encoder
    pooling = Pooling(
        word_emb.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    return SentenceTransformer(modules=[word_emb, pooling])

def load_mr_model(local_mr_dir: str) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Returns: (sentence_transformer_model, doc_ids, doc_vecs)
    Always builds ST from local HF files to avoid any repo-id validation.
    Requires doc_ids.npy and doc_embeddings.npy in the same folder.
    """
    ids_path = os.path.join(local_mr_dir, "doc_ids.npy")
    emb_path = os.path.join(local_mr_dir, "doc_embeddings.npy")
    if not (os.path.exists(ids_path) and os.path.exists(emb_path)):
        raise FileNotFoundError(
            f"Missing doc_ids.npy or doc_embeddings.npy in {local_mr_dir}."
        )

    # Force local manual build (ignore modules.json to prevent HF Hub path handling)
    student = _build_sentence_transformer_from_hf_folder(local_mr_dir)

    doc_ids = np.load(ids_path)
    doc_vecs = np.load(emb_path).astype("float32")
    return student, doc_ids, doc_vecs

def load_faiss_index(local_mr_dir: str, dim: int) -> Optional[Any]:
    """
    Load FAISS index if present (pets_hnsw.index / pets_flat.index / faiss.index).
    Returns None if faiss is not installed or no index found.
    """
    try:
        import faiss  # noqa
    except Exception:
        return None

    for name in ["pets_hnsw.index", "pets_flat.index", "faiss.index"]:
        p = os.path.join(local_mr_dir, name)
        if os.path.exists(p):
            idx = faiss.read_index(p)
            if hasattr(idx, "d") and idx.d != dim:
                # dimension mismatch, skip
                continue
            return idx
    return None
