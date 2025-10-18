# src/azure_io.py
from typing import Iterable, List, Tuple, Optional
import os
from azure.storage.blob import BlobServiceClient

def _norm(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("/")

def list_blobs_with_prefix(conn_str: str, container: str, prefix: str) -> List[str]:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cont = bsc.get_container_client(container)
    prefix = _norm(prefix)
    # try with and without trailing slash
    pref_candidates = [prefix, f"{prefix}/"] if not prefix.endswith("/") else [prefix]
    names = []
    for p in pref_candidates:
        for b in cont.list_blobs(name_starts_with=p):
            names.append(b.name)
        if names:
            break
    return names

def list_all_blobs(conn_str: str, container: str, max_items: Optional[int] = None) -> List[str]:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cont = bsc.get_container_client(container)
    names = []
    for b in cont.list_blobs():
        names.append(b.name)
        if max_items and len(names) >= max_items:
            break
    return names

def download_blobs(conn_str: str, container: str, blob_names: Iterable[str], local_dir: str) -> List[str]:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cont = bsc.get_container_client(container)
    os.makedirs(local_dir, exist_ok=True)
    local_paths = []
    for blob in blob_names:
        blob = _norm(blob)
        fname = os.path.basename(blob)
        dst = os.path.join(local_dir, fname)
        with open(dst, "wb") as f:
            stream = cont.download_blob(blob).readall()
            f.write(stream)
        local_paths.append(dst)
    return local_paths

def download_prefix_flat(conn_str: str, container: str, prefix: str, local_dir: str) -> List[str]:
    prefix = _norm(prefix)
    blobs = list_blobs_with_prefix(conn_str, container, prefix)
    if not blobs:
        catalog = list_all_blobs(conn_str, container, max_items=50)
        raise FileNotFoundError(
            f"No blobs under prefix '{prefix}' in container '{container}'. "
            f"Preview (first 50): {catalog}"
        )
    return download_blobs(conn_str, container, blobs, local_dir)

def try_download_single_blob(conn_str: str, container: str, blob_name: str, local_path: str) -> Tuple[bool, Optional[str]]:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cont = bsc.get_container_client(container)
    blob_name = _norm(blob_name)
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(cont.download_blob(blob_name).readall())
        return True, None
    except Exception as e:
        return False, str(e)

def smart_download_single_blob(conn_str: str, container: str, blob_name: str, local_path: str) -> str:
    blob_name = _norm(blob_name)
    ok, err = try_download_single_blob(conn_str, container, blob_name, local_path)
    if ok:
        return local_path

    catalog = list_all_blobs(conn_str, container)
    base = os.path.basename(blob_name)

    exact = [b for b in catalog if os.path.basename(b) == base]
    if exact:
        fb = exact[0]
        ok2, err2 = try_download_single_blob(conn_str, container, fb, local_path)
        if ok2:
            return local_path
        raise FileNotFoundError(f"Tried basename match '{fb}' but failed: {err2}")

    subs = [b for b in catalog if base.lower() in b.lower()]
    if subs:
        fb = subs[0]
        ok3, err3 = try_download_single_blob(conn_str, container, fb, local_path)
        if ok3:
            return local_path
        raise FileNotFoundError(f"Found substring match '{fb}' but failed: {err3}")

    preview = catalog[:50]
    raise FileNotFoundError(
        f"Blob not found: container='{container}', blob_name='{blob_name}'. "
        f"Preview first 50: {preview}"
    )
