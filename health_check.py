#!/usr/bin/env python3
"""
RAG Health Check
Validates Chroma collections and embedded files without modifying data.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths_config import PathsConfig
from cli_common import add_standard_args, build_selection, configure_logging

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _normalize_customer_id(customer_id: str) -> str:
    """Normalize customer ID to match collection naming."""
    return customer_id.lower().replace(" ", "_").replace("-", "_")


def _discover_embedded_files(customer_dir: Path) -> List[Path]:
    """Find all embedded files for a customer."""
    return list(customer_dir.glob("*/content_embedded.json"))


def _load_embedding_metadata(file_path: Path) -> Dict:
    """Load metadata from a content_embedded.json file."""
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            'embedding_model': data.get('embedding_model'),
            'embedding_dim': data.get('embedding_dim'),
            'total_chunks': data.get('total_chunks', 0),
        }
    except Exception as exc:
        logger.error(f"Failed to load {file_path}: {exc}")
        return {}


def _check_customer_files(customer_id: str, customer_dir: Path) -> Dict:
    """Check embedded files for a customer."""
    embedded_files = _discover_embedded_files(customer_dir)
    if not embedded_files:
        return {'status': 'warning', 'message': 'no embedded files', 'files': 0}

    models = set()
    dims = set()
    total_chunks = 0

    for file_path in embedded_files:
        meta = _load_embedding_metadata(file_path)
        if meta.get('embedding_model'):
            models.add(meta['embedding_model'])
        if meta.get('embedding_dim'):
            dims.add(meta['embedding_dim'])
        total_chunks += int(meta.get('total_chunks', 0))

    if len(models) > 1 or len(dims) > 1:
        return {
            'status': 'warning',
            'message': 'mixed embedding metadata',
            'files': len(embedded_files),
            'models': sorted(models),
            'dims': sorted(dims),
            'total_chunks': total_chunks
        }

    return {
        'status': 'ok',
        'files': len(embedded_files),
        'models': sorted(models),
        'dims': sorted(dims),
        'total_chunks': total_chunks
    }


def _check_chroma_collection(client, customer_id: str) -> Dict:
    """Check Chroma collection presence and count."""
    collection_name = _normalize_customer_id(customer_id)
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        status = 'ok' if count > 0 else 'warning'
        return {'status': status, 'collection': collection_name, 'count': count}
    except Exception as exc:
        logger.warning(f"Collection not found for {customer_id}: {exc}")
        return {'status': 'error', 'collection': collection_name, 'count': 0}


def _get_customers(server_root: Path, customer_filter: Optional[List[str]]) -> List[str]:
    customers_dir = server_root / "customers"
    if not customers_dir.exists():
        logger.error(f"Customers directory not found: {customers_dir}")
        return []

    customers = [d.name for d in customers_dir.iterdir() if d.is_dir()]
    if customer_filter:
        customers = [c for c in customers if c in customer_filter]
    return sorted(customers)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='RAG Health Check (read-only)'
    )
    add_standard_args(parser, include_file=True, include_customer_id=False, include_server_root=True)
    parser.add_argument('--db-path', type=str, help='ChromaDB path override')
    args = parser.parse_args()
    selection = build_selection(args)

    configure_logging("health_check", server_root=selection.server_root)

    if selection.file:
        file_path = Path(selection.file)
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return 1
        meta = _load_embedding_metadata(file_path)
        status = 'OK' if meta else 'ERROR'
        print(f"[{status}] {file_path}")
        if meta:
            print(f"  Embedding model: {meta.get('embedding_model')}")
            print(f"  Embedding dim:   {meta.get('embedding_dim')}")
            print(f"  Total chunks:    {meta.get('total_chunks')}")
        return 0 if meta else 1

    db_path = Path(args.db_path) if args.db_path else PathsConfig.CHROMA_DB_PATH
    client = None
    if CHROMA_AVAILABLE:
        try:
            client = chromadb.PersistentClient(path=str(db_path))
        except Exception as exc:
            logger.error(f"Failed to initialize ChromaDB at {db_path}: {exc}")
            client = None
    else:
        logger.error("chromadb not installed - skipping collection checks")

    customers = _get_customers(selection.server_root, selection.customer_filter())
    if not customers:
        print("[WARN] No customers found to check")
        return 1

    print("\n" + "=" * 70)
    print("RAG HEALTH CHECK")
    print("=" * 70)
    print(f"Server root: {selection.server_root}")
    print(f"Chroma DB:   {db_path}")
    print("=" * 70)

    ok_count = 0
    warn_count = 0
    err_count = 0

    for customer_id in customers:
        customer_dir = selection.server_root / "customers" / customer_id
        file_check = _check_customer_files(customer_id, customer_dir)
        collection_check = {'status': 'skipped'}
        if client:
            collection_check = _check_chroma_collection(client, customer_id)

        status = 'OK'
        if file_check['status'] == 'warning' or collection_check.get('status') == 'warning':
            status = 'WARN'
            warn_count += 1
        elif file_check['status'] == 'ok' and collection_check.get('status') in ('ok', 'skipped'):
            ok_count += 1
        else:
            status = 'ERROR'
            err_count += 1

        print(f"[{status}] {customer_id}")
        print(f"  Embedded files: {file_check.get('files', 0)}")
        if file_check.get('models'):
            print(f"  Models:        {', '.join(file_check['models'])}")
        if file_check.get('dims'):
            print(f"  Dimensions:    {', '.join(str(d) for d in file_check['dims'])}")
        print(f"  Total chunks:  {file_check.get('total_chunks', 0)}")
        if collection_check.get('status') != 'skipped':
            print(f"  Collection:    {collection_check.get('collection')}")
            print(f"  Chroma count:  {collection_check.get('count')}")
        if file_check.get('message'):
            print(f"  Note:          {file_check['message']}")

    print("\n" + "-" * 70)
    print(f"OK: {ok_count}  WARN: {warn_count}  ERROR: {err_count}")
    print("=" * 70)

    return 1 if err_count > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
