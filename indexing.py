"""
ChromaDB Indexing for RAG Pipeline
Indexes embeddings into ChromaDB for fast semantic search
"""

import json
import sys
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import os

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths_config import PathsConfig
from cli_common import add_standard_args, build_selection, configure_logging

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[WARN] chromadb not installed")
    print("       Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN] sentence-transformers not installed")
    print("       Install with: pip install sentence-transformers")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChromaDBIndexer:
    """Index embeddings into ChromaDB for semantic search"""

    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize ChromaDB client

        Args:
            db_path: Path to ChromaDB persistent storage
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Lazy-load embedder only when querying (saves memory during indexing)
        self.embedder = None
        self.model_name = "BAAI/bge-base-en-v1.5"
        print(f"[INIT] Embedder will be loaded when needed")

        # Initialize ChromaDB with persistent storage (new API)
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            print(f"[INIT] ChromaDB initialized at: {self.db_path}")
            logger.info(f"ChromaDB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            print(f"[ERROR] ChromaDB initialization failed: {e}")
            raise RuntimeError(f"Cannot initialize ChromaDB at {self.db_path}: {e}") from e

    def create_or_get_collection(self, collection_name: str) -> any:
        """
        Create or get a ChromaDB collection

        Args:
            collection_name: Name of collection (customer_id)

        Returns:
            ChromaDB collection object
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=collection_name)
            print(f"[OK] Using existing collection: {collection_name}")
            logger.info(f"Using existing collection: {collection_name}")
        except Exception as e:
            # Create new collection
            logger.info(f"Collection {collection_name} does not exist, creating new: {e}")
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"[CREATE] New collection: {collection_name}")
            logger.info(f"Created collection: {collection_name}")

        return collection

    def index_chunks(
        self,
        collection,
        chunks: List[Dict],
        customer_id: str,
        skip_duplicates: bool = True
    ) -> Dict:
        """
        Add embedded chunks to ChromaDB collection

        Args:
            collection: ChromaDB collection
            chunks: List of chunks with embeddings
            customer_id: Customer identifier
            skip_duplicates: If True, skip chunks that already exist in collection

        Returns:
            Indexing result summary
        """
        if not chunks:
            return {'status': 'no_chunks', 'indexed': 0}

        print(f"\n[INDEX] Adding {len(chunks)} chunks to collection...")

        EXPECTED_DIM = 768  # BGE-base embedding dimension
        BATCH_SIZE = 512    # Batch size for adding to ChromaDB

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        skipped = 0
        invalid_dims = 0

        # Build candidate IDs using content-based hash (stable across runs)
        candidate_ids = []
        for chunk in chunks:
            # Use chunk_hash (SHA256) for stable, collision-free IDs
            chunk_hash = chunk.get('metadata', {}).get('chunk_hash', '')
            if not chunk_hash:
                # Fallback: generate hash from text if missing
                text = chunk.get('text', '')
                chunk_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            candidate_ids.append(chunk_hash)

        # Batch-check for existing IDs if skipping duplicates
        existing_ids = set()
        if skip_duplicates:
            # Check existence in batches (avoid loading entire collection)
            for i in range(0, len(candidate_ids), BATCH_SIZE):
                batch_ids = candidate_ids[i:i+BATCH_SIZE]
                try:
                    existing = collection.get(ids=batch_ids)
                    if existing and existing['ids']:
                        existing_ids.update(existing['ids'])
                except Exception as e:
                    logger.warning(f"Could not check duplicates for batch {i}: {e}")

            if existing_ids:
                print(f"[INFO] Found {len(existing_ids)} existing chunks in collection")

        # Prepare chunks for indexing
        for i, chunk in enumerate(chunks):
            chunk_id = candidate_ids[i]

            # Skip if already indexed
            if skip_duplicates and chunk_id in existing_ids:
                skipped += 1
                continue

            # Extract and validate embedding
            embedding = chunk.get('embedding', [])
            if not embedding:
                logger.warning(f"No embedding found for chunk {chunk_id[:16]}...")
                continue

            # CRITICAL: Validate embedding dimension
            if len(embedding) != EXPECTED_DIM:
                logger.warning(f"Skipping chunk {chunk_id[:16]}... - invalid dimension: {len(embedding)} (expected {EXPECTED_DIM})")
                invalid_dims += 1
                continue

            # Document text for retrieval
            text = chunk.get('text', '')[:5000]  # Truncate very long texts

            # Metadata for filtering (keep numerics as int for ChromaDB filters)
            metadata = {
                'customer_id': customer_id,
                'doc_name': chunk.get('metadata', {}).get('doc_name', 'unknown'),
                'page_num': int(chunk.get('metadata', {}).get('page_num', 0)),
                'chunk_hash': chunk_id,
                'token_count': int(chunk.get('metadata', {}).get('token_count', 0)),
            }

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append(metadata)

        # Add to ChromaDB in batches
        indexed = 0
        try:
            for i in range(0, len(ids), BATCH_SIZE):
                batch_ids = ids[i:i+BATCH_SIZE]
                batch_embeddings = embeddings[i:i+BATCH_SIZE]
                batch_documents = documents[i:i+BATCH_SIZE]
                batch_metadatas = metadatas[i:i+BATCH_SIZE]

                try:
                    collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    indexed += len(batch_ids)
                except Exception as e:
                    logger.error(f"Failed to add batch starting at {i}: {e}")
                    # Continue with next batch instead of failing completely

            print(f"[OK] Indexed {indexed} chunks")
            if skipped > 0:
                print(f"[INFO] Skipped {skipped} duplicate chunks")
            if invalid_dims > 0:
                print(f"[WARN] Skipped {invalid_dims} chunks with invalid dimensions")
            logger.info(f"Successfully indexed {indexed} chunks")

            return {
                'status': 'success',
                'indexed': indexed,
                'skipped': skipped,
                'invalid_dims': invalid_dims,
                'collection_name': collection.name,
                'total_in_collection': collection.count()
            }

        except Exception as e:
            logger.error(f"Error indexing chunks: {e}", exc_info=True)
            print(f"[ERROR] {e}")
            return {'status': 'error', 'indexed': indexed, 'error': str(e)}

    def query_collection(
        self,
        collection,
        query_text: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Query the collection with semantic search

        Args:
            collection: ChromaDB collection
            query_text: Query text
            n_results: Number of results to return

        Returns:
            List of search results
        """
        try:
            # Lazy-load embedder on first query
            if self.embedder is None:
                print(f"[INIT] Loading embedding model: {self.model_name}")
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

                self.embedder = SentenceTransformer(self.model_name, device=device)
                print(f"[OK] Embedding model loaded on device: {device}")
                logger.info(f"Embedding model loaded on {device}")

            # Generate query embedding using same BGE model used for indexing
            query_embedding = self.embedder.encode(
                query_text,
                convert_to_tensor=False,
                normalize_embeddings=True
            ).tolist()

            # Query using pre-generated embeddings (matching indexing model)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i, chunk_id in enumerate(results['ids'][0]):
                    formatted_results.append({
                        'chunk_id': chunk_id,
                        'distance': results['distances'][0][i],
                        'text': results['documents'][0][i] if results['documents'] else '',
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            print(f"[ERROR] {e}")
            return []

    def get_collection_stats(self, collection) -> Dict:
        """Get statistics about a collection"""
        try:
            count = collection.count()
            return {
                'name': collection.name,
                'total_chunks': count,
                'status': 'ok'
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'status': 'error', 'error': str(e)}


def index_single_file(indexer, embedded_file: Path, customer_id: str) -> Dict:
    """
    Index a single embedded file

    Args:
        indexer: ChromaDBIndexer instance
        embedded_file: Path to content_embedded.json
        customer_id: Customer identifier

    Returns:
        Result dict
    """
    if not embedded_file.exists():
        print(f"[ERROR] File not found: {embedded_file}")
        return {'status': 'error', 'error': 'File not found'}

    print(f"\n[SINGLE FILE] Indexing: {embedded_file}")
    print("=" * 70)

    try:
        # Load embedded data
        with open(embedded_file, 'r', encoding='utf-8') as f:
            embedded_data = json.load(f)

        chunks = embedded_data.get('chunks', [])
        if not chunks:
            print("[WARN] No chunks found in file")
            return {'status': 'no_chunks', 'indexed': 0}

        # Get or create collection
        collection = indexer.create_or_get_collection(customer_id)

        # Index chunks
        result = indexer.index_chunks(collection, chunks, customer_id)

        if result['status'] == 'success':
            stats = indexer.get_collection_stats(collection)
            print(f"\n[SUMMARY]")
            print(f"  Customer: {customer_id}")
            print(f"  Chunks indexed: {result['indexed']}")
            print(f"  Total in collection: {stats['total_chunks']}")

        return result

    except Exception as e:
        logger.error(f"Failed to index file: {e}", exc_info=True)
        print(f"[ERROR] {e}")
        return {'status': 'error', 'error': str(e)}


def index_all_customers(indexer, customers_dir: Optional[Path] = None) -> Dict:
    """
    Index all customers' embedded documents

    Args:
        indexer: ChromaDBIndexer instance

    Returns:
        Summary of indexing results
    """
    print("\n" + "=" * 70)
    print("CHROMADB INDEXING PIPELINE")
    print("=" * 70)

    customers_dir = customers_dir or PathsConfig.CUSTOMERS_DIR

    if not customers_dir.exists():
        logger.error(f"Customers directory not found: {customers_dir}")
        print(f"[ERROR] {customers_dir} not found")
        return {}

    # Discover all embedded files
    all_files = list(customers_dir.glob("*/*/content_embedded.json"))

    if not all_files:
        print("[WARN] No embedded files found")
        return {}

    print(f"\n[DISCOVER] Found {len(all_files)} embedded files")

    results = {}
    total_indexed = 0

    for embedded_file in all_files:
        # Extract customer_id from path
        customer_id = embedded_file.parent.parent.name

        try:
            result = index_single_file(indexer, embedded_file, customer_id)
            results[customer_id] = result

            if result['status'] == 'success':
                total_indexed += result['indexed']

        except Exception as e:
            logger.error(f"Failed to index {customer_id}: {e}")
            results[customer_id] = {'status': 'error', 'error': str(e)}

    # Print summary
    print("\n" + "=" * 70)
    print("INDEXING SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results.values() if r['status'] == 'success')

    for customer_id, result in sorted(results.items()):
        status_icon = "[OK]" if result['status'] == 'success' else "[WARN]"
        indexed_count = result.get('indexed', 0)
        print(f"{status_icon} {customer_id:25} {indexed_count} chunks indexed")

    print(f"\nTotal chunks indexed: {total_indexed}")
    print(f"Successful: {successful}/{len(results)}")
    print("=" * 70)

    logger.info(f"Indexing complete: {total_indexed} chunks, {successful} successful")

    return results


def main():
    """Main entry point"""
    if DOTENV_AVAILABLE:
        load_dotenv()

    if not CHROMADB_AVAILABLE:
        print("[ERROR] chromadb not installed")
        print("Install with: pip install chromadb")
        return 1

    # Configure ChromaDB temp directory to avoid /tmp full errors
    PathsConfig.CHROMA_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(PathsConfig.CHROMA_TEMP_DIR)
    os.environ['TEMP'] = str(PathsConfig.CHROMA_TEMP_DIR)
    os.environ['TMP'] = str(PathsConfig.CHROMA_TEMP_DIR)
    print(f"[CONFIG] ChromaDB temp directory: {PathsConfig.CHROMA_TEMP_DIR}")

    # Use centralized paths
    chroma_db_path = str(PathsConfig.CHROMA_DB_PATH)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Index embeddings into ChromaDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python indexing.py --all                                # Index all customers
  python indexing.py --customer stuart_dean               # Index single customer
  python indexing.py --file /path/to/content_embedded.json  # Index single file
  python indexing.py --query "What is 3CX?" --collection stuart_dean  # Query collection
        """
    )
    add_standard_args(parser, include_file=True, include_customer_id=False, include_server_root=True)
    parser.add_argument('--query', type=str, help='Query text for semantic search')
    parser.add_argument('--collection', type=str, help='Collection name for query')
    parser.add_argument('--db-path', type=str, default=chroma_db_path, help='ChromaDB path')

    args = parser.parse_args()
    selection = build_selection(args)

    try:
        configure_logging("indexing", server_root=selection.server_root)

        # Initialize indexer
        indexer = ChromaDBIndexer(args.db_path)

        # If --query provided, just query
        if args.query:
            if not args.collection:
                print("[ERROR] --collection required for queries")
                return 1

            collection = indexer.create_or_get_collection(args.collection)
            results = indexer.query_collection(collection, args.query, n_results=5)

            print(f"\n[QUERY] '{args.query}'")
            print("=" * 70)
            print(f"[RESULTS] Found {len(results)} matches\n")

            for i, result in enumerate(results, 1):
                distance = result['distance']
                relevance = max(0, 100 - (distance * 100))  # Convert to relevance score
                print(f"Match {i} (Relevance: {relevance:.1f}%)")
                print(f"  Chunk: {result['metadata'].get('chunk_id', 'unknown')}")
                print(f"  Page: {result['metadata'].get('page_num', 'unknown')}")
                print(f"  Text: {result['text'][:200]}...")
                print()

            return 0

        # If --file provided, index single file
        if selection.file:
            customer_id = selection.file.parent.parent.name
            result = index_single_file(indexer, selection.file, customer_id)
            return 0 if result['status'] in ['success', 'no_chunks'] else 1

        customer_filter = selection.customer_filter()
        if customer_filter:
            customers_dir = selection.server_root / "customers"
            had_error = False
            for customer_id in customer_filter:
                customer_dir = customers_dir / customer_id
                if not customer_dir.exists():
                    print(f"[ERROR] Customer directory not found: {customer_dir}")
                    had_error = True
                    continue

                embedded_files = list(customer_dir.glob("*/content_embedded.json"))
                if not embedded_files:
                    print(f"[WARN] No embedded files found for customer: {customer_id}")
                    continue

                print(f"\n[CUSTOMER] Indexing {len(embedded_files)} documents for {customer_id}")
                for embedded_file in embedded_files:
                    index_single_file(indexer, embedded_file, customer_id)

            return 1 if had_error else 0

        # Otherwise index all customers (--all or default)
        results = index_all_customers(indexer, customers_dir=selection.server_root / "customers")
        success = all(r['status'] in ['success', 'no_chunks'] for r in results.values())

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Indexing pipeline failed: {e}", exc_info=True)
        print(f"[ERROR] {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
