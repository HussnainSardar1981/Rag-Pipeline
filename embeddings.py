"""
Production-Grade Batch Embedding Generator for RAG Pipeline
Combines best of embeddings.py + embedder.py
Uses all-mpnet-base-v2 for best accuracy (768 dims)
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import os

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths_config import PathsConfig
from cli_common import add_standard_args, build_selection, configure_logging

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN] sentence-transformers not installed.")
    print("       Install with: pip install sentence-transformers")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


class BatchEmbedder:
    """Generate embeddings using sentence-transformers with best accuracy"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 16,
        device: Optional[str] = None
    ):
        """
        Initialize embedder with best-in-class models

        Args:
            model_name: HuggingFace model name
                RECOMMENDED FOR BEST ACCURACY:
                - 'BAAI/bge-base-en-v1.5': BGE model, highest retrieval accuracy (768 dim)
                - 'BAAI/bge-m3': BGE-M3, versatile and fast (1024 dim)
                - 'nomic-ai/nomic-embed-text-v1': Highest top-5 accuracy (768 dim)
                - 'intfloat/e5-base-v2': E5 family, strong all-rounder (768 dim)

                BALANCED OPTIONS:
                - 'all-mpnet-base-v2': Solid accuracy (768 dim)

                SPEED FOCUSED:
                - 'all-MiniLM-L6-v2': Fast, low memory (384 dim)
            batch_size: Batch size for GPU processing (8-16 for larger models, 16-32 for smaller)
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto-detect
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if not TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            raise ImportError("Install: pip install sentence-transformers")

        # Lazy-load model only when actually needed
        self.model = None
        print(f"[INIT] Embedder ready (model will load when needed)")
        print(f"[INIT] Device: {self.device}")
        print(f"[INIT] Batch size: {batch_size}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors (normalized for similarity search)
        """
        if not texts:
            return []

        # Lazy-load model on first use
        if self.model is None:
            print(f"[LOAD] Loading embedding model: {self.model_name}")
            logger.info(f"Loading model {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"[OK] Model loaded. Embedding dimension: {embedding_dim}")
                logger.info(f"Model {self.model_name} loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}", exc_info=True)
                raise

        print(f"[EMBED] Processing {len(texts)} texts in batches of {self.batch_size}...")
        logger.info(f"Embedding {len(texts)} texts")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        # Convert numpy arrays to lists
        embeddings_list = [emb.tolist() for emb in embeddings]

        print(f"[OK] Generated {len(embeddings_list)} embeddings")
        logger.info(f"Successfully generated {len(embeddings_list)} embeddings")

        return embeddings_list

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed all chunks with metadata preservation

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' keys

        Returns:
            List of chunks with added 'embedding' key
        """
        if not chunks:
            return []

        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        embeddings = self.embed_texts(texts)

        # Attach embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding
            chunks_with_embeddings.append(chunk_copy)

        return chunks_with_embeddings


class RAGEmbedder:
    """Coordinate embedding for all customer documents"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 16,
        device: Optional[str] = None,
        server_root: Optional[Path] = None
    ):
        """
        Initialize RAG embedder with production-grade embedding model

        Default: BAAI/bge-base-en-v1.5 (BGE - highest retrieval accuracy for technical docs)
        """
        self.embedder = BatchEmbedder(model_name, batch_size, device)
        self.server_root = server_root or PathsConfig.RAG_DIR
        self.model_name = model_name

    def embed_customer_chunks(
        self,
        customer_id: str,
        chunks: List[Dict]
    ) -> Dict:
        """
        Embed all chunks for a customer

        Args:
            customer_id: Customer identifier
            chunks: List of chunks from chunker

        Returns:
            {
                'customer_id': '...',
                'total_chunks': N,
                'embedding_model': 'all-mpnet-base-v2',
                'embedding_dim': 768,
                'chunks_with_embeddings': [...]
            }
        """
        print(f"\n[EMBED] Processing chunks for customer: {customer_id}")
        print("=" * 70)

        if not chunks:
            logger.warning(f"No chunks to embed for {customer_id}")
            return {
                'customer_id': customer_id,
                'total_chunks': 0,
                'status': 'no_chunks'
            }

        # Embed all chunks
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        embedding_dim = len(chunks_with_embeddings[0]['embedding']) if chunks_with_embeddings else 0

        print("\n" + "=" * 70)
        print(f"[SUMMARY] {customer_id}")
        print(f"  Total chunks: {len(chunks_with_embeddings)}")
        print(f"  Embedding model: {self.model_name}")
        print(f"  Embedding dimension: {embedding_dim}")
        if chunks_with_embeddings:
            print(f"  Sample chunk size: {len(chunks_with_embeddings[0]['text'])} chars")

        logger.info(f"Embedded {len(chunks_with_embeddings)} chunks for {customer_id}")

        return {
            'customer_id': customer_id,
            'total_chunks': len(chunks_with_embeddings),
            'embedding_model': self.model_name,
            'embedding_dim': embedding_dim,
            'chunks_with_embeddings': chunks_with_embeddings,
            'created_at': datetime.now().isoformat()
        }

    def _save_embeddings(self, embedding_result: Dict, output_path: Path) -> None:
        """
        Save already-embedded chunks to file (atomic write)

        Args:
            embedding_result: Result from embed_customer_chunks containing chunks_with_embeddings
            output_path: Where to save embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[SAVE] Saving embeddings to: {output_path}")

        # Prepare for JSON serialization
        chunks_for_save = []
        for chunk in embedding_result.get('chunks_with_embeddings', []):
            chunks_for_save.append({
                'metadata': chunk['metadata'],
                'text': chunk['text'],
                'embedding': chunk['embedding']
            })

        save_data = {
            'customer_id': embedding_result['customer_id'],
            'total_chunks': embedding_result['total_chunks'],
            'embedding_model': embedding_result['embedding_model'],
            'embedding_dim': embedding_result['embedding_dim'],
            'device': self.embedder.device,
            'created_at': datetime.now().isoformat(),
            'chunks': chunks_for_save
        }

        # Atomic write to prevent corruption
        tmp_path = output_path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_path)

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"[OK] Saved {embedding_result['total_chunks']} embeddings ({file_size_mb:.1f} MB)")
        logger.info(f"Saved embeddings to {output_path} ({file_size_mb:.1f} MB)")

    def embed_and_save(
        self,
        customer_id: str,
        chunks: List[Dict],
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Embed chunks and save to file

        Args:
            customer_id: Customer identifier
            chunks: List of chunks
            output_path: Where to save embeddings (if None, uses same dir as input)

        Returns:
            Result dict with embedding info
        """
        # Embed
        result = self.embed_customer_chunks(customer_id, chunks)

        if result.get('status') == 'no_chunks':
            return result

        # Determine output path - use same dir as chunked file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[SAVE] Saving embeddings to: {output_path}")

        # Prepare for JSON serialization
        chunks_for_save = []
        for chunk in result.get('chunks_with_embeddings', []):
            chunks_for_save.append({
                'metadata': chunk['metadata'],
                'text': chunk['text'],
                'embedding': chunk['embedding']
            })

        save_data = {
            'customer_id': customer_id,
            'total_chunks': result['total_chunks'],
            'embedding_model': result['embedding_model'],
            'embedding_dim': result['embedding_dim'],
            'device': self.embedder.device,
            'created_at': datetime.now().isoformat(),
            'chunks': chunks_for_save
        }

        # Atomic write to prevent corruption
        tmp_path = output_path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_path)

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"[OK] Saved {result['total_chunks']} embeddings ({file_size_mb:.1f} MB)")
        logger.info(f"Saved embeddings to {output_path} ({file_size_mb:.1f} MB)")

        return result

    def embed_all_customers(
        self,
        customer_filter: Optional[List[str]] = None
    ) -> Dict:
        """
        Embed all customers' documents

        Args:
            customer_filter: List of specific customers to process

        Returns:
            Summary of embedding results
        """
        print("\n" + "=" * 70)
        print("VECTOR EMBEDDING PIPELINE (Production)")
        print(f"Model: {self.model_name}")
        print("=" * 70)

        customers_dir = self.server_root / "customers"

        if not customers_dir.exists():
            logger.error(f"Customers directory not found: {customers_dir}")
            return {}

        # Discover customers
        all_customers = []
        for customer_folder in sorted(customers_dir.glob("*")):
            if customer_folder.is_dir():
                customer_id = customer_folder.name
                all_customers.append(customer_id)

        print(f"\n[DISCOVER] Found {len(all_customers)} customers")
        for customer_id in all_customers:
            print(f"  - {customer_id}")

        if customer_filter:
            all_customers = [c for c in all_customers if c in customer_filter]
            print(f"\n[FILTER] Processing {len(all_customers)} selected customers")

        results = {}
        total_chunks_embedded = 0

        for customer_id in all_customers:
            try:
                customer_dir = customers_dir / customer_id

                # Look for content_cleaned.json files (cleaned content from cleaner stage)
                # Priority 1: content_chunked.json (if chunking stage exists)
                # Priority 2: content_cleaned.json (from cleaner stage)
                # Fallback: content.json (from extraction stage)
                cleaned_files = list(customer_dir.glob("*/content_cleaned.json"))
                chunked_files = list(customer_dir.glob("*/content_chunked.json"))

                # Use chunked if available, otherwise use cleaned
                files_to_process = chunked_files if chunked_files else cleaned_files

                if not files_to_process:
                    print(f"\n[SKIP] {customer_id}: No cleaned/chunked documents found")
                    results[customer_id] = {
                        'customer_id': customer_id,
                        'status': 'no_chunked_documents',
                        'chunks_embedded': 0
                    }
                    continue

                print(f"\n[PROCESS] {customer_id}")
                customer_chunks_embedded = 0
                skipped_count = 0

                for chunked_file in files_to_process:
                    doc_name = chunked_file.parent.name
                    embedded_file = chunked_file.parent / "content_embedded.json"

                    # Skip if already embedded AND source file hasn't changed
                    if embedded_file.exists():
                        source_mtime = chunked_file.stat().st_mtime
                        embedded_mtime = embedded_file.stat().st_mtime

                        if embedded_mtime >= source_mtime:
                            print(f"  [SKIP] {doc_name} (already embedded)")
                            skipped_count += 1
                            continue

                    print(f"  [EMBED] {doc_name}")

                    with open(chunked_file, 'r', encoding='utf-8') as f:
                        chunked_data = json.load(f)

                    # Handle different file formats
                    # Priority 1: content_chunked.json (has 'chunks' key)
                    # Priority 2: content_cleaned.json (has 'pages' key - convert to chunks)
                    if 'chunks' in chunked_data:
                        chunks = chunked_data.get('chunks', [])
                    elif 'pages' in chunked_data:
                        # Convert pages format to chunks format
                        chunks = []
                        pages = chunked_data.get('pages', [])

                        for page_num, page in enumerate(pages, 1):
                            text = page.get('text', '')
                            if text.strip():
                                chunks.append({
                                    'text': text,
                                    'metadata': {
                                        'doc_name': doc_name,
                                        'page_num': page_num,
                                        'chunk_id': f"{doc_name}_page_{page_num}",
                                        'token_count': len(text.split())
                                    }
                                })
                    else:
                        chunks = []

                    # Embed
                    result = self.embed_customer_chunks(customer_id, chunks)

                    if result.get('status') != 'no_chunks':
                        # Save to content_embedded.json in same directory
                        embedded_file = chunked_file.parent / "content_embedded.json"
                        # Save already-embedded chunks (don't re-embed)
                        self._save_embeddings(result, embedded_file)

                        chunks_embedded = result['total_chunks']
                        customer_chunks_embedded += chunks_embedded
                        total_chunks_embedded += chunks_embedded

                results[customer_id] = {
                    'customer_id': customer_id,
                    'status': 'success',
                    'chunks_embedded': customer_chunks_embedded
                }

            except Exception as e:
                logger.error(f"Failed to embed {customer_id}: {e}", exc_info=True)
                print(f"  [ERROR] {e}")
                results[customer_id] = {
                    'customer_id': customer_id,
                    'status': 'error',
                    'error': str(e),
                    'chunks_embedded': 0
                }

        # Print summary
        print("\n" + "=" * 70)
        print("EMBEDDING SUMMARY")
        print("=" * 70)

        successful = sum(1 for r in results.values() if r['status'] == 'success')

        for customer_id, result in sorted(results.items()):
            status_icon = "[OK]" if result['status'] == 'success' else "[WARN]"
            print(f"{status_icon} {customer_id:25} {result['chunks_embedded']} chunks embedded")

        print("\n" + "-" * 70)
        print(f"Total chunks embedded: {total_chunks_embedded}")
        print(f"Successful customers: {successful}/{len(results)}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.embedder.device}")
        print("=" * 70)

        logger.info(f"Embedding complete: {total_chunks_embedded} chunks, {successful} successful")

        return results


def main():
    """Main entry point"""
    if DOTENV_AVAILABLE:
        load_dotenv()

    # Check if sentence-transformers is available
    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] sentence-transformers not installed")
        print("Install: pip install sentence-transformers")
        return 1

    # Use centralized paths
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate embeddings from chunked documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python embeddings.py --all                                    # Embed all customers
  python embeddings.py --customer stuart_dean                   # Embed single customer
  python embeddings.py --file /path/to/content_chunked.json     # Embed single file
        """
    )
    add_standard_args(parser, include_file=True, include_customer_id=False, include_server_root=True)
    args = parser.parse_args()
    selection = build_selection(args)

    try:
        configure_logging("embeddings", server_root=selection.server_root)

        # Initialize embedder with highest-accuracy model
        # BAAI/bge-base-en-v1.5 is recommended for technical documentation RAG
        embedder = RAGEmbedder(
            model_name="BAAI/bge-base-en-v1.5",  # Highest retrieval accuracy
            batch_size=16,                        # Batch size for 768-dim model
            device=None,                          # Auto-detect GPU/CPU
            server_root=Path(selection.server_root)
        )

        # If --file argument provided, embed single file
        if selection.file:
            chunked_file = Path(selection.file)
            if not chunked_file.exists():
                print(f"[ERROR] File not found: {chunked_file}")
                return 1

            print(f"\n[SINGLE FILE] Processing: {chunked_file}")
            print("=" * 70)

            try:
                with open(chunked_file, 'r', encoding='utf-8') as f:
                    chunked_data = json.load(f)

                chunks = chunked_data.get('chunks', [])
                if not chunks:
                    print("[WARN] No chunks found in file")
                    return 0

                # Extract customer_id from chunks metadata
                customer_id = chunks[0].get('metadata', {}).get('customer_id', 'unknown')

                # Embed
                result = embedder.embed_customer_chunks(customer_id, chunks)

                if result.get('status') != 'no_chunks':
                    # Save to content_embedded.json in same directory
                    embedded_file = chunked_file.parent / "content_embedded.json"
                    # Save already-embedded chunks (don't re-embed)
                    embedder._save_embeddings(result, embedded_file)
                    print(f"\n[OK] Embeddings saved to: {embedded_file}")
                    return 0
                else:
                    return 1

            except Exception as e:
                logger.error(f"Failed to embed file: {e}", exc_info=True)
                print(f"[ERROR] {e}")
                return 1

        customer_filter = selection.customer_filter()
        results = embedder.embed_all_customers(customer_filter=customer_filter)
        success = all(r['status'] in ['success', 'no_chunked_documents'] for r in results.values())
        return 0 if success else 1

    except Exception as e:
        logger.error(f"Embedding pipeline failed: {e}", exc_info=True)
        print(f"[ERROR] {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
