"""
Token-Aware Text Chunking for RAG Pipeline
Splits cleaned text into 400-600 token chunks with 20-30% overlap
Generates SHA256 hashes for deduplication
"""

import json
import hashlib
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths_config import PathsConfig
from cli_common import add_standard_args, build_selection, configure_logging

# Try to import tokenizers for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("[WARN] tiktoken not installed. Using character-based estimation.")
    print("       For accurate token counting: pip install tiktoken")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


class TokenCounter:
    """Count tokens using tiktoken or character estimation"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize token counter"""
        self.use_tiktoken = TIKTOKEN_AVAILABLE
        if self.use_tiktoken:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base encoding
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Character-based estimation: ~4 chars per token
            self.chars_per_token = 4

    def count(self, text: str) -> int:
        """Count tokens in text"""
        if self.use_tiktoken:
            return len(self.encoding.encode(text))
        else:
            # Rough estimation: ~4 characters = 1 token
            return len(text) // self.chars_per_token


class DocumentChunker:
    """Split documents into token-aware chunks with overlap"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 250,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target tokens per chunk (800-1200 recommended for voicebot)
            chunk_overlap: Overlap in tokens (20-30% of chunk_size)
            model: Model for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = TokenCounter(model)

        # Validate parameters
        if not (0 <= chunk_overlap < chunk_size):
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be >=0 and < chunk_size ({chunk_size})")

        overlap_percent = (chunk_overlap / chunk_size) * 100
        if overlap_percent < 15 or overlap_percent > 35:
            logger.warning(f"Overlap is {overlap_percent:.1f}% (recommended 20-30%)")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) < 50:
            return [text] if text.strip() else []

        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.token_counter.count(para)

            # If paragraph alone exceeds chunk size, split further
            if para_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split paragraph by sentences
                sentences = para.split('. ')
                for sentence in sentences:
                    if not sentence:
                        continue

                    sent_tokens = self.token_counter.count(sentence + '. ')

                    if sent_tokens > self.chunk_size:
                        # Word-level split as last resort
                        words = sentence.split(' ')
                        sub_chunk = []
                        sub_tokens = 0

                        for word in words:
                            word_tokens = self.token_counter.count(word + ' ')
                            if sub_tokens + word_tokens > self.chunk_size and sub_chunk:
                                chunks.append(' '.join(sub_chunk))
                                sub_chunk = [word]
                                sub_tokens = word_tokens
                            else:
                                sub_chunk.append(word)
                                sub_tokens += word_tokens

                        if sub_chunk:
                            chunks.append(' '.join(sub_chunk))

                    elif current_tokens + sent_tokens <= self.chunk_size:
                        current_chunk.append(sentence + '. ')
                        current_tokens += sent_tokens
                    else:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [sentence + '. ']
                        current_tokens = sent_tokens

            elif current_tokens + para_tokens <= self.chunk_size:
                current_chunk.append(para)
                current_tokens += para_tokens

            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        # Apply overlap: each chunk includes last N tokens of previous
        if len(chunks) > 1 and self.chunk_overlap > 0:
            chunks_with_overlap = []

            for i, chunk in enumerate(chunks):
                if i == 0:
                    chunks_with_overlap.append(chunk)
                    continue

                prev_chunk = chunks[i - 1]
                # If previous chunk has fewer tokens than overlap, take whole prev chunk
                prev_tokens = self.token_counter.count(prev_chunk)
                if prev_tokens <= self.chunk_overlap:
                    overlap_text = prev_chunk
                else:
                    # Walk words backwards from the end until we reach ~chunk_overlap tokens
                    words = prev_chunk.split()
                    overlap_words = []
                    overlap_token_count = 0

                    for w in reversed(words):
                        # Count tokens for the single word (include a trailing space)
                        wt = self.token_counter.count(w + ' ')
                        if overlap_token_count + wt > self.chunk_overlap:
                            break
                        overlap_words.insert(0, w)
                        overlap_token_count += wt

                    overlap_text = ' '.join(overlap_words)

                # Build combined chunk: keep clean separation if overlap_text non-empty
                combined = (overlap_text + '\n\n' if overlap_text else '') + chunk
                chunks_with_overlap.append(combined)

            return chunks_with_overlap

        return chunks

    def chunk_document(
        self,
        customer_id: str,
        doc_name: str,
        pages: List[Dict],
        existing_hashes: Optional[set] = None
    ) -> List[Dict]:
        """
        Chunk a complete document (all pages)

        Args:
            customer_id: Customer identifier
            doc_name: Document name
            pages: List of page dicts with 'page_num' and 'pdf_text' keys
            existing_hashes: Set of existing chunk hashes (for deduplication)

        Returns:
            List of chunk dicts with metadata
        """
        if existing_hashes is None:
            existing_hashes = set()

        chunks = []
        chunk_id = 0

        for page in pages:
            page_num = page.get('page_num', 0)
            # Support both 'pdf_text' (cleaned) and 'text' (raw extraction)
            text_clean = page.get('pdf_text', page.get('text', ''))

            if not text_clean.strip():
                logger.debug(f"Skipping empty page {page_num}")
                continue

            # Split page into chunks
            page_chunks = self.chunk_text(text_clean)

            for chunk_text in page_chunks:
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                # Generate chunk hash for deduplication
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

                # Skip if duplicate
                if chunk_hash in existing_hashes:
                    logger.debug(f"Skipping duplicate chunk: {chunk_hash[:8]}...")
                    continue

                chunk_id += 1

                # Count tokens
                token_count = self.token_counter.count(chunk_text)

                # Create chunk metadata
                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'chunk_hash': chunk_hash,
                    'customer_id': customer_id,
                    'doc_name': doc_name,
                    'page_num': page_num,
                    'text_length': len(chunk_text),
                    'token_count': token_count,
                    'created_at': datetime.now().isoformat()
                }

                chunks.append({
                    'metadata': chunk_metadata,
                    'text': chunk_text
                })

                existing_hashes.add(chunk_hash)

        logger.info(f"Chunked {doc_name}: {chunk_id} chunks from {len(pages)} pages")

        return chunks


class RAGChunker:
    """Coordinate chunking for all customer documents"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 250,
        server_root: Optional[Path] = None
    ):
        """Initialize RAG chunker"""
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.server_root = server_root or PathsConfig.RAG_DIR

    def chunk_customer_documents(self, customer_id: str) -> Dict:
        """
        Chunk all documents for a customer

        Directory structure:
        customers/{customer_id}/{pdf_name}/
        ├── content.json (extracted, raw)
        ├── content_cleaned.json (cleaned, ready for chunking)
        └── images/

        Returns:
            {
                'customer_id': '...',
                'documents_processed': N,
                'total_chunks': N,
                'chunks': [...],
                'created_at': '...'
            }
        """
        print(f"\n[CHUNK] Processing customer: {customer_id}")
        print("=" * 70)

        # Paths
        customer_dir = self.server_root / "customers" / customer_id
        kb_metadata_path = customer_dir / "kb_metadata.json"

        if not customer_dir.exists():
            logger.error(f"Customer directory not found: {customer_dir}")
            return {'status': 'error', 'message': 'No customer directory'}

        # Load existing chunk hashes for deduplication
        existing_hashes = set()
        if kb_metadata_path.exists():
            with open(kb_metadata_path, 'r', encoding='utf-8') as f:
                kb_metadata = json.load(f)
                for file_info in kb_metadata.get('files', {}).values():
                    existing_hashes.update(file_info.get('chunk_hashes', []))

        # Process each PDF folder in customer directory
        all_chunks = []
        documents_processed = 0
        skipped_count = 0

        for pdf_dir in sorted(customer_dir.iterdir()):
            if not pdf_dir.is_dir():
                continue

            # Skip metadata files
            if pdf_dir.name == 'kb_metadata.json':
                continue

            # Look for content_cleaned.json (cleaned content)
            cleaned_file = pdf_dir / "content_cleaned.json"
            if not cleaned_file.exists():
                logger.debug(f"No content_cleaned.json in {pdf_dir.name}, skipping")
                continue

            doc_name = pdf_dir.name
            chunked_output = pdf_dir / "content_chunked.json"

            # Skip if already chunked AND content_cleaned.json hasn't changed
            if chunked_output.exists():
                cleaned_mtime = cleaned_file.stat().st_mtime
                chunked_mtime = chunked_output.stat().st_mtime

                if chunked_mtime >= cleaned_mtime:
                    print(f"  [SKIP] {doc_name} (already chunked)")
                    skipped_count += 1
                    continue

            # Load cleaned content
            try:
                print(f"  [PROCESS] {doc_name}")
                logger.info(f"Chunking document: {doc_name}")

                with open(cleaned_file, 'r', encoding='utf-8') as f:
                    cleaned_data = json.load(f)

                pages = cleaned_data.get('pages', [])
                has_text = any(
                    (page.get('pdf_text', page.get('text', '')).strip())
                    for page in pages
                )
                if not has_text:
                    print(f"  [SKIP] {doc_name} (no text)")
                    logger.warning(f"No text found in {doc_name}; writing empty chunk file")
                    skipped_count += 1

                    chunked_data = {
                        'metadata': cleaned_data.get('metadata', {}),
                        'chunks': [],
                        'chunked_at': datetime.now().isoformat()
                    }

                    tmp_path = chunked_output.with_suffix('.tmp')
                    with open(tmp_path, 'w', encoding='utf-8') as f:
                        json.dump(chunked_data, f, indent=2, ensure_ascii=False)
                    os.replace(tmp_path, chunked_output)
                    continue

                # Chunk document
                doc_chunks = self.chunker.chunk_document(
                    customer_id=customer_id,
                    doc_name=doc_name,
                    pages=pages,
                    existing_hashes=existing_hashes
                )

                print(f"    [OK] Created {len(doc_chunks)} chunks")
                logger.info(f"Created {len(doc_chunks)} chunks for {doc_name}")

                # Save chunked result to content_chunked.json
                chunked_data = {
                    'metadata': cleaned_data.get('metadata', {}),
                    'chunks': doc_chunks,
                    'chunked_at': datetime.now().isoformat()
                }

                # Atomic write to prevent corruption
                tmp_path = chunked_output.with_suffix('.tmp')
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(chunked_data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, chunked_output)

                all_chunks.extend(doc_chunks)
                documents_processed += 1

            except Exception as e:
                logger.error(f"Failed to chunk {pdf_dir.name}: {e}", exc_info=True)
                print(f"    [ERROR] {e}")
                continue

        # Update kb_metadata.json with chunk hashes
        if kb_metadata_path.exists():
            with open(kb_metadata_path, 'r', encoding='utf-8') as f:
                kb_metadata = json.load(f)
        else:
            kb_metadata = {'files': {}}

        for chunk in all_chunks:
            doc_name = chunk['metadata']['doc_name']
            chunk_hash = chunk['metadata']['chunk_hash']

            if doc_name not in kb_metadata['files']:
                kb_metadata['files'][doc_name] = {'chunk_hashes': []}

            if chunk_hash not in kb_metadata['files'][doc_name]['chunk_hashes']:
                kb_metadata['files'][doc_name]['chunk_hashes'].append(chunk_hash)

        with open(kb_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(kb_metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Updated kb_metadata.json for {customer_id}")

        print("\n" + "=" * 70)
        print(f"[SUMMARY] {customer_id}")
        print(f"  Documents processed: {documents_processed}")
        print(f"  Documents skipped: {skipped_count}")
        print(f"  Total chunks created: {len(all_chunks)}")
        if all_chunks:
            token_counts = [c['metadata']['token_count'] for c in all_chunks]
            print(f"  Avg chunk size: {sum(token_counts) // len(token_counts)} tokens")
            print(f"  Token range: {min(token_counts)}-{max(token_counts)}")

        return {
            'customer_id': customer_id,
            'documents_processed': documents_processed,
            'total_chunks': len(all_chunks),
            'chunks': all_chunks,
            'created_at': datetime.now().isoformat()
        }

    def chunk_all_customers(self, customer_ids: Optional[List[str]] = None) -> Dict:
        """Chunk documents for all customers"""
        print("\n" + "=" * 70)
        print("RAG DOCUMENT CHUNKING")
        print("=" * 70)

        results = {}

        # Get customer list
        if customer_ids is None:
            customers_dir = self.server_root / "customers"
            customer_ids = [d.name for d in customers_dir.iterdir() if d.is_dir()]

        for customer_id in customer_ids:
            result = self.chunk_customer_documents(customer_id)
            results[customer_id] = result

        # Final summary
        print("\n" + "=" * 70)
        print("CHUNKING COMPLETE")
        print("=" * 70)

        total_chunks = sum(r.get('total_chunks', 0) for r in results.values())
        print(f"Total customers: {len(results)}")
        print(f"Total chunks: {total_chunks}")

        return results


def chunk_single_file(cleaned_file_path: Path) -> bool:
    """Chunk a single content_cleaned.json file"""
    cleaned_file_path = Path(cleaned_file_path)

    if not cleaned_file_path.exists():
        logger.error(f"File not found: {cleaned_file_path}")
        print(f"[ERROR] File not found: {cleaned_file_path}")
        return False

    try:
        # Extract customer and document name from path
        # Expected: /path/to/customers/{customer_id}/{pdf_name}/content_cleaned.json
        parts = cleaned_file_path.parts
        pdf_name = parts[-2]
        customer_id = parts[-3]

        print("\n" + "=" * 70)
        print(f"CHUNKING SINGLE FILE: {customer_id}/{pdf_name}")
        print("=" * 70)

        # Load cleaned content
        with open(cleaned_file_path, 'r', encoding='utf-8') as f:
            cleaned_data = json.load(f)

        pages = cleaned_data.get('pages', [])
        print(f"  Pages: {len(pages)}")
        has_text = any(
            (page.get('pdf_text', page.get('text', '')).strip())
            for page in pages
        )
        if not has_text:
            print(f"  [SKIP] {pdf_name} (no text)")

            chunked_output = cleaned_file_path.parent / "content_chunked.json"
            chunked_data = {
                'metadata': cleaned_data.get('metadata', {}),
                'chunks': [],
                'chunked_at': datetime.now().isoformat()
            }

            tmp_path = chunked_output.with_suffix('.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, chunked_output)

            print(f"\n[OK] Saved: {chunked_output.name}")
            print("=" * 70)
            logger.info(f"Saved empty chunk file for {customer_id}/{pdf_name}")
            return True

        # Initialize chunker with production parameters (optimized for voicebot)
        chunker_obj = DocumentChunker(chunk_size=1000, chunk_overlap=250)

        # Chunk document
        doc_chunks = chunker_obj.chunk_document(
            customer_id=customer_id,
            doc_name=pdf_name,
            pages=pages,
            existing_hashes=set()
        )

        print(f"  Chunks created: {len(doc_chunks)}")

        if doc_chunks:
            token_counts = [c['metadata']['token_count'] for c in doc_chunks]
            print(f"  Token range: {min(token_counts)}-{max(token_counts)} (avg {sum(token_counts) // len(token_counts)})")

        # Save chunked result
        chunked_output = cleaned_file_path.parent / "content_chunked.json"
        chunked_data = {
            'metadata': cleaned_data.get('metadata', {}),
            'chunks': doc_chunks,
            'chunked_at': datetime.now().isoformat()
        }

        # Atomic write to prevent corruption
        tmp_path = chunked_output.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(chunked_data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, chunked_output)

        print(f"\n[OK] Saved: {chunked_output.name}")
        print("=" * 70)
        logger.info(f"Successfully chunked {customer_id}/{pdf_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to chunk file {cleaned_file_path}: {e}", exc_info=True)
        print(f"[ERROR] {e}")
        return False


def main():
    if DOTENV_AVAILABLE:
        load_dotenv()

    server_root = PathsConfig.RAG_DIR

    import argparse
    parser = argparse.ArgumentParser(
        description='Token-Aware Document Chunking for RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Chunk single file
  python chunker.py --file /path/to/customers/customer_id/PDF_Name/content_cleaned.json

  # Chunk all documents for one customer
  python chunker.py --customer customer_id

  # Chunk specific customers
  python chunker.py --customers stuart_dean cidny

  # Chunk all customers
  python chunker.py --all

  # Custom server root
  python chunker.py --server-root /path/to/server --customer customer_id
        '''
    )
    add_standard_args(parser, include_file=True, include_customer_id=False, include_server_root=True)

    args = parser.parse_args()
    selection = build_selection(args)

    try:
        configure_logging("chunker", server_root=selection.server_root)

        # Mode 1: Single file
        if selection.file:
            success = chunk_single_file(selection.file)
            return 0 if success else 1

        # Mode 2: Multiple modes
        server_root_path = selection.server_root
        chunker = RAGChunker(
            chunk_size=1000,
            chunk_overlap=250,
            server_root=server_root_path
        )

        customer_ids = selection.customer_filter()
        results = chunker.chunk_all_customers(customer_ids=customer_ids)

        # Summary - return success if we processed any customers
        # (Skipping existing chunks is expected behavior, not a failure)
        success = len(results) > 0
        return 0 if success else 1

    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
