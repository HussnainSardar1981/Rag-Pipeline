"""
Production Text Cleaning Script - Text-Only Extraction
Regex-based programmatic cleanup (no LLM, no OCR)

Cleaning pipeline:
  Stage 1: Remove boilerplate patterns (Madison headers, copyright, page numbers)
  Stage 2: Normalize whitespace and formatting
  Stage 3: Preserve numbered steps and procedural structure

For born-digital instruction PDFs with clean, selectable text.

Usage:
  # Clean single document
  python cleaner.py --file /path/to/content.json --customer CUSTOMER_ID

  # Clean all documents for one customer
  python cleaner.py --customer CUSTOMER_ID

  # Clean all documents for all customers
  python cleaner.py --all

  # Clean specific customers
  python cleaner.py --customers CUST1 CUST2 CUST3
"""

import json
import sys
import logging
import re
import os
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths_config import PathsConfig
from cli_common import add_standard_args, build_selection, configure_logging

logger = logging.getLogger(__name__)


def rigorous_pre_cleanup(text: str) -> str:
    """
    Stage 1: Rigorous programmatic cleanup before LLM processing
    Removes boilerplate, excessive whitespace, and structural noise
    This ensures the LLM focuses on semantic fixes, not garbage removal
    """
    if not text or not text.strip():
        return ""

    # Common boilerplate patterns found in PDFs (manually identified - NOT LLM dependent)
    # IMPORTANT: These patterns are carefully crafted to NOT remove solution content
    # especially numbered steps (1., 2., 3., etc.) that are part of instructions
    boilerplate_patterns = [
        # Document prep lines - match "Prepared by/By Madison Technology for {customer}"
        # Matches line breaks after the removal to clean up orphaned text
        r'Prepared\s+(by|By)\s+Madison\s+Technology\s+for\s+[^\n]+\n*',
        # Madison IT company header/footer (completely ignore this)
        r'MADISON\s+TECHNOLO[GC].*?Managed Hosting.*?Services',
        # MTI Support Help Desk contact info (completely ignore)
        r'MTI\s+Support\s+Help\s+Desk\s+T:\s*\+1\s*\(\s*212\s*\)\s*400-7550.*?www\.madisonti\.com',
        # Madison Technology How-To header (completely ignore)
        r'Madison\s+Technology\s*\n\s*.*?How\s+To\s+Use.*?Rev\s+1a',
        # Page numbers (remove)
        r'Page\s+\d+\s+of\s+\d+',
        # Confidential notice (remove)
        r'Confidential',
        # Copyright (remove)
        r'Copyright.*?\d{4}',
        # Footer with copyright (remove)
        r'Footer.*?©.*?\d{4}',
    ]

    # Remove boilerplate patterns
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # After removing boilerplate, clean up any orphaned single words followed by newlines
    # (handles cases like "Dean\n\nGetting Started" where only part of line was removed)
    text = re.sub(r'^\w+\s*\n+', '', text, flags=re.MULTILINE)

    # Normalize excessive whitespace and indentation
    # Replace 2+ spaces/tabs with single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Clean up multiple consecutive newlines (keep max 2 for paragraphs)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Remove empty lines at start/end
    text = text.strip()

    return text


class TextCleaner:
    """Clean extracted PDF text using regex-based programmatic cleaning (no LLM)"""

    def __init__(self):
        """Initialize text cleaner with programmatic cleaning only"""
        # No LLM initialization needed - using regex-based cleaning only
        pass

    def clean_text_programmatically(self, text: str) -> str:
        """
        Clean extracted text using only programmatic (regex-based) methods
        No LLM needed - deterministic and fast
        """
        if not text.strip():
            return ""

        # Stage: Rigorous programmatic cleanup (removes boilerplate, OCR artifacts, whitespace)
        text = rigorous_pre_cleanup(text)

        # Additional OCR error fixes (deterministic patterns only)
        ocr_fixes = [
            (r'(?<!\w)1l(?!\w)', 'll'),  # '1l' → 'll' (not in middle of words)
            (r'(?<!\w)rn(?!\w)', 'm'),   # 'rn' → 'm' (not in middle of words)
            (r'(?<!\w)O0(?!\w)', '00'),  # 'O0' → '00' (obvious digit confusion)
        ]

        for pattern, replacement in ocr_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Fix hyphenated word breaks (e.g., "informa-\ntion" → "information")
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

        # Clean up remaining artifacts
        text = text.strip()

        return text

    def clean_extraction_result(self, extraction_json: Dict) -> Dict:
        """
        Clean all text in an extraction result

        IMPORTANT: OCR text from images is cleaned but kept SEPARATE from PDF text
        This preserves the distinction between:
        - PDF text: procedural instructions and documentation
        - OCR text: visual context from interface screenshots
        """
        pdf_name = extraction_json['metadata'].get('source_pdf', 'unknown')
        logger.info(f"Cleaning extraction result for {pdf_name}")

        # Use deep copy to prevent mutating original data
        cleaned_result = copy.deepcopy(extraction_json)
        cleaned_pages = []

        for page_data in extraction_json['pages']:
            cleaned_page = page_data

            # Clean PDF text
            # Note: extraction_pipeline.py creates pages with 'text' key, not 'pdf_text'
            pdf_text = page_data.get('text', '').strip()

            if pdf_text:
                logger.debug(f"Cleaning page {page_data['page_num']} PDF text ({len(pdf_text)} chars)")
                cleaned_pdf_text = self.clean_text_programmatically(pdf_text)
                logger.debug(f"Cleaned to {len(cleaned_pdf_text)} chars")
                cleaned_page['pdf_text'] = cleaned_pdf_text
            else:
                cleaned_page['pdf_text'] = ""

            # Note: Image/OCR processing removed - text-only extraction
            # If images exist in old extractions, skip them
            cleaned_pages.append(cleaned_page)

        cleaned_result['pages'] = cleaned_pages
        cleaned_result['metadata']['cleaned_at'] = datetime.now().isoformat()
        cleaned_result['metadata']['cleaning_notes'] = (
            "Text-only cleaning (no OCR): "
            "1) Remove boilerplate patterns (Madison headers, copyright notices), "
            "2) Normalize whitespace and fix formatting, "
            "3) Preserve numbered steps and procedural structure. "
            "Born-digital PDF text extraction only."
        )

        return cleaned_result


class CleaningPipeline:
    """Manage text cleaning for customers (regex-based, no LLM)"""

    def __init__(self, server_root: Path):
        self.server_root = Path(server_root)

        logger.info(f"Cleaning pipeline initialized with root: {self.server_root}")
        logger.info("Using regex-based cleaning (no LLM)")

    def clean_single_file(self, json_path: Path) -> bool:
        """Clean a single extracted JSON file"""

        if not json_path.exists():
            logger.error(f"File not found: {json_path}")
            return False

        try:
            logger.info(f"Cleaning file: {json_path}")

            with open(json_path, 'r', encoding='utf-8') as f:
                extraction_result = json.load(f)

            cleaner = TextCleaner()
            cleaned_result = cleaner.clean_extraction_result(extraction_result)

            # Save with _cleaned suffix (atomic write to prevent corruption)
            cleaned_path = json_path.parent / f"{json_path.stem}_cleaned.json"
            temp_path = cleaned_path.with_suffix('.tmp')

            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_result, f, indent=2, ensure_ascii=False)

            # Atomic rename
            os.replace(temp_path, cleaned_path)

            logger.info(f"Saved cleaned result to {cleaned_path}")
            print(f"[OK] Cleaned: {json_path.name} -> {cleaned_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clean {json_path}: {e}")
            print(f"[ERROR] {json_path.name}: {e}")
            return False

    def clean_customer(self, customer_id: str) -> Dict:
        """Clean all extracted PDFs for one customer"""
        customer_dir = self.server_root / "customers" / customer_id

        if not customer_dir.exists():
            logger.warning(f"No customer folder for {customer_id}")
            return {
                'customer_id': customer_id,
                'status': 'no_documents',
                'total_documents': 0,
                'cleaned': 0,
                'failed': 0
            }

        extracted_files = list(customer_dir.glob("*/content.json"))

        cleaned_count = 0
        skipped_count = 0
        failed_count = 0

        print(f"\n[CLEAN] Starting cleaning for customer: {customer_id}")
        print("="*70)

        for extracted_json_path in sorted(extracted_files):
            try:
                doc_name = extracted_json_path.parent.name
                cleaned_json_path = extracted_json_path.parent / "content_cleaned.json"

                # Skip if already cleaned AND content.json hasn't changed
                if cleaned_json_path.exists():
                    content_mtime = extracted_json_path.stat().st_mtime
                    cleaned_mtime = cleaned_json_path.stat().st_mtime

                    if cleaned_mtime >= content_mtime:
                        print(f"  [SKIP] {doc_name} (already cleaned)")
                        skipped_count += 1
                        continue

                print(f"  [PROCESS] {doc_name}")

                with open(extracted_json_path, 'r', encoding='utf-8') as f:
                    extraction_result = json.load(f)

                cleaner = TextCleaner()
                cleaned_result = cleaner.clean_extraction_result(extraction_result)

                # Atomic write to prevent corruption
                temp_path = cleaned_json_path.with_suffix('.tmp')

                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_result, f, indent=2, ensure_ascii=False)

                os.replace(temp_path, cleaned_json_path)

                print(f"    [OK] {len(cleaned_result['pages'])} pages cleaned")
                cleaned_count += 1

            except Exception as e:
                logger.error(f"Failed to clean {extracted_json_path.parent.name}: {e}", exc_info=True)
                print(f"    [ERROR] {e}")
                failed_count += 1

        print("="*70)
        print(f"[SUMMARY] {customer_id}")
        print(f"  Cleaned: {cleaned_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(extracted_files)}")

        return {
            'customer_id': customer_id,
            'status': 'success' if failed_count == 0 else 'partial',
            'total_documents': len(extracted_files),
            'cleaned': cleaned_count,
            'failed': failed_count
        }

    def clean_all_customers(self, customer_filter: Optional[List[str]] = None) -> Dict:
        """Clean extracted documents for all customers"""
        print("\n" + "="*70)
        print("TEXT CLEANING PIPELINE")
        print("="*70)

        customers_dir = self.server_root / "customers"

        if not customers_dir.exists():
            logger.error(f"Customers directory not found: {customers_dir}")
            return {}

        all_customers = []
        for customer_folder in sorted(customers_dir.glob("*")):
            if customer_folder.is_dir():
                if any(customer_folder.glob("*/content.json")):
                    all_customers.append(customer_folder.name)

        print(f"\n[DISCOVER] Found {len(all_customers)} customers with extracted documents")
        for customer_id in all_customers:
            print(f"  - {customer_id}")

        if customer_filter:
            all_customers = [c for c in all_customers if c in customer_filter]
            print(f"\n[FILTER] Processing {len(all_customers)} selected customers")

        results = {}
        for customer_id in all_customers:
            result = self.clean_customer(customer_id)
            results[customer_id] = result

        # Print final summary
        print("\n" + "="*70)
        print("CLEANING SUMMARY")
        print("="*70)

        total_documents = sum(r['total_documents'] for r in results.values())
        total_cleaned = sum(r['cleaned'] for r in results.values())
        total_failed = sum(r['failed'] for r in results.values())

        for customer_id, result in sorted(results.items()):
            status_icon = "[OK]" if result['status'] == 'success' else "[WARN]"
            print(f"{status_icon} {customer_id:25} {result['cleaned']}/{result['total_documents']} cleaned")

        print("\n" + "-"*70)
        print(f"Total Documents: {total_documents}")
        print(f"Cleaned: {total_cleaned}")
        print(f"Failed: {total_failed}")
        print("="*70)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Text Cleaning Pipeline for Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean single file
  python clean_documents_server.py --file /path/to/content.json

  # Clean all documents for one customer
  python clean_documents_server.py --customer CUSTOMER_ID

  # Clean all documents
  python clean_documents_server.py --all

  # Clean specific customers
  python clean_documents_server.py --customers CUST1 CUST2
        """
    )

    add_standard_args(parser, include_file=True, include_customer_id=False, include_server_root=True)

    args = parser.parse_args()
    selection = build_selection(args)

    server_root = selection.server_root
    configure_logging("cleaner", server_root=server_root)

    try:
        pipeline = CleaningPipeline(server_root)

        if selection.file:
            # Clean single file
            success = pipeline.clean_single_file(selection.file)
            sys.exit(0 if success else 1)

        customer_filter = selection.customer_filter()
        results = pipeline.clean_all_customers(customer_filter=customer_filter)
        failed = sum(1 for r in results.values() if r['status'] not in ['success', 'no_documents'])
        sys.exit(1 if failed > 0 else 0)

    except Exception as e:
        logger.error(f"Cleaning pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
