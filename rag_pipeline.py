"""
Master RAG Pipeline Orchestrator
Runs all stages: Download -> Extract -> Clean -> Chunk -> Embed -> Index
For one customer, all customers, or a single file

CHANGE DETECTION & SKIP LOGIC (Built-in):
==========================================

Each stage has intelligent change detection to avoid re-processing:

1. DOWNLOAD Stage:
   - Uses ETag (HTTP headers) to detect if files changed on SharePoint
   - Only downloads if file is NEW or MODIFIED
   - Stores ETag in metadata for comparison

2. EXTRACT Stage:
   - Tracks extraction_metadata.json per customer
   - Skips PDFs that were already extracted
   - Re-extracts if PDF was modified (detected by ETag)

3. CLEAN Stage:
   - Checks if document was already cleaned
   - Skips unchanged documents
   - Only cleans new or modified documents

4. CHUNK Stage:
   - Converts cleaned content into optimal-sized chunks
   - Skips if content_chunked.json already exists
   - Creates chunks with proper boundaries and metadata

5. EMBED Stage:
   - Reads chunked content from Stage 4
   - Skips duplicate chunks
   - Only embeds new chunks

6. INDEX Stage:
   - Uses skip_duplicates=True by default
   - Skips chunks already in ChromaDB
   - Only indexes new chunks

RESULT: When you run the pipeline again, it will:
  - Automatically detect changed files on SharePoint
  - Download only new/modified PDFs
  - Extract only new/modified PDFs
  - Clean only new documents
  - Chunk only new documents
  - Embed only new chunks
  - Index only new chunks

This makes re-running the pipeline FAST and EFFICIENT!
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import argparse
import json
import os

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_common import add_standard_args, build_selection, configure_logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SCRIPTS = {
    'download': SCRIPT_DIR / 'downloader.py',
    'extract': SCRIPT_DIR / 'extraction.py',
    'clean': SCRIPT_DIR / 'cleaner.py',
    'chunk': SCRIPT_DIR / 'chunker.py',
    'embed': SCRIPT_DIR / 'embeddings.py',
    'index': SCRIPT_DIR / 'indexing.py',
}

STAGES = ['download', 'extract', 'clean', 'chunk', 'embed', 'index']


class RAGPipeline:
    """Orchestrate the complete RAG pipeline"""

    def __init__(self, skip_stages=None, only_stages=None):
        """
        Initialize pipeline

        Args:
            skip_stages: List of stages to skip (e.g., ['download', 'extract'])
            only_stages: Only run these stages (e.g., ['embed', 'index'])
        """
        self.skip_stages = skip_stages or []
        self.only_stages = only_stages or []
        self.results = {}

    def get_stages_to_run(self):
        """Determine which stages to run"""
        if self.only_stages:
            return [s for s in STAGES if s in self.only_stages]
        return [s for s in STAGES if s not in self.skip_stages]

    def _build_args_for_stage(self, stage, mode, target=None, server_root: Optional[Path] = None):
        """
        Build correct arguments for each stage based on mode

        Args:
            stage: Stage name (download, extract, clean, chunk, embed, index)
            mode: 'all', 'customer', or 'file'
            target: customer_id or file_path depending on mode

        Returns:
            List of command line arguments
        """
        args = []

        if mode == 'all':
            # ALL CUSTOMERS mode
            if stage == 'download':
                args = ['--all']
            elif stage in ('extract', 'clean', 'chunk', 'embed', 'index'):
                args = ['--all']

        elif mode == 'customer':
            # SINGLE CUSTOMER mode
            customer_id = target
            if stage in ('download', 'extract', 'clean', 'chunk', 'embed', 'index'):
                args = ['--customer', customer_id]

        elif mode == 'file':
            # SINGLE FILE mode
            file_path, customer_id = target
            if stage == 'download':
                # Can't download single file - skip or error
                logger.warning("Download stage not supported for single file mode")
                return None
            elif stage == 'extract':
                # Extract requires a PDF path; single-file mode targets later stages
                logger.warning("Extract stage not supported for single file mode (needs PDF)")
                return None
            elif stage in ('clean', 'chunk', 'embed', 'index'):
                args = ['--file', str(file_path)]

        if server_root:
            args.extend(['--server-root', str(server_root)])

        return args

    def run_stage(self, stage, mode, target=None, server_root: Optional[Path] = None):
        """
        Run a single stage

        Args:
            stage: Stage name (download, extract, clean, chunk, embed, index)
            mode: 'all', 'customer', or 'file'
            target: customer_id or (file_path, customer_id) depending on mode

        Returns:
            True if successful, False otherwise
        """
        script = SCRIPTS.get(stage)
        if not script or not script.exists():
            logger.error(f"Script not found: {script}")
            return False

        # Build correct arguments for this stage
        args = self._build_args_for_stage(stage, mode, target, server_root=server_root)
        if args is None:
            # Stage not supported for this mode - skip gracefully
            logger.info(f"Stage {stage} skipped (not applicable for {mode} mode)")
            self.results[stage] = 'skipped'
            return True

        print(f"\n{'='*70}")
        print(f"RUNNING STAGE: {stage.upper()}")
        print(f"{'='*70}")
        print(f"Mode: {mode}")
        if mode == 'customer':
            print(f"Customer: {target}")
        elif mode == 'file':
            print(f"File: {target[0]}")
        print(f"{'='*70}\n")

        try:
            cmd = ['python', str(script)] + args
            logger.info(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(cmd, check=True, capture_output=False)

            self.results[stage] = 'success'
            logger.info(f"Stage {stage} completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            self.results[stage] = 'failed'
            logger.error(f"Stage {stage} failed with exit code {e.returncode}")
            return False
        except Exception as e:
            self.results[stage] = 'error'
            logger.error(f"Stage {stage} error: {e}")
            return False

    def run_all_customers(self, skip_stages=None, only_stages=None, server_root: Optional[Path] = None):
        """
        Run full pipeline for all customers

        Args:
            skip_stages: Stages to skip
            only_stages: Only run these stages
        """
        print(f"\n{'='*70}")
        print("RAG PIPELINE: ALL CUSTOMERS")
        print(f"{'='*70}\n")

        if skip_stages:
            self.skip_stages = skip_stages
        if only_stages:
            self.only_stages = only_stages

        stages = self.get_stages_to_run()
        logger.info(f"Running stages: {', '.join(stages)}")

        for stage in stages:
            success = self.run_stage(stage, mode='all', server_root=server_root)
            if not success and self.results.get(stage) != 'skipped':
                logger.error(f"Stage {stage} failed")
                print(f"\n[ERROR] Stage {stage} failed. Continue anyway? (y/n): ", end='')
                try:
                    response = input().lower()
                    if response != 'y':
                        logger.error("Pipeline aborted by user")
                        return False
                except (KeyboardInterrupt, EOFError):
                    logger.error("Pipeline aborted")
                    return False

        return self._print_summary()

    def run_single_customer(self, customer_id, skip_stages=None, only_stages=None,
                            server_root: Optional[Path] = None):
        """
        Run full pipeline for a single customer

        Args:
            customer_id: Customer identifier
            skip_stages: Stages to skip
            only_stages: Only run these stages
        """
        print(f"\n{'='*70}")
        print(f"RAG PIPELINE: CUSTOMER {customer_id}")
        print(f"{'='*70}\n")

        if skip_stages:
            self.skip_stages = skip_stages
        if only_stages:
            self.only_stages = only_stages

        stages = self.get_stages_to_run()
        logger.info(f"Running stages for {customer_id}: {', '.join(stages)}")

        for stage in stages:
            success = self.run_stage(stage, mode='customer', target=customer_id, server_root=server_root)
            if not success and self.results.get(stage) != 'skipped':
                logger.error(f"Stage {stage} failed for {customer_id}")
                print(f"\n[ERROR] Stage {stage} failed. Continue? (y/n): ", end='')
                try:
                    response = input().lower()
                    if response != 'y':
                        logger.error("Pipeline aborted by user")
                        return False
                except (KeyboardInterrupt, EOFError):
                    logger.error("Pipeline aborted")
                    return False

        return self._print_summary()

    def run_single_file(self, file_path, customer_id=None, skip_stages=None, only_stages=None,
                        server_root: Optional[Path] = None):
        """
        Run pipeline stages for a single file

        Note: Download and Extract stages are not applicable for single file mode

        Args:
            file_path: Path to file (content_cleaned.json, content_chunked.json, etc.)
            customer_id: Customer identifier (optional, extracted from path if not provided)
            skip_stages: Stages to skip
            only_stages: Only run these stages
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        if not customer_id:
            # Try to extract customer_id from path structure
            # Expected: .../customers/customer_id/doc_name/file.json
            try:
                customer_id = file_path.parent.parent.name
            except:
                logger.error("Could not extract customer_id from path. Please provide --customer-id")
                return False

        print(f"\n{'='*70}")
        print(f"RAG PIPELINE: SINGLE FILE")
        print(f"File: {file_path}")
        print(f"Customer: {customer_id}")
        print(f"{'='*70}\n")

        if skip_stages:
            self.skip_stages = skip_stages
        if only_stages:
            self.only_stages = only_stages

        stages = self.get_stages_to_run()
        logger.info(f"Running stages for file: {', '.join(stages)}")

        # Warn about unsupported stages
        if 'download' in stages or 'extract' in stages:
            logger.warning("Download and Extract stages are not supported for single file mode (will be skipped)")

        for stage in stages:
            success = self.run_stage(stage, mode='file', target=(file_path, customer_id),
                                     server_root=server_root)
            if not success and self.results.get(stage) != 'skipped':
                logger.error(f"Stage {stage} failed")
                print(f"\n[ERROR] Stage {stage} failed. Continue? (y/n): ", end='')
                try:
                    response = input().lower()
                    if response != 'y':
                        logger.error("Pipeline aborted by user")
                        return False
                except (KeyboardInterrupt, EOFError):
                    logger.error("Pipeline aborted")
                    return False

        return self._print_summary()

    def _print_summary(self):
        """Print pipeline execution summary"""
        print(f"\n{'='*70}")
        print("PIPELINE SUMMARY")
        print(f"{'='*70}")
        for stage, status in self.results.items():
            if status == 'success':
                icon = "[OK]"
            elif status == 'skipped':
                icon = "[SKIP]"
            elif status == 'failed':
                icon = "[FAIL]"
            else:
                icon = "[ERROR]"
            print(f"{icon} {stage:15} {status}")
        print(f"{'='*70}\n")

        all_success = all(s in ('success', 'skipped') for s in self.results.values())
        return all_success


def main():
    parser = argparse.ArgumentParser(
        description='RAG Pipeline Orchestrator - Run all stages at once',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for all customers
  python rag_pipeline.py --all

  # Run full pipeline for one customer
  python rag_pipeline.py --customer third_ave_apothecary

  # Run pipeline for a single file (skips download/extract)
  python rag_pipeline.py --file "customers/customer_id/doc/content_cleaned.json"

  # Skip download/extract stages, only clean/chunk/embed/index
  python rag_pipeline.py --all --skip download extract

  # Run only chunk/embed/index stages
  python rag_pipeline.py --all --only chunk embed index

  # Run only embedding stage for one customer
  python rag_pipeline.py --customer stuart_dean --only embed

Pipeline Stages (in order):
  1. download  - Download PDFs from SharePoint
  2. extract   - Extract text from PDFs
  3. clean     - Clean and normalize extracted text
  4. chunk     - Split cleaned content into optimal chunks (1000 tokens)
  5. embed     - Generate embeddings for chunks (BGE model)
  6. index     - Index embeddings into ChromaDB

Note: Single file mode automatically skips download/extract stages
        """
    )

    parser.add_argument(
        '--skip',
        nargs='+',
        choices=STAGES,
        help='Stages to skip'
    )
    parser.add_argument(
        '--only',
        nargs='+',
        choices=STAGES,
        help='Only run these stages'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        help='ChromaDB path (passed to indexing stage)'
    )
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run RAG health check after pipeline completes'
    )

    add_standard_args(parser, include_file=True, include_customer_id=True, include_server_root=True)

    args = parser.parse_args()
    selection = build_selection(args)

    configure_logging("rag_pipeline", server_root=selection.server_root)

    if args.skip and args.only:
        logger.error("Cannot use --skip and --only together")
        return 1

    # Add db-path to environment if provided
    if args.db_path:
        os.environ['CHROMA_DB_PATH'] = args.db_path

    try:
        if selection.file:
            pipeline = RAGPipeline(skip_stages=args.skip, only_stages=args.only)
            success = pipeline.run_single_file(
                selection.file,
                customer_id=selection.customer_id,
                server_root=selection.server_root
            )
        else:
            customer_filter = selection.customer_filter()
            if customer_filter:
                success = True
                for customer_id in customer_filter:
                    pipeline = RAGPipeline(skip_stages=args.skip, only_stages=args.only)
                    if not pipeline.run_single_customer(
                        customer_id,
                        server_root=selection.server_root
                    ):
                        success = False
            else:
                pipeline = RAGPipeline(skip_stages=args.skip, only_stages=args.only)
                success = pipeline.run_all_customers(server_root=selection.server_root)

        if args.health_check:
            health_script = SCRIPT_DIR / "health_check.py"
            if health_script.exists():
                hc_args = ['python', str(health_script), '--server-root', str(selection.server_root)]
                if selection.file and str(selection.file).endswith("content_embedded.json"):
                    hc_args.extend(['--file', str(selection.file)])
                elif selection.customer_id:
                    hc_args.extend(['--customer', selection.customer_id])
                elif selection.customer or selection.customers:
                    customer_filter = selection.customer_filter() or []
                    if len(customer_filter) == 1:
                        hc_args.extend(['--customer', customer_filter[0]])
                    elif customer_filter:
                        hc_args.append('--customers')
                        hc_args.extend(customer_filter)

                logger.info(f"Running health check: {' '.join(hc_args)}")
                try:
                    subprocess.run(hc_args, check=False)
                except Exception as exc:
                    logger.error(f"Health check failed to run: {exc}")
            else:
                logger.warning("health_check.py not found; skipping")

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.error("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
