"""
Shared CLI helpers for RAG pipeline scripts.
Provides a consistent argument pattern across stages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging

from paths_config import PathsConfig


@dataclass
class RagSelection:
    """Normalized selection for RAG CLI commands."""
    file: Optional[Path]
    customer: Optional[str]
    customers: Optional[List[str]]
    all_customers: bool
    server_root: Path
    customer_id: Optional[str] = None

    def customer_filter(self) -> Optional[List[str]]:
        """Return a list of customer IDs/names to process, or None for all."""
        if self.customers:
            return self.customers
        if self.customer:
            return [self.customer]
        if self.all_customers:
            return None
        return None


def add_standard_args(
    parser,
    include_file: bool = True,
    include_customer_id: bool = False,
    include_server_root: bool = True
):
    """Add standard RAG selection arguments to an argparse parser."""
    parser.add_argument('--all', action='store_true', help='Process all customers')
    parser.add_argument('--customer', type=str, help='Process a single customer')
    parser.add_argument('--customers', nargs='+', help='Process multiple customers')

    if include_file:
        parser.add_argument('--file', type=Path, help='Process a single input file')

    if include_customer_id:
        parser.add_argument('--customer-id', type=str,
                            help='Customer ID (used with --file when not in path)')

    if include_server_root:
        parser.add_argument('--server-root', default=str(PathsConfig.RAG_DIR),
                            help='RAG root directory')

    return parser


def build_selection(args) -> RagSelection:
    """Normalize parsed args into a RagSelection."""
    file_arg = getattr(args, 'file', None)
    has_file = file_arg is not None

    if not args.all and not args.customer and not args.customers and not has_file:
        args.all = True

    server_root = Path(getattr(args, 'server_root', PathsConfig.RAG_DIR))
    file_path = Path(file_arg) if file_arg else None

    return RagSelection(
        file=file_path,
        customer=getattr(args, 'customer', None),
        customers=getattr(args, 'customers', None),
        all_customers=args.all,
        server_root=server_root,
        customer_id=getattr(args, 'customer_id', None)
    )


def configure_logging(stage: str, server_root: Optional[Path] = None,
                      level: int = logging.INFO) -> None:
    """Configure consistent logging for RAG scripts."""
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    if server_root:
        logs_dir = Path(server_root) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(logs_dir / f"{stage}.log"), encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
