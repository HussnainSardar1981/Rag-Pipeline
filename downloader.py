"""
MS Graph API PDF Downloader for SharePoint
Downloads PDFs from /General/Knowledgebase/Shared with Customer/{CustomerName}/How - To/
Creates customer-isolated folders on server with ETag-based change detection
"""

import requests
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths_config import PathsConfig
from cli_common import add_standard_args, build_selection, configure_logging

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[WARN] python-dotenv not installed. Install with: pip install python-dotenv")

logger = logging.getLogger(__name__)


class MicrosoftGraphDownloader:
    """Download PDFs from SharePoint using MS Graph API"""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        sharepoint_site_url: str,
        kb_path: str = "/General/Knowledgebase/Shared with Customer",
        subfolder: str = "How - To"
    ):
        """Initialize MS Graph API client"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.sharepoint_site_url = sharepoint_site_url
        self.kb_path = kb_path
        self.subfolder = subfolder

        # MS Graph endpoints
        self.token_endpoint = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"

        self.access_token = None
        self.token_expiry = None

    def get_access_token(self) -> str:
        """Get access token from Azure AD using client credentials flow"""
        print("[AUTH] Requesting access token from Azure AD...")

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "resource": "https://graph.microsoft.com",
            "grant_type": "client_credentials"
        }

        try:
            response = requests.post(self.token_endpoint, data=payload, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data['access_token']
            expires_in = int(token_data['expires_in'])
            self.token_expiry = datetime.now().timestamp() + expires_in

            print(f"[OK] Access token obtained (expires in {expires_in} seconds)")
            return self.access_token

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get access token: {e}")
            raise

    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with Authorization"""
        if not self.access_token:
            self.get_access_token()

        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def get_site_id(self) -> str:
        """Get SharePoint site ID"""
        print(f"[API] Fetching site ID for {self.sharepoint_site_url}...")

        # Extract hostname from URL
        # Example: https://mad4it.sharepoint.com → mad4it.sharepoint.com
        hostname = self.sharepoint_site_url.replace("https://", "").replace("http://", "")

        # The site name is sophia_do_not_reply
        url = f"{self.graph_endpoint}/sites/{hostname}:/sites/sophia_do_not_reply"

        try:
            response = requests.get(url, headers=self.get_headers(), timeout=10)
            response.raise_for_status()

            site_data = response.json()
            site_id = site_data['id']

            print(f"[OK] Site ID: {site_id}")
            return site_id

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get site ID: {e}")
            raise

    def get_folder_id(self, site_id: str, folder_path: str) -> Optional[str]:
        """Get folder ID from path"""
        print(f"[API] Fetching folder ID for: {folder_path}")

        url = f"{self.graph_endpoint}/sites/{site_id}/drive/root:/{folder_path}"

        try:
            response = requests.get(url, headers=self.get_headers(), timeout=10)

            if response.status_code == 404:
                logger.warning(f"Folder not found: {folder_path}")
                return None

            response.raise_for_status()

            folder_data = response.json()
            folder_id = folder_data['id']

            print(f"[OK] Folder ID: {folder_id}")
            return folder_id

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get folder ID: {e}")
            return None

    def list_items_in_folder(self, site_id: str, folder_id: str) -> List[Dict]:
        """List all items in a folder"""
        print(f"[API] Listing items in folder: {folder_id}")

        url = f"{self.graph_endpoint}/sites/{site_id}/drive/items/{folder_id}/children"
        all_items = []

        try:
            while url:
                response = requests.get(url, headers=self.get_headers(), timeout=10)
                response.raise_for_status()

                data = response.json()
                items = data.get('value', [])
                all_items.extend(items)

                # Handle pagination
                url = data.get('@odata.nextLink', None)

            print(f"[OK] Found {len(all_items)} items")
            return all_items

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list items: {e}")
            return []

    def list_customers(self, site_id: str) -> List[str]:
        """List all customer folders in /Shared with Customer/"""
        print("\n[DISCOVER] Discovering customer folders...")
        print("=" * 70)

        kb_folder_id = self.get_folder_id(site_id, self.kb_path)

        if not kb_folder_id:
            logger.error(f"Cannot find KB folder: {self.kb_path}")
            return []

        items = self.list_items_in_folder(site_id, kb_folder_id)

        customers = []
        for item in items:
            if item.get('folder'):  # It's a folder
                customer_name = item['name']
                customers.append(customer_name)
                print(f"  - {customer_name}")

        print("=" * 70)
        print(f"[OK] Discovered {len(customers)} customers")

        return customers

    def list_pdfs_for_customer(
        self,
        site_id: str,
        customer_name: str
    ) -> List[Dict]:
        """List all PDFs in customer's How-To folder"""
        customer_kb_path = f"{self.kb_path}/{customer_name}/{self.subfolder}"

        print(f"[API] Listing PDFs for customer: {customer_name}")

        folder_id = self.get_folder_id(site_id, customer_kb_path)

        if not folder_id:
            logger.warning(f"No {self.subfolder} folder for {customer_name}")
            return []

        items = self.list_items_in_folder(site_id, folder_id)

        # Filter only PDFs
        pdfs = [item for item in items if item['name'].lower().endswith('.pdf')]

        print(f"[OK] Found {len(pdfs)} PDFs for {customer_name}")

        return pdfs

    def download_file(
        self,
        site_id: str,
        item_id: str,
        save_path: Path
    ) -> bool:
        """Download file from SharePoint"""
        url = f"{self.graph_endpoint}/sites/{site_id}/drive/items/{item_id}/content"

        try:
            response = requests.get(url, headers=self.get_headers(), timeout=30, stream=True)
            response.raise_for_status()

            # Save file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = save_path.stat().st_size
            print(f"    [OK] Downloaded: {save_path.name} ({file_size:,} bytes)")

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file: {e}")
            return False

    def download_customer_pdfs(
        self,
        site_id: str,
        customer_name: str,
        server_root: Path,
        force_redownload: bool = False
    ) -> Dict:
        """Download all PDFs for a customer"""
        print(f"\n[DOWNLOAD] Starting download for customer: {customer_name}")
        print("=" * 70)

        # Create customer folder
        customer_id = customer_name.lower().replace(" ", "_")
        customer_dir = server_root / "customers" / customer_id
        pdfs_dir = customer_dir / "pdfs"
        pdfs_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata file
        kb_metadata_path = customer_dir / "kb_metadata.json"
        if kb_metadata_path.exists():
            with open(kb_metadata_path, 'r', encoding='utf-8') as f:
                kb_metadata = json.load(f)
        else:
            kb_metadata = {
                "customer_id": customer_id,
                "customer_name": customer_name,
                "created_at": datetime.now().isoformat(),
                "files": {}
            }

        # Get PDFs from SharePoint
        pdfs = self.list_pdfs_for_customer(site_id, customer_name)

        if not pdfs:
            print(f"[WARN] No PDFs found for customer: {customer_name}")
            return {
                "customer_id": customer_id,
                "total_pdfs": 0,
                "downloaded": 0,
                "skipped": 0,
                "failed": 0,
                "status": "no_pdfs"
            }

        downloaded_count = 0
        skipped_count = 0
        failed_count = 0

        for pdf_item in pdfs:
            pdf_name = pdf_item['name']
            item_id = pdf_item['id']
            etag = pdf_item.get('eTag', '')
            modified = pdf_item.get('lastModifiedDateTime', '')

            pdf_path = pdfs_dir / pdf_name

            # Check if should download
            should_download = force_redownload

            if not should_download:
                if not pdf_path.exists():
                    should_download = True
                    reason = "New PDF"
                else:
                    # Check ETag
                    existing_etag = kb_metadata.get('files', {}).get(pdf_name, {}).get('etag', '')
                    if etag and existing_etag != etag:
                        should_download = True
                        reason = "Modified (ETag changed)"
                    else:
                        reason = "Already downloaded"

            if should_download:
                print(f"  [{reason}] {pdf_name}")
                if self.download_file(site_id, item_id, pdf_path):
                    # Update metadata
                    kb_metadata['files'][pdf_name] = {
                        'etag': etag,
                        'last_modified': modified,
                        'downloaded_at': datetime.now().isoformat(),
                        'file_size': pdf_path.stat().st_size
                    }
                    downloaded_count += 1
                else:
                    failed_count += 1
            else:
                print(f"  [SKIP] {pdf_name} ({reason})")
                skipped_count += 1

        # Save updated metadata
        with open(kb_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(kb_metadata, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print(f"[SUMMARY] {customer_name}")
        print(f"  Downloaded: {downloaded_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(pdfs)}")
        print(f"  Location: {pdfs_dir}")

        return {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "total_pdfs": len(pdfs),
            "downloaded": downloaded_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "status": "success" if failed_count == 0 else "partial"
        }

    def download_all_customers(
        self,
        server_root: Path,
        customer_filter: Optional[List[str]] = None,
        force_redownload: bool = False
    ) -> Dict:
        """Download PDFs for all customers"""
        print("\n" + "=" * 70)
        print("MS GRAPH PDF DOWNLOADER")
        print("=" * 70)

        # Get access token
        self.get_access_token()

        # Get site ID
        site_id = self.get_site_id()

        # List all customers
        all_customers = self.list_customers(site_id)

        # Filter if specified
        if customer_filter:
            # Case-insensitive matching: match by customer name or customer_id
            filtered_customers = []
            for requested in customer_filter:
                requested_lower = requested.lower()
                requested_id = requested.lower().replace(" ", "_")

                for customer_name in all_customers:
                    customer_id = customer_name.lower().replace(" ", "_")
                    # Match by full name, name prefix, or customer_id
                    if (customer_name.lower() == requested_lower or
                        customer_id == requested_id or
                        customer_name.lower().startswith(requested_lower)):
                        if customer_name not in filtered_customers:
                            filtered_customers.append(customer_name)
                        break

            customers = filtered_customers
            skipped = set(all_customers) - set(customers)
            if skipped:
                print(f"\n[INFO] Filtered customers. Skipping: {', '.join(skipped)}")

            if not customers:
                print(f"\n[WARN] No customers matched filter: {', '.join(customer_filter)}")
                print(f"[INFO] Available customers: {', '.join(all_customers)}")
        else:
            customers = all_customers

        # Download for each customer
        results = {}
        for customer_name in customers:
            result = self.download_customer_pdfs(
                site_id,
                customer_name,
                server_root,
                force_redownload=force_redownload
            )
            results[customer_name] = result

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL DOWNLOAD SUMMARY")
        print("=" * 70)

        total_downloaded = sum(r['downloaded'] for r in results.values())
        total_failed = sum(r['failed'] for r in results.values())

        for customer_name, result in results.items():
            status_str = "OK" if result['status'] == 'success' else "PARTIAL"
            print(f"  {customer_name:30} [{status_str}] {result['downloaded']}/{result['total_pdfs']} downloaded")

        print(f"\nTotal downloaded: {total_downloaded}")
        if total_failed > 0:
            print(f"Total failed: {total_failed}")

        print("\n" + "=" * 70)

        return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Download PDFs from SharePoint for RAG pipeline'
    )
    add_standard_args(parser, include_file=True, include_customer_id=False, include_server_root=True)
    parser.add_argument('--force', action='store_true', help='Force re-download all files')

    args = parser.parse_args()
    selection = build_selection(args)

    if selection.file:
        print("[ERROR] --file is not supported for downloader.")
        print("Use --customer/--customers or --all instead.")
        sys.exit(1)

    configure_logging("downloader", server_root=selection.server_root)

    # Load environment variables
    if DOTENV_AVAILABLE:
        load_dotenv()
    else:
        print("[WARN] .env file will not be loaded automatically")
        print("[INFO] Set environment variables manually:")
        print("  GRAPH_CLIENT_ID, GRAPH_CLIENT_SECRET, GRAPH_TENANT_ID")
        print("  SHAREPOINT_SITE_URL, SHAREPOINT_KB_PATH, SHAREPOINT_SUBFOLDER")

    # Get credentials from environment
    client_id = os.getenv('GRAPH_CLIENT_ID')
    client_secret = os.getenv('GRAPH_CLIENT_SECRET')
    tenant_id = os.getenv('GRAPH_TENANT_ID')
    sharepoint_site_url = os.getenv('SHAREPOINT_SITE_URL', 'https://mad4it.sharepoint.com')
    kb_path = os.getenv('SHAREPOINT_KB_PATH', '/General/Knowledgebase/Shared with Customer')
    subfolder = os.getenv('SHAREPOINT_SUBFOLDER', 'How - To')
    server_root = selection.server_root

    # Validate credentials
    if not all([client_id, client_secret, tenant_id]):
        print("[ERROR] Missing required environment variables:")
        print("  GRAPH_CLIENT_ID, GRAPH_CLIENT_SECRET, GRAPH_TENANT_ID")
        print("\nSet them in .env file or as environment variables")
        sys.exit(1)

    # Initialize downloader
    downloader = MicrosoftGraphDownloader(
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        sharepoint_site_url=sharepoint_site_url,
        kb_path=kb_path,
        subfolder=subfolder
    )

    # Download PDFs for specified customers
    server_root = Path(server_root)

    # Determine which customers to process
    customers = selection.customer_filter()
    if customers:
        print(f"[INFO] Filtering to customers: {', '.join(customers)}")
    else:
        print("[INFO] Processing all customers")

    results = downloader.download_all_customers(
        server_root=server_root,
        customer_filter=customers,
        force_redownload=args.force
    )

    # Exit with status
    failed = sum(1 for r in results.values() if r['status'] != 'success')
    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
