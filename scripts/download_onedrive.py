"""
Download the shared OneDrive workbook before starting the Dash app.

Set `ONEDRIVE_DOWNLOAD_URL` to the share link (make sure the link ends with `download=1`
or re-run it through OneDrive's "copy link" dialog) and optionally override `DATA_PATH`
so the file lands where `app.py` can read it.
"""

import os
from pathlib import Path

from data_sync import download_file


def main() -> None:
    source_url = os.environ.get("ONEDRIVE_DOWNLOAD_URL")
    if not source_url:
        raise SystemExit(
            "Set ONEDRIVE_DOWNLOAD_URL (e.g., https://1drv.ms/...) before running this script."
        )

    target_path = Path(os.environ.get("DATA_PATH", "LLU Imaging AI 2025.xlsx"))
    print(f"Downloading OneDrive workbook to {target_path} ...")
    download_file(source_url, target_path)
    print("Download complete.")


if __name__ == "__main__":
    main()

