from __future__ import annotations

from pathlib import Path
from typing import Iterable

import requests


def download_file(url: str, destination: Path) -> None:
    """Stream a remote file to a local path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

