from __future__ import annotations

import logging
from pathlib import Path
from time import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> None:
    """Stream a remote file to a local path."""
    logger.info("download_file called with URL: %s", url[:100] + "..." if len(url) > 100 else url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)


def cache_bust_url(url: str) -> str:
    """Add cache-busting parameter to URL."""
    logger.debug("cache_bust_url called with URL: %s", url[:100] + "..." if len(url) > 100 else url)
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["_"] = str(int(time() * 1000))
    result = urlunparse(parsed._replace(query=urlencode(params)))
    logger.debug("cache_bust_url result: %s", result[:100] + "..." if len(result) > 100 else result)
    return result
