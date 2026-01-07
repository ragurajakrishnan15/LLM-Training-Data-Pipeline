"""
Wikipedia dump downloader for the LLM Training Data Pipeline.

Downloads Wikipedia XML dumps for processing.
"""

import bz2
import hashlib
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from src.utils import get_config, get_logger

logger = get_logger(__name__)

# Wikipedia dump URLs
WIKI_DUMPS = {
    "simplewiki": "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2",
    "enwiki": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
}


def download_file(
    url: str,
    output_path: str,
    chunk_size: int = 8192,
    show_progress: bool = True
) -> str:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
        chunk_size: Download chunk size in bytes
        show_progress: Whether to show progress bar
        
    Returns:
        Path to downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if output_path.exists():
        logger.info(f"File already exists: {output_path}")
        return str(output_path)
    
    logger.info(f"Downloading from {url}")
    
    # Start download
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress
    with open(output_path, 'wb') as f:
        if show_progress and total_size > 0:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    
    logger.info(f"Downloaded to {output_path}")
    return str(output_path)


def decompress_bz2(
    input_path: str,
    output_path: Optional[str] = None,
    show_progress: bool = True
) -> str:
    """
    Decompress a bz2 file.
    
    Args:
        input_path: Path to bz2 file
        output_path: Path for decompressed file (default: remove .bz2 extension)
        show_progress: Whether to show progress bar
        
    Returns:
        Path to decompressed file
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = str(input_path).replace('.bz2', '')
    
    output_path = Path(output_path)
    
    # Check if already decompressed
    if output_path.exists():
        logger.info(f"Decompressed file already exists: {output_path}")
        return str(output_path)
    
    logger.info(f"Decompressing {input_path}")
    
    # Get file size for progress
    total_size = input_path.stat().st_size
    
    with bz2.BZ2File(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            if show_progress:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc="Decompressing"
                ) as pbar:
                    while True:
                        chunk = f_in.read(65536)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        # Approximate progress based on compressed size
                        pbar.update(len(chunk) // 10)
            else:
                while True:
                    chunk = f_in.read(65536)
                    if not chunk:
                        break
                    f_out.write(chunk)
    
    logger.info(f"Decompressed to {output_path}")
    return str(output_path)


def download_wikipedia(
    source: str = "simplewiki",
    output_dir: Optional[str] = None,
    decompress: bool = True
) -> str:
    """
    Download a Wikipedia dump.
    
    Args:
        source: Wikipedia source (simplewiki, enwiki, or custom URL)
        output_dir: Directory to save downloaded file
        decompress: Whether to decompress the file
        
    Returns:
        Path to the downloaded (and optionally decompressed) file
    """
    config = get_config()
    
    # Determine URL
    if source in WIKI_DUMPS:
        url = WIKI_DUMPS[source]
    elif source.startswith("http"):
        url = source
    else:
        raise ValueError(f"Unknown source: {source}. Use: {list(WIKI_DUMPS.keys())}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = config.paths.get("raw_data", "data/raw")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = Path(urlparse(url).path).name
    output_path = output_dir / filename
    
    # Download
    downloaded_path = download_file(url, str(output_path))
    
    # Decompress if requested
    if decompress and downloaded_path.endswith('.bz2'):
        return decompress_bz2(downloaded_path)
    
    return downloaded_path


def main():
    """Main entry point for Wikipedia download."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Wikipedia dumps")
    parser.add_argument(
        "--source",
        type=str,
        default="simplewiki",
        help="Wikipedia source (simplewiki, enwiki, or URL)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--no-decompress",
        action="store_true",
        help="Don't decompress the downloaded file"
    )
    
    args = parser.parse_args()
    
    output_path = download_wikipedia(
        source=args.source,
        output_dir=args.output_dir,
        decompress=not args.no_decompress
    )
    
    logger.info(f"Wikipedia dump ready: {output_path}")


if __name__ == "__main__":
    main()
