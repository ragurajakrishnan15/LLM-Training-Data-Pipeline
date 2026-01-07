"""
Wikipedia XML dump parser for the LLM Training Data Pipeline.

Parses Wikipedia XML dumps and extracts article content.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import mwparserfromhell
from tqdm import tqdm

from src.utils import get_config, get_logger, get_metrics

logger = get_logger(__name__)


@dataclass
class WikiArticle:
    """Represents a parsed Wikipedia article."""
    
    title: str
    text: str
    page_id: int
    namespace: int
    redirect: Optional[str] = None
    
    @property
    def is_redirect(self) -> bool:
        """Check if article is a redirect."""
        return self.redirect is not None
    
    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.text)


class WikipediaParser:
    """
    Parser for Wikipedia XML dumps.
    
    Uses streaming XML parsing to handle large files efficiently.
    """
    
    # XML namespaces (supports both 0.10 and 0.11)
    NAMESPACE = "{http://www.mediawiki.org/xml/export-0.11/}"
    NAMESPACE_OLD = "{http://www.mediawiki.org/xml/export-0.10/}"
    
    # Namespace IDs to include (0 = main articles)
    ALLOWED_NAMESPACES = {0}
    
    def __init__(self, remove_markup: bool = True):
        """
        Initialize the parser.
        
        Args:
            remove_markup: Whether to remove wiki markup from text
        """
        self.remove_markup = remove_markup
        self.config = get_config()
    
    def parse_file(
        self,
        filepath: str,
        max_articles: Optional[int] = None,
        show_progress: bool = True
    ) -> Generator[WikiArticle, None, None]:
        """
        Parse a Wikipedia XML dump file.
        
        Args:
            filepath: Path to XML file
            max_articles: Maximum number of articles to parse
            show_progress: Whether to show progress bar
            
        Yields:
            WikiArticle objects
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Parsing Wikipedia dump: {filepath}")
        
        # Get file size for progress estimation
        file_size = filepath.stat().st_size
        
        articles_parsed = 0
        articles_yielded = 0
        
        # Use iterparse for memory-efficient parsing
        context = ET.iterparse(str(filepath), events=('end',))
        
        if show_progress:
            pbar = tqdm(desc="Parsing articles", unit=" articles")
        
        try:
            for event, elem in context:
                # Look for page elements (support both 0.10 and 0.11 namespaces)
                if elem.tag in (f"{self.NAMESPACE}page", f"{self.NAMESPACE_OLD}page", "page"):
                    article = self._parse_page(elem)
                    articles_parsed += 1
                    
                    if article is not None:
                        articles_yielded += 1
                        
                        if show_progress:
                            pbar.update(1)
                        
                        yield article
                        
                        if max_articles and articles_yielded >= max_articles:
                            logger.info(f"Reached max articles limit: {max_articles}")
                            break
                    
                    # Clear element to free memory
                    elem.clear()
        
        finally:
            if show_progress:
                pbar.close()
        
        logger.info(
            f"Parsed {articles_parsed:,} pages, "
            f"yielded {articles_yielded:,} articles"
        )
    
    def _parse_page(self, page_elem: ET.Element) -> Optional[WikiArticle]:
        """
        Parse a single page element.
        
        Args:
            page_elem: XML page element
            
        Returns:
            WikiArticle or None if page should be skipped
        """
        # Extract basic info
        title = self._get_text(page_elem, "title")
        page_id = int(self._get_text(page_elem, "id") or 0)
        ns = int(self._get_text(page_elem, "ns") or 0)
        
        # Skip non-article namespaces
        if ns not in self.ALLOWED_NAMESPACES:
            return None
        
        # Check for redirect (try both namespaces)
        redirect_elem = page_elem.find(f"{self.NAMESPACE}redirect")
        if redirect_elem is None:
            redirect_elem = page_elem.find(f"{self.NAMESPACE_OLD}redirect")
        if redirect_elem is None:
            redirect_elem = page_elem.find("redirect")
        redirect = redirect_elem.get("title") if redirect_elem is not None else None
        
        # Skip redirects
        if redirect is not None:
            return None
        
        # Get revision text (try both namespaces)
        revision = page_elem.find(f"{self.NAMESPACE}revision")
        if revision is None:
            revision = page_elem.find(f"{self.NAMESPACE_OLD}revision")
        if revision is None:
            revision = page_elem.find("revision")
        if revision is None:
            return None
        
        text_elem = revision.find(f"{self.NAMESPACE}text")
        if text_elem is None:
            text_elem = revision.find(f"{self.NAMESPACE_OLD}text")
        if text_elem is None:
            text_elem = revision.find("text")
        if text_elem is None or text_elem.text is None:
            return None
        
        raw_text = text_elem.text
        
        # Process text
        if self.remove_markup:
            text = self._remove_wiki_markup(raw_text)
        else:
            text = raw_text
        
        # Skip empty articles
        if not text or len(text.strip()) < 50:
            return None
        
        return WikiArticle(
            title=title,
            text=text,
            page_id=page_id,
            namespace=ns,
            redirect=redirect
        )
    
    def _get_text(self, elem: ET.Element, tag: str) -> Optional[str]:
        """Get text content of a child element (try both namespaces)."""
        child = elem.find(f"{self.NAMESPACE}{tag}")
        if child is None:
            child = elem.find(f"{self.NAMESPACE_OLD}{tag}")
        if child is None:
            child = elem.find(tag)
        return child.text if child is not None else None
    
    def _remove_wiki_markup(self, text: str) -> str:
        """
        Remove Wikipedia markup from text.
        
        Args:
            text: Raw wiki text
            
        Returns:
            Clean plain text
        """
        try:
            # Parse wiki markup
            wikicode = mwparserfromhell.parse(text)
            
            # Remove templates, tables, etc.
            for template in wikicode.filter_templates():
                try:
                    wikicode.remove(template)
                except ValueError:
                    pass
            
            # Get plain text
            text = wikicode.strip_code()
            
        except Exception as e:
            # Fallback: basic regex cleaning
            logger.debug(f"mwparserfromhell failed, using regex: {e}")
            text = self._regex_clean(text)
        
        # Additional cleaning
        text = self._clean_text(text)
        
        return text
    
    def _regex_clean(self, text: str) -> str:
        """Fallback regex-based cleaning."""
        # Remove templates {{...}}
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        
        # Remove references <ref>...</ref>
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^/>]*/>', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki links [[...]]
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        
        # Remove external links [http...]
        text = re.sub(r'\[https?://[^\]]+\]', '', text)
        
        # Remove categories
        text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
        
        # Remove bold/italic markup
        text = re.sub(r"'{2,5}", '', text)
        
        # Remove headings
        text = re.sub(r'^=+\s*([^=]+)\s*=+$', r'\1', text, flags=re.MULTILINE)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Final text cleaning."""
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove empty parentheses
        text = re.sub(r'\(\s*\)', '', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


def parse_wikipedia(
    filepath: str,
    max_articles: Optional[int] = None
) -> Generator[WikiArticle, None, None]:
    """
    Convenience function to parse Wikipedia dump.
    
    Args:
        filepath: Path to Wikipedia XML dump
        max_articles: Maximum articles to parse
        
    Yields:
        WikiArticle objects
    """
    parser = WikipediaParser()
    yield from parser.parse_file(filepath, max_articles=max_articles)


def main():
    """Main entry point for testing parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse Wikipedia XML dump")
    parser.add_argument("filepath", help="Path to Wikipedia XML dump")
    parser.add_argument(
        "--max-articles",
        type=int,
        default=100,
        help="Maximum articles to parse"
    )
    
    args = parser.parse_args()
    
    wiki_parser = WikipediaParser()
    
    for article in wiki_parser.parse_file(
        args.filepath,
        max_articles=args.max_articles
    ):
        print(f"\n{'='*60}")
        print(f"Title: {article.title}")
        print(f"ID: {article.page_id}")
        print(f"Words: {article.word_count}")
        print(f"{'='*60}")
        print(article.text[:500] + "..." if len(article.text) > 500 else article.text)


if __name__ == "__main__":
    main()
