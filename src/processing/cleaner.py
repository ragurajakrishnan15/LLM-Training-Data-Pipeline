"""
Text cleaning module for the LLM Training Data Pipeline.

Handles text normalization, encoding fixes, and cleanup.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import ftfy
from unidecode import unidecode

from src.utils import get_config, get_logger, get_metrics

logger = get_logger(__name__)


@dataclass
class CleaningStats:
    """Statistics from text cleaning."""
    
    original_length: int
    cleaned_length: int
    chars_removed: int
    encoding_fixes: int
    urls_removed: int
    emails_removed: int
    
    @property
    def reduction_ratio(self) -> float:
        """Get percentage of text removed."""
        if self.original_length == 0:
            return 0.0
        return (self.chars_removed / self.original_length) * 100


class TextCleaner:
    """
    Text cleaner for LLM training data.
    
    Performs:
    - Unicode normalization
    - Encoding fixes (mojibake, etc.)
    - URL/email removal
    - Whitespace normalization
    - Control character removal
    """
    
    # Regex patterns
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\-.~:/?#\[\]@!$&\'()*+,;=%]*'
    )
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    CITATION_PATTERN = re.compile(r'\[\d+\]|\[citation needed\]|\[note \d+\]', re.IGNORECASE)
    MULTIPLE_SPACES = re.compile(r'[ \t]+')
    MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_citations: bool = True,
        normalize_unicode: bool = True,
        fix_encoding: bool = True,
        normalize_whitespace: bool = True,
        min_length_chars: int = 100
    ):
        """
        Initialize the text cleaner.
        
        Args:
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_citations: Remove citation markers [1], etc.
            normalize_unicode: Apply Unicode normalization
            fix_encoding: Fix encoding issues (mojibake)
            normalize_whitespace: Normalize whitespace
            min_length_chars: Minimum text length after cleaning
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_citations = remove_citations
        self.normalize_unicode = normalize_unicode
        self.fix_encoding = fix_encoding
        self.normalize_whitespace = normalize_whitespace
        self.min_length_chars = min_length_chars
    
    def clean(self, text: str) -> Tuple[Optional[str], CleaningStats]:
        """
        Clean a text string.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (cleaned text or None if too short, cleaning stats)
        """
        original_length = len(text)
        urls_removed = 0
        emails_removed = 0
        encoding_fixes = 0
        
        # Fix encoding issues
        if self.fix_encoding:
            fixed_text = ftfy.fix_text(text)
            if fixed_text != text:
                encoding_fixes = 1
            text = fixed_text
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        if self.remove_urls:
            urls = self.URL_PATTERN.findall(text)
            urls_removed = len(urls)
            text = self.URL_PATTERN.sub(' ', text)
        
        # Remove emails
        if self.remove_emails:
            emails = self.EMAIL_PATTERN.findall(text)
            emails_removed = len(emails)
            text = self.EMAIL_PATTERN.sub(' ', text)
        
        # Remove citations
        if self.remove_citations:
            text = self.CITATION_PATTERN.sub('', text)
        
        # Remove control characters (except newlines and tabs)
        text = ''.join(
            char for char in text
            if unicodedata.category(char) != 'Cc' or char in '\n\t'
        )
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.MULTIPLE_SPACES.sub(' ', text)
            text = self.MULTIPLE_NEWLINES.sub('\n\n', text)
            text = text.strip()
        
        # Create stats
        cleaned_length = len(text)
        stats = CleaningStats(
            original_length=original_length,
            cleaned_length=cleaned_length,
            chars_removed=original_length - cleaned_length,
            encoding_fixes=encoding_fixes,
            urls_removed=urls_removed,
            emails_removed=emails_removed
        )
        
        # Check minimum length
        if cleaned_length < self.min_length_chars:
            return None, stats
        
        return text, stats
    
    def clean_batch(
        self,
        texts: List[str]
    ) -> Tuple[List[str], List[CleaningStats]]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (cleaned texts, stats for all texts)
        """
        cleaned_texts = []
        all_stats = []
        
        for text in texts:
            cleaned, stats = self.clean(text)
            all_stats.append(stats)
            if cleaned is not None:
                cleaned_texts.append(cleaned)
        
        return cleaned_texts, all_stats


def clean_text(text: str, **kwargs) -> Optional[str]:
    """
    Convenience function to clean text.
    
    Args:
        text: Input text
        **kwargs: Arguments passed to TextCleaner
        
    Returns:
        Cleaned text or None if too short
    """
    cleaner = TextCleaner(**kwargs)
    cleaned, _ = cleaner.clean(text)
    return cleaned


def main():
    """Main entry point for testing cleaner."""
    # Test text with various issues
    test_text = """
    This is a test article about Python programming. [1]
    
    Visit https://www.python.org for more info!
    Contact us at test@example.com for questions.
    
    Pythonâ€™s syntax is clean and readable. [citation needed]
    
    
    
    Multiple     spaces    and
    
    
    
    newlines should be normalized.
    """
    
    cleaner = TextCleaner()
    cleaned, stats = cleaner.clean(test_text)
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*60 + "\n")
    print("Cleaned text:")
    print(cleaned)
    print("\n" + "="*60 + "\n")
    print("Stats:")
    print(f"  Original length: {stats.original_length}")
    print(f"  Cleaned length: {stats.cleaned_length}")
    print(f"  Chars removed: {stats.chars_removed} ({stats.reduction_ratio:.1f}%)")
    print(f"  URLs removed: {stats.urls_removed}")
    print(f"  Emails removed: {stats.emails_removed}")
    print(f"  Encoding fixes: {stats.encoding_fixes}")


if __name__ == "__main__":
    main()
