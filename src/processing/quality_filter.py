"""
Quality filtering module for the LLM Training Data Pipeline.

Filters documents based on various quality heuristics.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from langdetect import detect, LangDetectException
from tqdm import tqdm

from src.utils import get_config, get_logger, get_metrics

logger = get_logger(__name__)


class FilterReason(Enum):
    """Reasons for filtering a document."""
    
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    WRONG_LANGUAGE = "wrong_language"
    LOW_ALPHA_RATIO = "low_alpha_ratio"
    HIGH_DIGIT_RATIO = "high_digit_ratio"
    HIGH_SYMBOL_RATIO = "high_symbol_ratio"
    SHORT_AVG_WORD = "short_avg_word_length"
    LONG_AVG_WORD = "long_avg_word_length"
    HIGH_REPETITION = "high_repetition"
    BULLET_LIST = "mostly_bullet_list"
    BOILERPLATE = "boilerplate_content"
    PASSED = "passed"


@dataclass
class QualityResult:
    """Result of quality check for a document."""
    
    passed: bool
    reason: FilterReason
    scores: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.passed:
            return "PASSED"
        return f"FILTERED: {self.reason.value}"


@dataclass
class QualityStats:
    """Statistics from quality filtering."""
    
    total_documents: int
    passed_documents: int
    filter_counts: Dict[FilterReason, int] = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        """Get percentage of documents that passed."""
        if self.total_documents == 0:
            return 0.0
        return (self.passed_documents / self.total_documents) * 100


class QualityFilter:
    """
    Quality filter for LLM training data.
    
    Implements various heuristics used in production LLM data pipelines:
    - Length filtering (too short/long)
    - Language detection
    - Character composition (alphabetic, digit, symbol ratios)
    - Word length statistics
    - Repetition detection
    - Boilerplate detection
    """
    
    # Common boilerplate patterns
    BOILERPLATE_PATTERNS = [
        r'copyright \d{4}',
        r'all rights reserved',
        r'terms of service',
        r'privacy policy',
        r'cookie policy',
        r'subscribe to our newsletter',
        r'click here to',
        r'share on facebook',
        r'follow us on twitter',
    ]
    
    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 100000,
        min_avg_word_length: float = 3.0,
        max_avg_word_length: float = 15.0,
        min_alpha_ratio: float = 0.7,
        max_digit_ratio: float = 0.3,
        max_symbol_ratio: float = 0.2,
        max_repetition_ratio: float = 0.3,
        allowed_languages: Optional[List[str]] = None,
        language_confidence: float = 0.9,
        check_boilerplate: bool = True,
        max_bullet_ratio: float = 0.5
    ):
        """
        Initialize the quality filter.
        
        Args:
            min_words: Minimum word count
            max_words: Maximum word count
            min_avg_word_length: Minimum average word length
            max_avg_word_length: Maximum average word length
            min_alpha_ratio: Minimum ratio of alphabetic characters
            max_digit_ratio: Maximum ratio of digit characters
            max_symbol_ratio: Maximum ratio of special characters
            max_repetition_ratio: Maximum ratio of repeated lines
            allowed_languages: List of allowed language codes (None = all)
            language_confidence: Minimum confidence for language detection
            check_boilerplate: Whether to check for boilerplate content
            max_bullet_ratio: Maximum ratio of lines starting with bullets
        """
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.min_alpha_ratio = min_alpha_ratio
        self.max_digit_ratio = max_digit_ratio
        self.max_symbol_ratio = max_symbol_ratio
        self.max_repetition_ratio = max_repetition_ratio
        self.allowed_languages = allowed_languages
        self.language_confidence = language_confidence
        self.check_boilerplate = check_boilerplate
        self.max_bullet_ratio = max_bullet_ratio
        
        # Compile boilerplate patterns
        self.boilerplate_regex = re.compile(
            '|'.join(self.BOILERPLATE_PATTERNS),
            re.IGNORECASE
        )
        
        logger.info(
            f"Initialized quality filter: min_words={min_words}, "
            f"max_words={max_words}, languages={allowed_languages}"
        )
    
    def check(self, text: str) -> QualityResult:
        """
        Check if a document passes quality filters.
        
        Args:
            text: Document text
            
        Returns:
            QualityResult with pass/fail status and reason
        """
        scores = {}
        
        # Word count check
        words = text.split()
        word_count = len(words)
        scores['word_count'] = word_count
        
        if word_count < self.min_words:
            return QualityResult(False, FilterReason.TOO_SHORT, scores)
        
        if word_count > self.max_words:
            return QualityResult(False, FilterReason.TOO_LONG, scores)
        
        # Average word length
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            scores['avg_word_length'] = avg_word_length
            
            if avg_word_length < self.min_avg_word_length:
                return QualityResult(False, FilterReason.SHORT_AVG_WORD, scores)
            
            if avg_word_length > self.max_avg_word_length:
                return QualityResult(False, FilterReason.LONG_AVG_WORD, scores)
        
        # Character composition
        total_chars = len(text)
        if total_chars > 0:
            alpha_count = sum(1 for c in text if c.isalpha())
            digit_count = sum(1 for c in text if c.isdigit())
            space_count = sum(1 for c in text if c.isspace())
            symbol_count = total_chars - alpha_count - digit_count - space_count
            
            # Calculate ratios excluding spaces
            non_space = total_chars - space_count
            if non_space > 0:
                alpha_ratio = alpha_count / non_space
                digit_ratio = digit_count / non_space
                symbol_ratio = symbol_count / non_space
                
                scores['alpha_ratio'] = alpha_ratio
                scores['digit_ratio'] = digit_ratio
                scores['symbol_ratio'] = symbol_ratio
                
                if alpha_ratio < self.min_alpha_ratio:
                    return QualityResult(False, FilterReason.LOW_ALPHA_RATIO, scores)
                
                if digit_ratio > self.max_digit_ratio:
                    return QualityResult(False, FilterReason.HIGH_DIGIT_RATIO, scores)
                
                if symbol_ratio > self.max_symbol_ratio:
                    return QualityResult(False, FilterReason.HIGH_SYMBOL_RATIO, scores)
        
        # Repetition check
        lines = text.split('\n')
        if len(lines) > 1:
            line_counts = Counter(line.strip() for line in lines if line.strip())
            if line_counts:
                most_common_count = line_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / len(lines)
                scores['repetition_ratio'] = repetition_ratio
                
                if repetition_ratio > self.max_repetition_ratio:
                    return QualityResult(False, FilterReason.HIGH_REPETITION, scores)
        
        # Bullet list check
        bullet_pattern = re.compile(r'^[\s]*[-•*►▪▸]\s')
        bullet_lines = sum(1 for line in lines if bullet_pattern.match(line))
        if len(lines) > 0:
            bullet_ratio = bullet_lines / len(lines)
            scores['bullet_ratio'] = bullet_ratio
            
            if bullet_ratio > self.max_bullet_ratio:
                return QualityResult(False, FilterReason.BULLET_LIST, scores)
        
        # Boilerplate check
        if self.check_boilerplate:
            boilerplate_matches = len(self.boilerplate_regex.findall(text))
            scores['boilerplate_matches'] = boilerplate_matches
            
            if boilerplate_matches > 3:
                return QualityResult(False, FilterReason.BOILERPLATE, scores)
        
        # Language check
        if self.allowed_languages:
            try:
                detected_lang = detect(text[:1000])  # Use first 1000 chars
                scores['detected_language'] = detected_lang
                
                if detected_lang not in self.allowed_languages:
                    return QualityResult(False, FilterReason.WRONG_LANGUAGE, scores)
            except LangDetectException:
                # Can't detect language, let it pass
                scores['detected_language'] = 'unknown'
        
        return QualityResult(True, FilterReason.PASSED, scores)
    
    def filter_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> Tuple[List[str], QualityStats]:
        """
        Filter a batch of texts.
        
        Args:
            texts: List of texts to filter
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (passed texts, quality stats)
        """
        passed_texts = []
        filter_counts: Dict[FilterReason, int] = {reason: 0 for reason in FilterReason}
        
        iterator = tqdm(texts, desc="Quality filtering") if show_progress else texts
        
        for text in iterator:
            result = self.check(text)
            filter_counts[result.reason] += 1
            
            if result.passed:
                passed_texts.append(text)
        
        stats = QualityStats(
            total_documents=len(texts),
            passed_documents=len(passed_texts),
            filter_counts=filter_counts
        )
        
        logger.info(
            f"Quality filtering complete: {stats.passed_documents:,}/{stats.total_documents:,} "
            f"passed ({stats.pass_rate:.1f}%)"
        )
        
        return passed_texts, stats


def main():
    """Main entry point for testing quality filter."""
    # Test texts
    test_texts = [
        # Good text
        """
        Python is a high-level, general-purpose programming language. 
        Its design philosophy emphasizes code readability with the use of 
        significant indentation. Python is dynamically typed and garbage-collected. 
        It supports multiple programming paradigms, including structured, 
        object-oriented and functional programming.
        """,
        
        # Too short
        "Hello world.",
        
        # Too many digits
        "12345 67890 12345 67890 12345 67890 12345 67890 text here 12345",
        
        # High repetition
        "This is repeated.\n" * 20,
        
        # Bullet list
        "• Item one\n• Item two\n• Item three\n• Item four\n• Item five\n" * 5,
        
        # Boilerplate
        """
        Copyright 2024. All rights reserved. Terms of service apply.
        Privacy policy and cookie policy. Subscribe to our newsletter!
        Click here to learn more. Share on Facebook. Follow us on Twitter.
        """,
    ]
    
    print("Testing Quality Filter")
    print("="*60)
    
    quality_filter = QualityFilter(
        min_words=20,
        allowed_languages=["en"]
    )
    
    for i, text in enumerate(test_texts):
        result = quality_filter.check(text)
        preview = text[:50].replace('\n', ' ')
        print(f"\nText {i+1}: '{preview}...'")
        print(f"  Result: {result}")
        if result.scores:
            for key, value in result.scores.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
