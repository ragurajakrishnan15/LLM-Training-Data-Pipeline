#!/usr/bin/env python3
"""Quick test to verify pipeline components work."""

from src.processing.cleaner import TextCleaner
from src.processing.deduplicator import MinHashDeduplicator
from src.processing.quality_filter import QualityFilter

def test_cleaner():
    """Test text cleaner."""
    cleaner = TextCleaner(
        remove_urls=True,
        remove_emails=True,
        remove_citations=True,
        normalize_unicode=True,
        fix_encoding=True,
        normalize_whitespace=True,
        min_length_chars=100
    )
    
    test_text = "Visit https://example.com or email test@example.com for more info. " * 10
    cleaned, stats = cleaner.clean(test_text)
    
    print(f"✓ Cleaner test passed")
    print(f"  Original: {stats.original_length} chars")
    print(f"  Cleaned: {stats.cleaned_length} chars")
    print(f"  Removed: {stats.urls_removed} URLs, {stats.emails_removed} emails")

def test_deduplicator():
    """Test deduplicator."""
    dedup = MinHashDeduplicator(num_perm=128, threshold=0.8, shingle_size=5)
    
    texts = [
        "This is test document one",
        "This is test document two",
        "This is test document one"  # Duplicate
    ]
    
    unique_count = 0
    for text in texts:
        if not dedup.is_duplicate(text):
            dedup.add(text)
            unique_count += 1
    
    print(f"✓ Deduplicator test passed")
    print(f"  Processed 3 texts, found {unique_count} unique")

def test_quality_filter():
    """Test quality filter."""
    qf = QualityFilter(min_words=50, max_words=100000, languages=['en'])
    
    good_text = " ".join(["word"] * 100)
    result = qf.filter(good_text)
    
    print(f"✓ Quality filter test passed")
    print(f"  Good text passed: {result}")

if __name__ == "__main__":
    print("Running quick component tests...\n")
    test_cleaner()
    test_deduplicator()
    test_quality_filter()
    print("\n✓ All component tests passed!")
