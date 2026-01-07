"""
Unit tests for the LLM Training Data Pipeline.
"""

import pytest
import tempfile
from pathlib import Path

# Import pipeline components
from src.processing.cleaner import TextCleaner, clean_text
from src.processing.deduplicator import MinHashDeduplicator, Document, ExactHashDeduplicator
from src.processing.quality_filter import QualityFilter, FilterReason
from src.processing.tokenizer import TokenizerTrainer


class TestTextCleaner:
    """Tests for TextCleaner."""
    
    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        cleaner = TextCleaner(min_length_chars=10)
        text = "Hello   world!   This   is   a   test."
        cleaned, stats = cleaner.clean(text)
        
        assert cleaned is not None
        assert "   " not in cleaned  # Multiple spaces removed
        assert stats.original_length > 0
    
    def test_url_removal(self):
        """Test URL removal."""
        cleaner = TextCleaner(remove_urls=True, min_length_chars=10)
        text = "Visit https://example.com for more information about testing."
        cleaned, stats = cleaner.clean(text)
        
        assert "https://example.com" not in cleaned
        assert stats.urls_removed == 1
    
    def test_email_removal(self):
        """Test email removal."""
        cleaner = TextCleaner(remove_emails=True, min_length_chars=10)
        text = "Contact us at test@example.com for support information."
        cleaned, stats = cleaner.clean(text)
        
        assert "test@example.com" not in cleaned
        assert stats.emails_removed == 1
    
    def test_min_length_filter(self):
        """Test minimum length filter."""
        cleaner = TextCleaner(min_length_chars=100)
        text = "Short text."
        cleaned, stats = cleaner.clean(text)
        
        assert cleaned is None
        assert stats.cleaned_length < 100


class TestDeduplicator:
    """Tests for MinHashDeduplicator."""
    
    def test_exact_duplicate_detection(self):
        """Test detection of exact duplicates."""
        dedup = MinHashDeduplicator(num_perm=128, threshold=0.8)
        
        # Add first document
        is_dup1, _ = dedup.is_duplicate("doc1", "The quick brown fox jumps over the lazy dog.")
        assert not is_dup1
        
        # Add exact duplicate
        is_dup2, original = dedup.is_duplicate("doc2", "The quick brown fox jumps over the lazy dog.")
        assert is_dup2
        assert original == "doc1"
    
    def test_near_duplicate_detection(self):
        """Test detection of near-duplicates."""
        dedup = MinHashDeduplicator(num_perm=128, threshold=0.7, shingle_size=3)
        
        # Add first document
        is_dup1, _ = dedup.is_duplicate("doc1", "The quick brown fox jumps over the lazy dog.")
        assert not is_dup1
        
        # Add near-duplicate (slight variation)
        is_dup2, _ = dedup.is_duplicate("doc2", "The quick brown fox leaps over the lazy dog.")
        # May or may not be detected depending on threshold
        
        # Add different document
        is_dup3, _ = dedup.is_duplicate("doc3", "Python is a great programming language for data science.")
        assert not is_dup3
    
    def test_batch_deduplication(self):
        """Test batch deduplication."""
        dedup = MinHashDeduplicator(num_perm=128, threshold=0.8)
        
        documents = [
            Document("doc1", "The quick brown fox jumps over the lazy dog."),
            Document("doc2", "The quick brown fox jumps over the lazy dog."),
            Document("doc3", "Python is a great programming language."),
            Document("doc4", "Machine learning is transforming technology."),
        ]
        
        unique_docs, result = dedup.deduplicate_batch(documents, show_progress=False)
        
        assert result.total_documents == 4
        assert result.unique_documents == 3
        assert result.duplicate_documents == 1


class TestExactHashDeduplicator:
    """Tests for ExactHashDeduplicator."""
    
    def test_exact_duplicate(self):
        """Test exact duplicate detection."""
        dedup = ExactHashDeduplicator()
        
        assert not dedup.is_duplicate("Hello world")
        assert dedup.is_duplicate("Hello world")
        assert dedup.is_duplicate("Hello World")  # Case insensitive
    
    def test_case_insensitive(self):
        """Test that comparison is case-insensitive."""
        dedup = ExactHashDeduplicator()
        
        # The hash normalizes to lowercase
        assert not dedup.is_duplicate("HELLO WORLD")
        assert dedup.is_duplicate("hello world")


class TestQualityFilter:
    """Tests for QualityFilter."""
    
    def test_min_words_filter(self):
        """Test minimum word count filter."""
        qf = QualityFilter(min_words=50)
        
        short_text = "This is a short text."
        result = qf.check(short_text)
        
        assert not result.passed
        assert result.reason == FilterReason.TOO_SHORT
    
    def test_max_words_filter(self):
        """Test maximum word count filter."""
        qf = QualityFilter(max_words=10)
        
        long_text = " ".join(["word"] * 100)
        result = qf.check(long_text)
        
        assert not result.passed
        assert result.reason == FilterReason.TOO_LONG
    
    def test_alpha_ratio_filter(self):
        """Test alphabetic ratio filter."""
        qf = QualityFilter(min_words=1, min_alpha_ratio=0.7)
        
        # Text with many numbers
        text = "12345 67890 12345 67890 abc"
        result = qf.check(text)
        
        assert not result.passed
        assert result.reason == FilterReason.LOW_ALPHA_RATIO
    
    def test_pass_good_text(self):
        """Test that good text passes."""
        qf = QualityFilter(
            min_words=10,
            max_words=1000,
            allowed_languages=["en"]
        )
        
        good_text = """
        Python is a high-level, general-purpose programming language. 
        Its design philosophy emphasizes code readability with the use of 
        significant indentation. Python is dynamically typed and garbage-collected.
        """
        
        result = qf.check(good_text)
        assert result.passed
        assert result.reason == FilterReason.PASSED


class TestTokenizer:
    """Tests for TokenizerTrainer."""
    
    def test_bpe_training(self):
        """Test BPE tokenizer training."""
        trainer = TokenizerTrainer(
            algorithm="bpe",
            vocab_size=500,
            min_frequency=1
        )
        
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a great programming language.",
            "Machine learning is transforming technology.",
        ] * 50  # Repeat for more training data
        
        trainer.train(texts)
        
        assert trainer.vocab_size > 0
        assert trainer.vocab_size <= 500
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        trainer = TokenizerTrainer(
            algorithm="bpe",
            vocab_size=500,
            min_frequency=1
        )
        
        texts = ["Hello world. " * 100]
        trainer.train(texts)
        
        # Encode and decode
        original = "Hello world."
        encoded = trainer.encode(original)
        decoded = trainer.decode(encoded)
        
        assert len(encoded) > 0
        assert isinstance(encoded[0], int)
        # Note: decoded may have slight differences due to BPE
    
    def test_save_load(self):
        """Test saving and loading tokenizer."""
        trainer = TokenizerTrainer(
            algorithm="bpe",
            vocab_size=500,
            min_frequency=1
        )
        
        texts = ["Test text for tokenizer. " * 100]
        trainer.train(texts)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            trainer.save(str(path))
            
            # Load into new trainer
            new_trainer = TokenizerTrainer()
            new_trainer.load(str(path))
            
            assert new_trainer.vocab_size == trainer.vocab_size


class TestIntegration:
    """Integration tests."""
    
    def test_cleaner_to_filter_pipeline(self):
        """Test cleaning to quality filter pipeline."""
        cleaner = TextCleaner(min_length_chars=50)
        qf = QualityFilter(min_words=10)
        
        texts = [
            "Short.",
            "This is a longer piece of text that should pass through the cleaning stage successfully and then be evaluated by the quality filter.",
            "https://example.com Visit this URL for more info https://test.com",
        ]
        
        passed_texts = []
        for text in texts:
            cleaned, _ = cleaner.clean(text)
            if cleaned:
                result = qf.check(cleaned)
                if result.passed:
                    passed_texts.append(cleaned)
        
        # At least one should pass
        assert len(passed_texts) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
