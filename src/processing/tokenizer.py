"""
Tokenization module for the LLM Training Data Pipeline.

Handles BPE tokenization and vocabulary building.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

from src.utils import get_config, get_logger, get_metrics

logger = get_logger(__name__)


@dataclass
class TokenizationStats:
    """Statistics from tokenization."""
    
    total_documents: int
    total_tokens: int
    total_chars: int
    vocab_size: int
    
    @property
    def avg_tokens_per_doc(self) -> float:
        """Average tokens per document."""
        if self.total_documents == 0:
            return 0.0
        return self.total_tokens / self.total_documents
    
    @property
    def compression_ratio(self) -> float:
        """Characters per token (compression ratio)."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_chars / self.total_tokens
    
    def __str__(self) -> str:
        return (
            f"Documents: {self.total_documents:,}\n"
            f"Total tokens: {self.total_tokens:,}\n"
            f"Total chars: {self.total_chars:,}\n"
            f"Vocab size: {self.vocab_size:,}\n"
            f"Avg tokens/doc: {self.avg_tokens_per_doc:.1f}\n"
            f"Compression ratio: {self.compression_ratio:.2f} chars/token"
        )


class TokenizerTrainer:
    """
    Train tokenizers for LLM training data.
    
    Supports:
    - BPE (Byte-Pair Encoding) - Used by GPT models
    - WordPiece - Used by BERT models
    - Unigram - Used by T5, ALBERT
    """
    
    DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
    
    def __init__(
        self,
        algorithm: str = "bpe",
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize the tokenizer trainer.
        
        Args:
            algorithm: Tokenization algorithm (bpe, wordpiece, unigram)
            vocab_size: Target vocabulary size
            min_frequency: Minimum token frequency
            special_tokens: Special tokens to include
        """
        self.algorithm = algorithm.lower()
        self._vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or self.DEFAULT_SPECIAL_TOKENS
        
        self.tokenizer: Optional[Tokenizer] = None
        self.trainer = None
        
        self._setup_tokenizer()
        
        logger.info(
            f"Initialized tokenizer trainer: algorithm={algorithm}, "
            f"vocab_size={vocab_size}"
        )
    
    def _setup_tokenizer(self) -> None:
        """Setup tokenizer and trainer based on algorithm."""
        if self.algorithm == "bpe":
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.trainer = BpeTrainer(
                vocab_size=self._vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=self.special_tokens,
                show_progress=True
            )
        
        elif self.algorithm == "wordpiece":
            self.tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
            self.tokenizer.pre_tokenizer = Whitespace()
            self.trainer = WordPieceTrainer(
                vocab_size=self._vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=self.special_tokens,
                show_progress=True
            )
        
        elif self.algorithm == "unigram":
            self.tokenizer = Tokenizer(Unigram())
            self.tokenizer.pre_tokenizer = Whitespace()
            self.trainer = UnigramTrainer(
                vocab_size=self._vocab_size,
                special_tokens=self.special_tokens,
                show_progress=True
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, texts: List[str]) -> None:
        """
        Train tokenizer on texts.
        
        Args:
            texts: List of training texts
        """
        logger.info(f"Training {self.algorithm} tokenizer on {len(texts):,} documents...")
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)
        logger.info(f"Tokenizer trained. Vocab size: {self.vocab_size}")
    
    def train_from_files(self, filepaths: List[str]) -> None:
        """
        Train tokenizer from text files.
        
        Args:
            filepaths: List of file paths
        """
        logger.info(f"Training {self.algorithm} tokenizer from {len(filepaths)} files...")
        self.tokenizer.train(filepaths, trainer=self.trainer)
        logger.info(f"Tokenizer trained. Vocab size: {self.vocab_size}")
    
    def save(self, filepath: str) -> None:
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save tokenizer
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
        logger.info(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to tokenizer file
        """
        self.tokenizer = Tokenizer.from_file(filepath)
        logger.info(f"Tokenizer loaded from {filepath}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text).ids
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts to token IDs.
        
        Args:
            texts: List of texts
            
        Returns:
            List of token ID lists
        """
        encodings = self.tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: Token IDs
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(ids)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary as dict."""
        return self.tokenizer.get_vocab()
    
    @property
    def vocab_size(self) -> int:
        """Get actual vocabulary size."""
        return self.tokenizer.get_vocab_size()


def tokenize_documents(
    texts: List[str],
    tokenizer: TokenizerTrainer,
    show_progress: bool = True
) -> Tuple[List[List[int]], TokenizationStats]:
    """
    Tokenize a batch of documents and compute statistics.
    
    Args:
        texts: List of texts to tokenize
        tokenizer: Trained tokenizer
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (tokenized documents, statistics)
    """
    total_tokens = 0
    total_chars = 0
    all_token_ids = []
    
    # Process in batches for efficiency
    batch_size = 1000
    
    if show_progress:
        pbar = tqdm(total=len(texts), desc="Tokenizing")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Encode batch
        batch_encodings = tokenizer.encode_batch(batch)
        all_token_ids.extend(batch_encodings)
        
        # Update stats
        for text, tokens in zip(batch, batch_encodings):
            total_tokens += len(tokens)
            total_chars += len(text)
        
        if show_progress:
            pbar.update(len(batch))
    
    if show_progress:
        pbar.close()
    
    stats = TokenizationStats(
        total_documents=len(texts),
        total_tokens=total_tokens,
        total_chars=total_chars,
        vocab_size=tokenizer.vocab_size
    )
    
    logger.info(f"Tokenization complete:\n{stats}")
    
    return all_token_ids, stats


def save_tokenized_data(
    token_ids: List[List[int]],
    output_path: str,
    format: str = "jsonl"
) -> None:
    """
    Save tokenized data to file.
    
    Args:
        token_ids: List of token ID lists
        output_path: Output file path
        format: Output format (jsonl, json)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(path, 'w') as f:
            for ids in token_ids:
                f.write(json.dumps({"tokens": ids}) + '\n')
    
    elif format == "json":
        with open(path, 'w') as f:
            json.dump({"documents": [{"tokens": ids} for ids in token_ids]}, f)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Saved tokenized data to {output_path}")


def main():
    """Main entry point for testing tokenization."""
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language.",
        "Machine learning models require large amounts of training data.",
        "Natural language processing enables computers to understand text.",
        "Transformers have revolutionized the field of NLP.",
    ]
    
    print("Testing Tokenizer")
    print("="*60)
    
    # Train tokenizer
    trainer = TokenizerTrainer(
        algorithm="bpe",
        vocab_size=1000,  # Small for testing
        min_frequency=1
    )
    
    print("\nTraining tokenizer...")
    trainer.train(test_texts * 100)  # Repeat for more training data
    
    print(f"\nVocab size: {trainer.vocab_size}")
    
    # Tokenize documents
    print("\nTokenizing documents...")
    token_ids, stats = tokenize_documents(test_texts, trainer, show_progress=False)
    
    print(f"\nTokenization Stats:")
    print(stats)
    
    # Show examples
    print("\nTokenization Examples:")
    print("-"*60)
    for text, ids in zip(test_texts[:3], token_ids[:3]):
        decoded = trainer.decode(ids)
        print(f"Original: {text}")
        print(f"Token IDs: {ids[:20]}..." if len(ids) > 20 else f"Token IDs: {ids}")
        print(f"Decoded: {decoded}")
        print("-"*60)


if __name__ == "__main__":
    main()
