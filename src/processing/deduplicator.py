"""
Deduplication module for the LLM Training Data Pipeline.

Uses MinHash LSH for efficient near-duplicate detection.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Tuple

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from src.utils import get_config, get_logger, get_metrics

logger = get_logger(__name__)


@dataclass
class Document:
    """Document with ID and text for deduplication."""
    
    doc_id: str
    text: str
    metadata: Optional[Dict] = None


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    
    total_documents: int
    unique_documents: int
    duplicate_documents: int
    duplicate_clusters: int
    
    @property
    def dedup_rate(self) -> float:
        """Get percentage of documents that were duplicates."""
        if self.total_documents == 0:
            return 0.0
        return (self.duplicate_documents / self.total_documents) * 100


class MinHashDeduplicator:
    """
    Near-duplicate detection using MinHash LSH.
    
    MinHash LSH (Locality-Sensitive Hashing) is an efficient algorithm
    for finding similar documents in large corpora. It's used by major
    LLM training pipelines like The Pile and RedPajama.
    
    How it works:
    1. Convert each document into a set of shingles (n-grams)
    2. Create a MinHash signature for each document
    3. Use LSH to efficiently find candidate pairs
    4. Documents with Jaccard similarity above threshold are duplicates
    """
    
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        shingle_size: int = 5,
        num_bands: Optional[int] = None
    ):
        """
        Initialize the deduplicator.
        
        Args:
            num_perm: Number of permutations for MinHash (more = more accurate)
            threshold: Jaccard similarity threshold for duplicates
            shingle_size: Size of character n-grams (shingles)
            num_bands: Number of LSH bands (auto-calculated if None)
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        
        # Calculate optimal number of bands for the threshold
        if num_bands is None:
            num_bands = self._calculate_optimal_bands(num_perm, threshold)
        
        self.num_bands = num_bands
        
        # Initialize LSH index
        self.lsh = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm
        )
        
        # Track documents
        self.documents: Dict[str, MinHash] = {}
        self.duplicate_of: Dict[str, str] = {}  # Maps duplicate -> original
        
        logger.info(
            f"Initialized MinHash deduplicator: "
            f"num_perm={num_perm}, threshold={threshold}, shingle_size={shingle_size}"
        )
    
    def _calculate_optimal_bands(self, num_perm: int, threshold: float) -> int:
        """Calculate optimal number of bands for target threshold."""
        # b * r = num_perm, where b = bands, r = rows per band
        # Probability of being candidate: 1 - (1 - s^r)^b ≈ threshold at s = threshold
        # Approximate: b ≈ num_perm / -log(threshold) 
        import math
        return max(1, int(num_perm / (-math.log(threshold) * 2)))
    
    def _get_shingles(self, text: str) -> Set[str]:
        """
        Convert text to set of character n-grams (shingles).
        
        Args:
            text: Input text
            
        Returns:
            Set of shingles
        """
        # Normalize text
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        # Generate shingles
        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            shingles.add(shingle)
        
        return shingles
    
    def _create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature for text.
        
        Args:
            text: Input text
            
        Returns:
            MinHash object
        """
        minhash = MinHash(num_perm=self.num_perm)
        
        shingles = self._get_shingles(text)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        
        return minhash
    
    def is_duplicate(self, doc_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a document is a duplicate of any seen document.
        
        Args:
            doc_id: Unique document identifier
            text: Document text
            
        Returns:
            Tuple of (is_duplicate, original_doc_id if duplicate else None)
        """
        minhash = self._create_minhash(text)
        
        # Query LSH for candidates
        candidates = self.lsh.query(minhash)
        
        if candidates:
            # Return first candidate as the "original"
            original_id = candidates[0]
            self.duplicate_of[doc_id] = original_id
            return True, original_id
        
        # Not a duplicate, add to index
        try:
            self.lsh.insert(doc_id, minhash)
            self.documents[doc_id] = minhash
        except ValueError:
            # Document ID already exists
            pass
        
        return False, None
    
    def deduplicate_batch(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> Tuple[List[Document], DeduplicationResult]:
        """
        Deduplicate a batch of documents.
        
        Args:
            documents: List of documents to deduplicate
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (unique documents, deduplication result)
        """
        unique_docs = []
        duplicate_count = 0
        
        iterator = tqdm(documents, desc="Deduplicating") if show_progress else documents
        
        for doc in iterator:
            is_dup, original_id = self.is_duplicate(doc.doc_id, doc.text)
            
            if not is_dup:
                unique_docs.append(doc)
            else:
                duplicate_count += 1
        
        result = DeduplicationResult(
            total_documents=len(documents),
            unique_documents=len(unique_docs),
            duplicate_documents=duplicate_count,
            duplicate_clusters=len(set(self.duplicate_of.values()))
        )
        
        logger.info(
            f"Deduplication complete: {result.unique_documents:,}/{result.total_documents:,} "
            f"unique ({result.dedup_rate:.1f}% duplicates)"
        )
        
        return unique_docs, result
    
    def deduplicate_stream(
        self,
        documents: Generator[Document, None, None],
        show_progress: bool = True
    ) -> Generator[Document, None, None]:
        """
        Deduplicate a stream of documents.
        
        Args:
            documents: Generator of documents
            show_progress: Whether to show progress
            
        Yields:
            Unique documents
        """
        total = 0
        duplicates = 0
        
        for doc in documents:
            total += 1
            is_dup, _ = self.is_duplicate(doc.doc_id, doc.text)
            
            if not is_dup:
                yield doc
            else:
                duplicates += 1
            
            if show_progress and total % 1000 == 0:
                logger.info(
                    f"Processed {total:,} documents, "
                    f"{duplicates:,} duplicates ({(duplicates/total)*100:.1f}%)"
                )
        
        logger.info(
            f"Stream deduplication complete: "
            f"{total - duplicates:,}/{total:,} unique"
        )
    
    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        return {
            "total_indexed": len(self.documents),
            "total_duplicates": len(self.duplicate_of),
            "unique_documents": len(self.documents),
            "duplicate_clusters": len(set(self.duplicate_of.values()))
        }
    
    def clear(self) -> None:
        """Clear the deduplication index."""
        self.lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm
        )
        self.documents.clear()
        self.duplicate_of.clear()


class ExactHashDeduplicator:
    """
    Simple exact-match deduplication using content hashing.
    
    Faster but only catches exact duplicates, not near-duplicates.
    """
    
    def __init__(self, hash_algorithm: str = "md5"):
        """
        Initialize the deduplicator.
        
        Args:
            hash_algorithm: Hash algorithm to use (md5, sha256, etc.)
        """
        self.hash_algorithm = hash_algorithm
        self.seen_hashes: Set[str] = set()
        self.duplicate_count = 0
    
    def _hash_text(self, text: str) -> str:
        """Hash text content."""
        text = text.strip().lower()
        hasher = hashlib.new(self.hash_algorithm)
        hasher.update(text.encode('utf-8'))
        return hasher.hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a duplicate.
        
        Args:
            text: Document text
            
        Returns:
            True if duplicate, False otherwise
        """
        content_hash = self._hash_text(text)
        
        if content_hash in self.seen_hashes:
            self.duplicate_count += 1
            return True
        
        self.seen_hashes.add(content_hash)
        return False
    
    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        return {
            "unique_documents": len(self.seen_hashes),
            "duplicate_count": self.duplicate_count
        }


def main():
    """Main entry point for testing deduplication."""
    # Test documents
    documents = [
        Document("doc1", "The quick brown fox jumps over the lazy dog."),
        Document("doc2", "The quick brown fox jumps over the lazy dog."),  # Exact dup
        Document("doc3", "The quick brown fox leaps over the lazy dog."),  # Near dup
        Document("doc4", "Python is a great programming language."),
        Document("doc5", "Python is an excellent programming language."),  # Near dup
        Document("doc6", "Machine learning is transforming technology."),
    ]
    
    print("Testing MinHash Deduplicator")
    print("="*60)
    
    deduplicator = MinHashDeduplicator(
        num_perm=128,
        threshold=0.7,  # Lower threshold to catch "leaps" vs "jumps"
        shingle_size=3
    )
    
    unique_docs, result = deduplicator.deduplicate_batch(documents, show_progress=False)
    
    print(f"\nInput documents: {result.total_documents}")
    print(f"Unique documents: {result.unique_documents}")
    print(f"Duplicates found: {result.duplicate_documents}")
    print(f"Dedup rate: {result.dedup_rate:.1f}%")
    
    print("\nUnique documents:")
    for doc in unique_docs:
        print(f"  - {doc.doc_id}: {doc.text[:50]}...")


if __name__ == "__main__":
    main()
