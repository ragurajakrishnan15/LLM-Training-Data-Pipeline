"""
Main pipeline orchestrator for the LLM Training Data Pipeline.

Coordinates all pipeline stages from ingestion to tokenization.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion import download_wikipedia, WikipediaParser, WikiArticle
from src.processing import (
    TextCleaner,
    MinHashDeduplicator,
    Document,
    QualityFilter,
    TokenizerTrainer,
    tokenize_documents,
)
from src.utils import get_config, get_logger, get_metrics, reset_metrics

logger = get_logger(__name__)


class LLMDataPipeline:
    """
    End-to-end LLM training data pipeline.
    
    Pipeline stages:
    1. Ingestion: Download and parse Wikipedia dumps
    2. Cleaning: Text normalization and cleanup
    3. Deduplication: Remove near-duplicate documents
    4. Quality filtering: Filter low-quality documents
    5. Tokenization: Train tokenizer and tokenize documents
    6. Output: Save processed data in training-ready format
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config()
        if config_path:
            self.config.load_config(config_path)
        
        self.metrics = get_metrics()
        
        # Initialize components
        self.cleaner = TextCleaner(
            **{k: v for k, v in self.config.cleaning.items() 
               if k in ['remove_urls', 'remove_emails', 'remove_citations',
                       'normalize_unicode', 'normalize_whitespace', 'min_length_chars']}
        )
        
        self.deduplicator = MinHashDeduplicator(
            num_perm=self.config.deduplication.get('num_permutations', 128),
            threshold=self.config.deduplication.get('threshold', 0.8),
            shingle_size=self.config.deduplication.get('shingle_size', 5)
        )
        
        self.quality_filter = QualityFilter(
            min_words=self.config.quality.get('min_words', 50),
            max_words=self.config.quality.get('max_words', 100000),
            allowed_languages=self.config.quality.get('language_filter', {}).get('allowed_languages'),
        )
        
        self.tokenizer: Optional[TokenizerTrainer] = None
        
        logger.info("Pipeline initialized")
    
    def run(
        self,
        source: str = "simplewiki",
        max_articles: Optional[int] = None,
        output_dir: Optional[str] = None,
        skip_download: bool = False,
        wiki_dump_path: Optional[str] = None
    ) -> dict:
        """
        Run the complete pipeline.
        
        Args:
            source: Wikipedia source (simplewiki, enwiki)
            max_articles: Maximum articles to process
            output_dir: Output directory for results
            skip_download: Skip download if file exists
            wiki_dump_path: Path to existing Wikipedia dump
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        reset_metrics()
        self.metrics = get_metrics()
        self.metrics.start_pipeline()
        
        # Setup paths
        if output_dir is None:
            output_dir = self.config.paths.get('output_data', 'data/output')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("Starting LLM Training Data Pipeline")
        logger.info("="*60)
        
        # Stage 1: Ingestion
        articles = self._run_ingestion(
            source=source,
            max_articles=max_articles,
            skip_download=skip_download,
            wiki_dump_path=wiki_dump_path
        )
        
        # Stage 2: Cleaning
        cleaned_articles = self._run_cleaning(articles)
        
        # Stage 3: Deduplication
        unique_articles = self._run_deduplication(cleaned_articles)
        
        # Stage 4: Quality filtering
        quality_articles = self._run_quality_filter(unique_articles)
        
        # Stage 5: Tokenization
        tokenized_data = self._run_tokenization(quality_articles)
        
        # Stage 6: Output
        self._save_output(quality_articles, tokenized_data, output_dir)
        
        self.metrics.end_pipeline()
        
        # Print final report
        logger.info("="*60)
        logger.info("Pipeline Complete!")
        logger.info("="*60)
        self.metrics.print_report()
        
        # Save metrics
        metrics_path = output_dir / "pipeline_metrics.json"
        self.metrics.save(str(metrics_path))
        
        return self.metrics.to_dict()
    
    def _run_ingestion(
        self,
        source: str,
        max_articles: Optional[int],
        skip_download: bool,
        wiki_dump_path: Optional[str]
    ) -> List[WikiArticle]:
        """Run ingestion stage."""
        logger.info("\n[Stage 1/5] INGESTION")
        logger.info("-"*40)
        
        self.metrics.start_stage("ingestion")
        
        # Get or download Wikipedia dump
        if wiki_dump_path:
            dump_path = wiki_dump_path
        else:
            raw_dir = self.config.paths.get('raw_data', 'data/raw')
            dump_path = download_wikipedia(source=source, output_dir=raw_dir)
        
        # Parse Wikipedia
        parser = WikipediaParser(remove_markup=True)
        articles = list(parser.parse_file(dump_path, max_articles=max_articles))
        
        total_bytes = sum(len(a.text.encode('utf-8')) for a in articles)
        
        self.metrics.end_stage(
            "ingestion",
            input_count=0,
            output_count=len(articles),
            bytes_processed=total_bytes
        )
        
        logger.info(f"Ingested {len(articles):,} articles ({total_bytes/1024/1024:.1f} MB)")
        
        return articles
    
    def _run_cleaning(self, articles: List[WikiArticle]) -> List[dict]:
        """Run cleaning stage."""
        logger.info("\n[Stage 2/5] CLEANING")
        logger.info("-"*40)
        
        self.metrics.start_stage("cleaning")
        
        cleaned_articles = []
        filtered_count = 0
        total_chars_removed = 0
        
        for article in articles:
            cleaned_text, stats = self.cleaner.clean(article.text)
            total_chars_removed += stats.chars_removed
            
            if cleaned_text:
                cleaned_articles.append({
                    'id': str(article.page_id),
                    'title': article.title,
                    'text': cleaned_text,
                    'original_length': stats.original_length,
                    'cleaned_length': stats.cleaned_length
                })
            else:
                filtered_count += 1
        
        self.metrics.end_stage(
            "cleaning",
            input_count=len(articles),
            output_count=len(cleaned_articles),
            filtered_count=filtered_count,
            chars_removed=total_chars_removed
        )
        
        logger.info(
            f"Cleaned {len(cleaned_articles):,} articles, "
            f"filtered {filtered_count:,} (too short)"
        )
        
        return cleaned_articles
    
    def _run_deduplication(self, articles: List[dict]) -> List[dict]:
        """Run deduplication stage."""
        logger.info("\n[Stage 3/5] DEDUPLICATION")
        logger.info("-"*40)
        
        self.metrics.start_stage("deduplication")
        
        # Convert to Document objects
        documents = [
            Document(
                doc_id=a['id'],
                text=a['text'],
                metadata={'title': a['title']}
            )
            for a in articles
        ]
        
        # Deduplicate
        unique_docs, result = self.deduplicator.deduplicate_batch(documents)
        
        # Convert back to dict format
        unique_articles = []
        unique_ids = {d.doc_id for d in unique_docs}
        
        for article in articles:
            if article['id'] in unique_ids:
                unique_articles.append(article)
        
        self.metrics.end_stage(
            "deduplication",
            input_count=result.total_documents,
            output_count=result.unique_documents,
            filtered_count=result.duplicate_documents,
            duplicate_clusters=result.duplicate_clusters
        )
        
        logger.info(
            f"Found {result.duplicate_documents:,} duplicates "
            f"({result.dedup_rate:.1f}%)"
        )
        
        return unique_articles
    
    def _run_quality_filter(self, articles: List[dict]) -> List[dict]:
        """Run quality filtering stage."""
        logger.info("\n[Stage 4/5] QUALITY FILTERING")
        logger.info("-"*40)
        
        self.metrics.start_stage("quality")
        
        quality_articles = []
        filter_reasons = {}
        
        for article in articles:
            result = self.quality_filter.check(article['text'])
            
            reason_name = result.reason.value
            filter_reasons[reason_name] = filter_reasons.get(reason_name, 0) + 1
            
            if result.passed:
                quality_articles.append(article)
            else:
                self.metrics.add_filter_reason(reason_name)
        
        filtered_count = len(articles) - len(quality_articles)
        
        self.metrics.end_stage(
            "quality",
            input_count=len(articles),
            output_count=len(quality_articles),
            filtered_count=filtered_count
        )
        
        logger.info(f"Passed {len(quality_articles):,} articles")
        for reason, count in filter_reasons.items():
            if reason != "passed":
                logger.info(f"  - {reason}: {count:,}")
        
        return quality_articles
    
    def _run_tokenization(self, articles: List[dict]) -> dict:
        """Run tokenization stage."""
        logger.info("\n[Stage 5/5] TOKENIZATION")
        logger.info("-"*40)
        
        self.metrics.start_stage("tokenization")
        
        texts = [a['text'] for a in articles]
        
        # Train tokenizer
        tok_config = self.config.tokenization
        self.tokenizer = TokenizerTrainer(
            algorithm=tok_config.get('algorithm', 'bpe'),
            vocab_size=tok_config.get('vocab_size', 32000),
            min_frequency=tok_config.get('min_frequency', 2)
        )
        
        logger.info("Training tokenizer...")
        self.tokenizer.train(texts)
        
        # Tokenize all documents
        logger.info("Tokenizing documents...")
        token_ids, stats = tokenize_documents(texts, self.tokenizer)
        
        self.metrics.end_stage(
            "tokenization",
            input_count=len(articles),
            output_count=len(token_ids),
            total_tokens=stats.total_tokens,
            vocab_size=stats.vocab_size,
            avg_tokens_per_doc=stats.avg_tokens_per_doc,
            compression_ratio=stats.compression_ratio
        )
        
        logger.info(f"Total tokens: {stats.total_tokens:,}")
        logger.info(f"Vocab size: {stats.vocab_size:,}")
        logger.info(f"Avg tokens/doc: {stats.avg_tokens_per_doc:.1f}")
        
        return {
            'token_ids': token_ids,
            'stats': stats
        }
    
    def _save_output(
        self,
        articles: List[dict],
        tokenized_data: dict,
        output_dir: Path
    ) -> None:
        """Save pipeline output."""
        logger.info("\n[OUTPUT] Saving results...")
        logger.info("-"*40)
        
        # Save tokenizer
        tokenizer_path = output_dir / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))
        logger.info(f"Saved tokenizer: {tokenizer_path}")
        
        # Save processed data as Parquet
        output_format = self.config.get('output.format', 'parquet')
        
        if output_format == 'parquet':
            # Create DataFrame
            df = pd.DataFrame([
                {
                    'id': a['id'],
                    'title': a['title'],
                    'text': a['text'],
                    'token_count': len(tokenized_data['token_ids'][i])
                }
                for i, a in enumerate(articles)
            ])
            
            parquet_path = output_dir / "processed_data.parquet"
            df.to_parquet(parquet_path, compression='snappy')
            logger.info(f"Saved data: {parquet_path}")
        
        elif output_format == 'jsonl':
            jsonl_path = output_dir / "processed_data.jsonl"
            with open(jsonl_path, 'w') as f:
                for i, article in enumerate(articles):
                    record = {
                        'id': article['id'],
                        'title': article['title'],
                        'text': article['text'],
                        'tokens': tokenized_data['token_ids'][i]
                    }
                    f.write(json.dumps(record) + '\n')
            logger.info(f"Saved data: {jsonl_path}")
        
        # Save tokenized data separately (for training)
        tokens_path = output_dir / "tokenized_data.jsonl"
        with open(tokens_path, 'w') as f:
            for tokens in tokenized_data['token_ids']:
                f.write(json.dumps({'tokens': tokens}) + '\n')
        logger.info(f"Saved tokenized data: {tokens_path}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(articles),
            'total_tokens': tokenized_data['stats'].total_tokens,
            'vocab_size': tokenized_data['stats'].vocab_size,
            'avg_tokens_per_doc': tokenized_data['stats'].avg_tokens_per_doc,
            'compression_ratio': tokenized_data['stats'].compression_ratio
        }
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Training Data Pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default="simplewiki",
        help="Wikipedia source (simplewiki, enwiki)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum articles to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--wiki-dump",
        type=str,
        default=None,
        help="Path to existing Wikipedia dump"
    )
    
    args = parser.parse_args()
    
    pipeline = LLMDataPipeline(config_path=args.config)
    
    results = pipeline.run(
        source=args.source,
        max_articles=args.max_articles,
        output_dir=args.output_dir,
        wiki_dump_path=args.wiki_dump
    )
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
