#!/usr/bin/env python3
"""Quick pipeline demo with sample data."""

import json
from pathlib import Path
from datetime import datetime

from src.processing.cleaner import TextCleaner
from src.processing.deduplicator import MinHashDeduplicator
from src.processing.quality_filter import QualityFilter
from src.processing.tokenizer import TokenizerTrainer
from src.utils import get_logger

logger = get_logger(__name__)

# Sample articles
SAMPLE_ARTICLES = [
    {
        "id": "1",
        "title": "Python Programming",
        "text": "Python is a high-level programming language known for its simplicity and readability. "
                "It supports multiple programming paradigms including procedural, object-oriented, and functional programming. "
                "Python is widely used in web development, data science, artificial intelligence, and many other fields. "
                "The language has a large standard library and a vibrant ecosystem of third-party packages."
    },
    {
        "id": "2",
        "title": "Machine Learning Basics",
        "text": "Machine learning is a subset of artificial intelligence that focuses on enabling computers to learn from data. "
                "There are three main types: supervised learning, unsupervised learning, and reinforcement learning. "
                "Supervised learning involves training on labeled data, while unsupervised learning works with unlabeled data. "
                "Common algorithms include decision trees, neural networks, and support vector machines."
    },
    {
        "id": "3",
        "title": "Data Science",
        "text": "Data science combines statistics, programming, and domain expertise to extract insights from data. "
                "The typical data science workflow includes data collection, cleaning, exploration, modeling, and visualization. "
                "Popular tools include Python, R, SQL, and various visualization libraries like Matplotlib and Tableau. "
                "Data scientists work across industries to solve real-world problems using data-driven approaches."
    },
    {
        "id": "4",
        "title": "Web Development",
        "text": "Web development involves creating and maintaining websites and web applications. "
                "It includes frontend development for user interfaces and backend development for server logic. "
                "Popular frontend frameworks include React, Vue, and Angular for building dynamic user experiences. "
                "Backend technologies range from traditional databases to modern cloud platforms and microservices."
    },
    {
        "id": "5",
        "title": "Cloud Computing",
        "text": "Cloud computing provides on-demand access to computing resources over the internet. "
                "Major cloud providers include Amazon Web Services, Google Cloud Platform, and Microsoft Azure. "
                "Cloud services are categorized as IaaS, PaaS, and SaaS based on the level of abstraction. "
                "Organizations benefit from scalability, flexibility, and cost-effectiveness of cloud infrastructure."
    },
]

def run_demo():
    """Run a quick demo of the pipeline stages."""
    
    print("\n" + "="*70)
    print("LLM DATA PIPELINE - QUICK DEMO")
    print("="*70 + "\n")
    
    # Stage 1: Cleaning
    print("[Stage 1/4] TEXT CLEANING")
    print("-" * 70)
    cleaner = TextCleaner(
        remove_urls=True,
        remove_emails=True,
        normalize_unicode=True,
        min_length_chars=100
    )
    
    cleaned_articles = []
    for article in SAMPLE_ARTICLES:
        text, stats = cleaner.clean(article['text'])
        if text:
            cleaned_articles.append({
                'id': article['id'],
                'title': article['title'],
                'text': text,
                'original_chars': stats.original_length,
                'cleaned_chars': stats.cleaned_length
            })
            print(f"  ✓ {article['title']}: {stats.original_length} → {stats.cleaned_length} chars")
    
    print(f"\n  Summary: {len(cleaned_articles)}/{len(SAMPLE_ARTICLES)} articles cleaned\n")
    
    # Stage 2: Deduplication
    print("[Stage 2/4] DEDUPLICATION")
    print("-" * 70)
    dedup = MinHashDeduplicator(num_perm=128, threshold=0.8)
    
    unique_articles = []
    for article in cleaned_articles:
        is_dup, original = dedup.is_duplicate(article['id'], article['text'])
        if not is_dup:
            unique_articles.append(article)
            print(f"  ✓ {article['title']}: UNIQUE")
        else:
            print(f"  ✗ {article['title']}: DUPLICATE (of {original})")
    
    print(f"\n  Summary: {len(unique_articles)}/{len(cleaned_articles)} unique articles\n")
    
    # Stage 3: Quality Filtering
    print("[Stage 3/4] QUALITY FILTERING")
    print("-" * 70)
    qf = QualityFilter(min_words=50, max_words=100000, allowed_languages=['en'])
    
    quality_articles = []
    for article in unique_articles:
        result = qf.check(article['text'])
        if result.passed:
            quality_articles.append(article)
            print(f"  ✓ {article['title']}: PASSED")
        else:
            print(f"  ✗ {article['title']}: FILTERED ({result.reason.value})")
    
    print(f"\n  Summary: {len(quality_articles)}/{len(unique_articles)} articles passed quality check\n")
    
    # Stage 4: Tokenization
    print("[Stage 4/4] TOKENIZATION")
    print("-" * 70)
    trainer = TokenizerTrainer(algorithm='bpe', vocab_size=1000)
    
    texts = [a['text'] for a in quality_articles]
    trainer.train(texts)
    print(f"  ✓ BPE Tokenizer trained with vocabulary size: 1000")
    
    # Tokenize all articles
    total_tokens = 0
    for article in quality_articles:
        tokens = trainer.encode(article['text'])
        total_tokens += len(tokens)
        print(f"  ✓ {article['title']}: {len(tokens)} tokens")
    
    avg_tokens = total_tokens / len(quality_articles) if quality_articles else 0
    print(f"\n  Summary: {total_tokens} total tokens, {avg_tokens:.1f} avg tokens/article\n")
    
    # Final Summary
    print("="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Input articles:           {len(SAMPLE_ARTICLES)}")
    print(f"After cleaning:           {len(cleaned_articles)}")
    print(f"After deduplication:      {len(unique_articles)}")
    print(f"After quality filtering:  {len(quality_articles)}")
    print(f"Total tokens generated:   {total_tokens}")
    print("="*70 + "\n")
    
    # Save demo results
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    demo_results = {
        "timestamp": datetime.now().isoformat(),
        "stage": "demo",
        "input_articles": len(SAMPLE_ARTICLES),
        "cleaned_articles": len(cleaned_articles),
        "unique_articles": len(unique_articles),
        "quality_articles": len(quality_articles),
        "total_tokens": total_tokens,
        "avg_tokens_per_article": avg_tokens,
        "articles": quality_articles
    }
    
    with open(output_dir / "demo_results.json", 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print(f"✓ Results saved to: data/output/demo_results.json\n")
    print("✓ Pipeline demo completed successfully!\n")

if __name__ == "__main__":
    run_demo()
