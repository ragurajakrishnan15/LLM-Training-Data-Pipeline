# LLM Training Data Pipeline

A production-ready data pipeline for preparing text data for Large Language Model training. Built with Python, designed for scalability and reproducibility.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM Training Data Pipeline                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐│
│  │          │    │          │    │          │    │          │    │        ││
│  │ Ingest   │───▶│  Clean   │───▶│  Dedup   │───▶│ Quality  │───▶│Tokenize││
│  │          │    │          │    │          │    │  Filter  │    │        ││
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └────────┘│
│       │              │               │               │               │      │
│       ▼              ▼               ▼               ▼               ▼      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Metrics & Logging                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Pipeline Stages

| Stage | Description | Key Metrics |
|-------|-------------|-------------|
| **Ingestion** | Download Wikipedia dumps, parse XML | Documents ingested, bytes processed |
| **Cleaning** | Remove markup, normalize text | Characters removed, encoding fixes |
| **Deduplication** | MinHash LSH for near-duplicate detection | Duplicates found, unique documents |
| **Quality Filter** | Length, language, perplexity filtering | Documents filtered by reason |
| **Tokenization** | BPE tokenization, vocabulary stats | Token counts, vocab coverage |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-data-pipeline.git
cd llm-data-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Download sample data (Simple English Wikipedia)
python -m src.ingestion.download_wiki

# Run full pipeline
python -m src.main

# Or run individual stages
python -m src.processing.cleaner
python -m src.processing.deduplicator
python -m src.processing.quality_filter
python -m src.processing.tokenizer
```

### Docker

```bash
# Build and run
docker-compose up --build

# Run pipeline inside container
docker-compose exec pipeline python -m src.main
```

## 📁 Project Structure

```
llm-data-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── download_wiki.py      # Wikipedia dump downloader
│   │   └── wiki_parser.py        # XML parser for Wikipedia
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── cleaner.py            # Text cleaning & normalization
│   │   ├── deduplicator.py       # MinHash LSH deduplication
│   │   ├── quality_filter.py     # Quality filtering
│   │   └── tokenizer.py          # Tokenization & vocab
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management
│   │   ├── logger.py             # Logging utilities
│   │   └── metrics.py            # Pipeline metrics tracking
│   └── main.py                   # Main pipeline orchestrator
├── configs/
│   └── pipeline_config.yaml      # Pipeline configuration
├── data/
│   ├── raw/                      # Raw downloaded data
│   ├── processed/                # Intermediate processed data
│   └── output/                   # Final output files
├── tests/
│   └── test_pipeline.py          # Unit tests
├── notebooks/
│   └── data_exploration.ipynb    # Data analysis notebook
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

Edit `configs/pipeline_config.yaml`:

```yaml
ingestion:
  source: "simplewiki"
  max_articles: 10000

cleaning:
  remove_citations: true
  min_length: 100

deduplication:
  num_perm: 128
  threshold: 0.8

quality:
  min_words: 50
  max_words: 100000
  language: "en"

tokenization:
  vocab_size: 32000
  model_type: "bpe"
```

## 📈 Metrics Dashboard

The pipeline tracks metrics at each stage:

```
============================================================
                    PIPELINE METRICS REPORT
============================================================

INGESTION
  Documents ingested:     50,000
  Total bytes:           245.6 MB
  Time elapsed:          2m 34s

CLEANING
  Documents processed:    50,000
  Avg chars removed:      12.3%
  Encoding fixes:         234

DEDUPLICATION
  Input documents:        50,000
  Duplicates found:       3,456
  Output documents:       46,544
  Dedup rate:            6.9%

QUALITY FILTER
  Input documents:        46,544
  Filtered (too short):   1,234
  Filtered (wrong lang):  567
  Output documents:       44,743

TOKENIZATION
  Documents tokenized:    44,743
  Total tokens:          89.4M
  Vocab size:            32,000
  Avg tokens/doc:        1,998

============================================================
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Technologies Used

- **Python 3.9+** - Core language
- **mwparserfromhell** - Wikipedia markup parsing
- **datasketch** - MinHash LSH deduplication
- **langdetect** - Language detection
- **tokenizers** - HuggingFace tokenizers
- **pandas** - Data manipulation
- **tqdm** - Progress bars
- **PyYAML** - Configuration
- **Docker** - Containerization

## 📚 References

- [The Pile: An 800GB Dataset of Diverse Text](https://arxiv.org/abs/2101.00027)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://arxiv.org/abs/2103.12028)

## 📄 License

MIT License - see LICENSE file for details.

## 👤 Author

**Raguraja Krishnan Natarajan Mangaleshwaran**
- MS Information Systems (Applied Data Science) - SUNY Binghamton
- [LinkedIn](https://linkedin.com/in/yourprofile)
- [GitHub](https://github.com/yourprofile)
