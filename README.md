# SlopBERT

SlopBERT is an interactive terminal-based application for training a modernbert model (or similar) for rating RSS feed headlines. It provides an intuitive interface for dataset creation, model training, and headline classification tasks.

## Key Features

- **Interactive Terminal UI**: Navigate effortlessly using arrow keys (`j`, `k`, or arrows).
- **Dataset Management**: Fetch and label RSS headlines from customizable sources.
- **Automated Labeling**: Easily label headlines as 'quality' or 'slop' for classification.
- **Model Training & Testing**: Seamlessly train and evaluate headline classification models.
- **Feed Management**: Add, remove, enable, or disable RSS feed sources directly from the interface.

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Install dependencies and run the application

We use `uv` take care of package management (if you haven't switched to uv yet - Now is the right time to do it):

```bash
uv run app.py
```

## Project Structure

```
.
├── app.py                 # Main interactive application
├── train.py               # Script to train the ML model
├── pyproject.toml         # Project dependencies managed by uv
├── uv.lock                # Locked dependencies for reproducibility
├── dataset.csv            # Labeled dataset
├── headlines_cache.json   # Cached headlines from RSS feeds
├── my_feeds.json          # User-defined RSS feeds
└── results/               # Trained model checkpoints

```

### Main Menu Options

- **Create or Update Dataset**: Fetch headlines from RSS sources and label them.
- **Train Model**: Initiate model training using labeled data.
- **Add or Remove Sources**: Manage your RSS feed sources.
- **Test Model**: Classify headlines using the latest trained model.

### RSS Feed Management

Feeds are managed via `my_feeds.json`. A default curated set is available, or you can manually add your own:

```json
{
  "feeds": [
    {
      "name": "TechCrunch",
      "url": "https://techcrunch.com/rss",
      "category": "Technology",
      "enabled": true
    }
  ]
}
```

## Training & Checkpoints

Training generates checkpoints stored in the `results/` directory. The latest checkpoint is automatically selected during model evaluation.

## Dependencies (Managed by uv)

- `rich` for interactive CLI
- `feedparser` for RSS parsing
- `tldextract` for domain extraction
- `pytz` for timezone management
- `transformers` & `torch` for ML model handling

## Acknowlegements

- [ModernBERT](https://github.com/AnswerDotAI/ModernBERT)

## Contributions

Contributions are welcome! Feel free to submit pull requests or report issues.

## License

MIT License
