# SlopBERT Benchmarking System

This document describes the comprehensive benchmarking system for SlopBERT models that evaluates different model variants against recent headlines.

## Overview

The benchmarking system downloads recent headlines from configured RSS feeds and evaluates them across all available model variants, including:

- **Full fine-tuned models** (multiple checkpoints)
- **LoRA models** (adapters and checkpoints) 
- **Quantized models** (INT8 and FLOAT8 variants)

## Components

### 1. benchmark.py
Main benchmarking script that:
- Discovers all available trained models
- Fetches recent headlines from RSS feeds
- Evaluates headlines across all model variants
- Measures inference time and confidence scores
- Saves detailed results and analysis

**Usage:**
```bash
python benchmark.py
```

### 2. benchmark_visualizer.py
Visualization and reporting tool that generates:
- Performance comparison charts
- Prediction analysis plots
- Checkpoint evolution tracking
- Comprehensive HTML reports

**Usage:**
```bash
python benchmark_visualizer.py [--output-dir ./benchmark_reports]
```

### 3. Integration with app.py
The benchmarking system is integrated into the main SlopBERT application as menu option "8. Run Model Benchmark".

## Features

### Model Discovery
Automatically discovers and tests:
- Latest full model checkpoints
- Early and mid-training checkpoints for comparison
- LoRA final adapters and checkpoint variants
- All available quantized models

### Performance Metrics
- **Inference Time**: Measures speed in milliseconds
- **Prediction Confidence**: Model certainty scores
- **Model Agreement**: How often models agree on predictions
- **Prediction Distribution**: Label frequency analysis

### Visualization Reports
- **Performance Comparison**: Speed vs accuracy trade-offs
- **Checkpoint Evolution**: Training progress analysis
- **Model Agreement Heatmaps**: Consensus visualization
- **Source Diversity**: Headline source breakdown

## Output Files

### Results Directory: `./benchmark_results/`
- `benchmark_results_YYYYMMDD_HHMMSS.csv` - Detailed evaluation results
- `benchmark_analysis_YYYYMMDD_HHMMSS.json` - Statistical analysis

### Reports Directory: `./benchmark_reports/`
- `benchmark_report.html` - Comprehensive HTML report
- `performance_comparison.png` - Model performance charts
- `prediction_analysis.png` - Prediction distribution plots
- `checkpoint_comparison.png` - Training evolution charts
- `benchmark_summary.png` - High-level summary visualization

## Data Schema

### BenchmarkResult
Each evaluation produces a result with:
```python
@dataclass
class BenchmarkResult:
    headline: str                # Evaluated headline text
    model_type: str             # Model variant name
    model_path: str             # Path to model files
    checkpoint: Optional[str]    # Checkpoint identifier
    predicted_label: str        # Human-readable prediction
    predicted_id: int           # Numeric prediction (0-5)
    confidence: float           # Model confidence score
    inference_time_ms: float    # Processing time
    source: str                 # News source
    category: str               # Content category
    timestamp: str              # Evaluation timestamp
```

## Requirements

### Dependencies
- `torch` - PyTorch for model inference
- `transformers` - Hugging Face models
- `peft` - LoRA model support (optional)
- `feedparser` - RSS feed processing
- `pandas` - Data manipulation
- `matplotlib` - Chart generation
- `seaborn` - Statistical visualization
- `rich` - Terminal UI (optional)

### Model Requirements
At least one trained model variant must be available:
- Full model in `./results/checkpoint-*/`
- LoRA adapter in `./results_lora/final_adapter/` or checkpoint
- Quantized model in `./results_quantized_*/`

### Feed Configuration
RSS feeds must be configured in `my_feeds.json` for headline fetching.

## Usage Examples

### Basic Benchmark
```bash
# Run from SlopBERT main menu
python app.py
# Select: "8. Run Model Benchmark"
```

### Standalone Benchmark
```bash
# Direct execution
python benchmark.py

# Generate visualization after benchmark
python benchmark_visualizer.py
```

### Custom Output Directory
```bash
python benchmark_visualizer.py --output-dir ./custom_reports/
```

## Analysis Insights

The benchmarking system provides insights into:

1. **Model Evolution**: How performance changes across training checkpoints
2. **Speed vs Accuracy**: Trade-offs between different model variants
3. **Quantization Impact**: Performance effects of model compression
4. **Prediction Consistency**: Agreement levels between model variants
5. **Content Bias**: How models perform on different news categories

## Troubleshooting

### Common Issues

**No models found:**
- Ensure at least one model has been trained
- Check that model directories exist and contain proper files

**No headlines fetched:**
- Verify RSS feeds are configured in `my_feeds.json`
- Check network connectivity
- Ensure feeds contain recent content

**Visualization errors:**
- Install missing dependencies: `matplotlib`, `seaborn`
- Check that benchmark results exist in `./benchmark_results/`
- Verify Python environment has required packages

**Memory issues:**
- Reduce number of headlines in `fetch_recent_headlines()`
- Test one model variant at a time
- Use quantized models for lower memory usage

### Performance Tips

- **Parallel Processing**: Benchmarking uses ThreadPoolExecutor for RSS fetching
- **Memory Management**: Models are unloaded between variants to free GPU memory
- **Batch Processing**: Headlines are processed individually for detailed timing
- **Caching**: Results are cached to avoid re-downloading headlines

## Future Enhancements

Potential improvements to the benchmarking system:

1. **Cross-validation**: Multiple evaluation runs for statistical significance
2. **A/B Testing**: Direct model comparison on identical headline sets
3. **Temporal Analysis**: Performance tracking over time
4. **Custom Metrics**: User-defined evaluation criteria
5. **Export Options**: Results export to various formats (Excel, PDF)
6. **API Integration**: REST API for programmatic benchmarking

## Contributing

When adding new features to the benchmarking system:

1. Follow the existing data schema for consistency
2. Add appropriate error handling and logging
3. Update visualization components for new metrics
4. Test with various model configurations
5. Document new parameters and outputs