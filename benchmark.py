#!/usr/bin/env python3
"""
Benchmarking script for SlopBERT models across different variants and checkpoints.
Downloads latest headlines and evaluates them with all available model variants.
"""

import os
import sys
import json
import csv
import time
import datetime
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import pandas as pd
import feedparser
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to basic print
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    
console = Console()

# Configuration
BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
FULL_RESULTS_DIR = "./results"
LORA_RESULTS_DIR = "./results_lora"
RESULTS_QUANTIZED_INT8_DIR = "./results_quantized_int8"
RESULTS_QUANTIZED_FLOAT8_DIR = "./results_quantized_float8"
FEEDS_FILE = "my_feeds.json"
BENCHMARK_CACHE_FILE = "benchmark_headlines_cache.json"
BENCHMARK_RESULTS_DIR = "./benchmark_results"

NUM_LABELS = 6
MAX_LENGTH = 128
LABEL_MAP = {
    0: "slop",
    1: "meh", 
    2: "ok",
    3: "not bad",
    4: "good stuff",
    5: "banger",
}

@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results"""
    headline: str
    model_type: str
    model_path: str
    checkpoint: Optional[str]
    predicted_label: str
    predicted_id: int
    confidence: float
    inference_time_ms: float
    source: str
    category: str
    timestamp: str

@dataclass
class ModelVariant:
    """Data class for model variant information"""
    name: str
    type: str  # 'full', 'lora', 'quantized'
    path: str
    checkpoint: Optional[str] = None
    available: bool = True

def get_available_models() -> List[ModelVariant]:
    """Discover all available model variants and checkpoints"""
    models = []
    
    # Full models
    if os.path.exists(FULL_RESULTS_DIR):
        checkpoints = get_checkpoints(FULL_RESULTS_DIR)
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            models.append(ModelVariant(
                name=f"Full Model (Latest: {latest})",
                type="full",
                path=os.path.join(FULL_RESULTS_DIR, latest),
                checkpoint=latest
            ))
            
            # Add a few other significant checkpoints for comparison
            if len(checkpoints) > 1:
                checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
                # Add earliest and middle checkpoint if available
                if len(checkpoints_sorted) >= 3:
                    earliest = checkpoints_sorted[0]
                    middle = checkpoints_sorted[len(checkpoints_sorted)//2]
                    models.extend([
                        ModelVariant(
                            name=f"Full Model (Early: {earliest})",
                            type="full",
                            path=os.path.join(FULL_RESULTS_DIR, earliest),
                            checkpoint=earliest
                        ),
                        ModelVariant(
                            name=f"Full Model (Mid: {middle})",
                            type="full", 
                            path=os.path.join(FULL_RESULTS_DIR, middle),
                            checkpoint=middle
                        )
                    ])
    
    # LoRA models
    if PEFT_AVAILABLE and os.path.exists(LORA_RESULTS_DIR):
        final_adapter = os.path.join(LORA_RESULTS_DIR, "final_adapter")
        if os.path.exists(final_adapter):
            models.append(ModelVariant(
                name="LoRA Model (Final)",
                type="lora",
                path=final_adapter
            ))
        
        # Add some checkpoint variants
        checkpoints = get_checkpoints(LORA_RESULTS_DIR)
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            models.append(ModelVariant(
                name=f"LoRA Model (Latest: {latest})",
                type="lora",
                path=os.path.join(LORA_RESULTS_DIR, latest),
                checkpoint=latest
            ))
    
    # Quantized models
    for quant_dir, quant_type in [
        (RESULTS_QUANTIZED_INT8_DIR, "int8"),
        (RESULTS_QUANTIZED_FLOAT8_DIR, "float8")
    ]:
        if os.path.exists(quant_dir) and os.path.exists(os.path.join(quant_dir, "config.json")):
            models.append(ModelVariant(
                name=f"Quantized Model ({quant_type.upper()})",
                type=f"quantized_{quant_type}",
                path=quant_dir
            ))
    
    return models

def get_checkpoints(results_dir: str) -> List[str]:
    """Get all checkpoint directories in a results directory"""
    if not os.path.exists(results_dir):
        return []
    
    return [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("checkpoint-")
    ]

def load_feeds_from_json(file_path: str = FEEDS_FILE) -> List[Dict]:
    """Load RSS feeds from JSON file"""
    if not os.path.isfile(file_path):
        console.print(f"[yellow]Feeds file '{file_path}' not found.[/yellow]")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            feeds = data.get("feeds", [])
            return [f for f in feeds if f.get("enabled", True) and f.get("url")]
    except (json.JSONDecodeError, IOError) as e:
        console.print(f"[red]Error loading feeds: {e}[/red]")
        return []

def fetch_recent_headlines(max_headlines: int = 50) -> List[Dict]:
    """Fetch recent headlines from RSS feeds"""
    feeds = load_feeds_from_json()
    if not feeds:
        console.print("[red]No feeds available for headline fetching[/red]")
        return []
    
    console.print(f"[cyan]Fetching headlines from {len(feeds)} feeds...[/cyan]")
    
    headlines = []
    today = datetime.datetime.now().date()
    
    def fetch_feed(feed):
        try:
            feed_data = feedparser.parse(feed["url"])
            feed_headlines = []
            
            for entry in feed_data.entries[:10]:  # Limit per feed
                title = entry.get("title", "").strip()
                if not title:
                    continue
                
                # Try to get publication date
                published_dt = None
                for key in ["published_parsed", "updated_parsed"]:
                    if hasattr(entry, key) and getattr(entry, key):
                        try:
                            parsed_time = getattr(entry, key)
                            if len(parsed_time) >= 6:
                                dt = datetime.datetime(*parsed_time[:6])
                                published_dt = dt.date()
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Only include recent headlines (today or yesterday)
                if published_dt and (today - published_dt).days <= 1:
                    feed_headlines.append({
                        "title": title,
                        "source": feed.get("name", "Unknown"),
                        "category": feed.get("category", "General"),
                        "url": entry.get("link", ""),
                        "published": published_dt.isoformat() if published_dt else None
                    })
            
            return feed_headlines
        except Exception as e:
            console.print(f"[red]Error fetching {feed.get('name', 'Unknown')}: {e}[/red]")
            return []
    
    # Fetch in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_feed = {executor.submit(fetch_feed, feed): feed for feed in feeds}
        
        for future in as_completed(future_to_feed):
            try:
                feed_headlines = future.result(timeout=30)
                headlines.extend(feed_headlines)
            except Exception as e:
                console.print(f"[red]Feed fetch failed: {e}[/red]")
    
    # Remove duplicates and limit
    seen_titles = set()
    unique_headlines = []
    for headline in headlines:
        if headline["title"] not in seen_titles:
            seen_titles.add(headline["title"])
            unique_headlines.append(headline)
            if len(unique_headlines) >= max_headlines:
                break
    
    console.print(f"[green]Fetched {len(unique_headlines)} unique recent headlines[/green]")
    return unique_headlines

def load_model_and_tokenizer(model_variant: ModelVariant) -> Tuple[Any, Any, torch.device]:
    """Load a model variant and its tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    try:
        if model_variant.type == "full":
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_variant.path
            ).to(device).eval()
            
        elif model_variant.type == "lora":
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT not available for LoRA models")
            
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_NAME, num_labels=NUM_LABELS
            )
            model = PeftModel.from_pretrained(base_model, model_variant.path).to(device).eval()
            
        elif model_variant.type.startswith("quantized"):
            tokenizer = AutoTokenizer.from_pretrained(model_variant.path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_variant.path, trust_remote_code=True
            ).to(device).eval()
            
        else:
            raise ValueError(f"Unknown model type: {model_variant.type}")
        
        return model, tokenizer, device
        
    except Exception as e:
        console.print(f"[red]Error loading {model_variant.name}: {e}[/red]")
        return None, None, None

def evaluate_headline(headline: str, model: Any, tokenizer: Any, device: torch.device) -> Tuple[int, float, float]:
    """Evaluate a single headline and return prediction, confidence, and inference time"""
    start_time = time.perf_counter()
    
    try:
        inputs = tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class_id].item()
        
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        return predicted_class_id, confidence, inference_time
        
    except Exception as e:
        console.print(f"[red]Error evaluating headline: {e}[/red]")
        inference_time = (time.perf_counter() - start_time) * 1000
        return 0, 0.0, inference_time

def run_benchmark(headlines: List[Dict], models: List[ModelVariant]) -> List[BenchmarkResult]:
    """Run benchmark across all headlines and models"""
    results = []
    total_evaluations = len(headlines) * len(models)
    
    console.print(f"[cyan]Starting benchmark: {len(headlines)} headlines × {len(models)} models = {total_evaluations} evaluations[/cyan]")
    
    if RICH_AVAILABLE:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
    
    for model_variant in models:
        console.print(f"\n[yellow]Loading {model_variant.name}...[/yellow]")
        
        model, tokenizer, device = load_model_and_tokenizer(model_variant)
        if model is None:
            console.print(f"[red]Skipping {model_variant.name} due to loading error[/red]")
            continue
        
        if RICH_AVAILABLE:
            with progress:
                task = progress.add_task(f"Evaluating with {model_variant.name}", total=len(headlines))
                
                for headline_data in headlines:
                    headline = headline_data["title"]
                    predicted_id, confidence, inference_time = evaluate_headline(
                        headline, model, tokenizer, device
                    )
                    
                    result = BenchmarkResult(
                        headline=headline,
                        model_type=model_variant.name,
                        model_path=model_variant.path,
                        checkpoint=model_variant.checkpoint,
                        predicted_label=LABEL_MAP.get(predicted_id, f"Unknown_{predicted_id}"),
                        predicted_id=predicted_id,
                        confidence=confidence,
                        inference_time_ms=inference_time,
                        source=headline_data.get("source", "Unknown"),
                        category=headline_data.get("category", "General"),
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    results.append(result)
                    progress.advance(task)
        else:
            # Fallback without rich progress
            for i, headline_data in enumerate(headlines):
                headline = headline_data["title"]
                predicted_id, confidence, inference_time = evaluate_headline(
                    headline, model, tokenizer, device
                )
                
                result = BenchmarkResult(
                    headline=headline,
                    model_type=model_variant.name,
                    model_path=model_variant.path,
                    checkpoint=model_variant.checkpoint,
                    predicted_label=LABEL_MAP.get(predicted_id, f"Unknown_{predicted_id}"),
                    predicted_id=predicted_id,
                    confidence=confidence,
                    inference_time_ms=inference_time,
                    source=headline_data.get("source", "Unknown"),
                    category=headline_data.get("category", "General"),
                    timestamp=datetime.datetime.now().isoformat()
                )
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    console.print(f"  Progress: {i + 1}/{len(headlines)} headlines")
        
        # Clean up model to free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze benchmark results and generate statistics"""
    if not results:
        return {}
    
    analysis = {
        "summary": {
            "total_evaluations": len(results),
            "unique_headlines": len(set(r.headline for r in results)),
            "models_tested": len(set(r.model_type for r in results)),
            "timestamp": datetime.datetime.now().isoformat()
        },
        "performance": {},
        "predictions": {},
        "agreement": {}
    }
    
    # Performance analysis
    by_model = {}
    for result in results:
        if result.model_type not in by_model:
            by_model[result.model_type] = []
        by_model[result.model_type].append(result)
    
    for model_type, model_results in by_model.items():
        inference_times = [r.inference_time_ms for r in model_results]
        confidences = [r.confidence for r in model_results]
        
        analysis["performance"][model_type] = {
            "avg_inference_time_ms": statistics.mean(inference_times),
            "median_inference_time_ms": statistics.median(inference_times),
            "avg_confidence": statistics.mean(confidences),
            "median_confidence": statistics.median(confidences),
            "evaluations": len(model_results)
        }
    
    # Prediction distribution analysis
    for model_type, model_results in by_model.items():
        label_counts = {}
        for result in model_results:
            label = result.predicted_label
            label_counts[label] = label_counts.get(label, 0) + 1
        
        analysis["predictions"][model_type] = label_counts
    
    # Model agreement analysis
    headlines = list(set(r.headline for r in results))
    agreements = []
    
    for headline in headlines:
        headline_results = [r for r in results if r.headline == headline]
        if len(headline_results) > 1:
            predictions = [r.predicted_id for r in headline_results]
            # Calculate agreement as percentage of models that agree with mode
            mode_prediction = max(set(predictions), key=predictions.count)
            agreement = predictions.count(mode_prediction) / len(predictions)
            agreements.append(agreement)
    
    if agreements:
        analysis["agreement"]["average"] = statistics.mean(agreements)
        analysis["agreement"]["median"] = statistics.median(agreements)
        analysis["agreement"]["min"] = min(agreements)
        analysis["agreement"]["max"] = max(agreements)
    
    return analysis

def save_results(results: List[BenchmarkResult], analysis: Dict[str, Any]):
    """Save benchmark results and analysis to files"""
    os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as CSV
    results_file = os.path.join(BENCHMARK_RESULTS_DIR, f"benchmark_results_{timestamp}.csv")
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        if results:
            fieldnames = list(asdict(results[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
    
    # Save analysis as JSON
    analysis_file = os.path.join(BENCHMARK_RESULTS_DIR, f"benchmark_analysis_{timestamp}.json")
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    
    console.print(f"[green]Results saved to:[/green]")
    console.print(f"  - {results_file}")
    console.print(f"  - {analysis_file}")

def display_summary(analysis: Dict[str, Any]):
    """Display a summary of benchmark results"""
    if not analysis:
        console.print("[red]No analysis data to display[/red]")
        return
    
    if RICH_AVAILABLE:
        # Performance summary table
        perf_table = Table(title="Model Performance Summary")
        perf_table.add_column("Model", style="cyan")
        perf_table.add_column("Avg Inference (ms)", justify="right")
        perf_table.add_column("Avg Confidence", justify="right")
        perf_table.add_column("Evaluations", justify="right")
        
        for model, stats in analysis.get("performance", {}).items():
            perf_table.add_row(
                model,
                f"{stats['avg_inference_time_ms']:.2f}",
                f"{stats['avg_confidence']:.3f}",
                str(stats['evaluations'])
            )
        
        console.print(perf_table)
        
        # Agreement summary
        if "agreement" in analysis:
            agreement_panel = Panel(
                f"Average Model Agreement: {analysis['agreement'].get('average', 0):.1%}\n"
                f"Median Agreement: {analysis['agreement'].get('median', 0):.1%}",
                title="Model Agreement Analysis"
            )
            console.print(agreement_panel)
    else:
        # Fallback display
        console.print("\n=== BENCHMARK SUMMARY ===")
        console.print(f"Total evaluations: {analysis['summary']['total_evaluations']}")
        console.print(f"Unique headlines: {analysis['summary']['unique_headlines']}")
        console.print(f"Models tested: {analysis['summary']['models_tested']}")
        
        console.print("\n=== PERFORMANCE ===")
        for model, stats in analysis.get("performance", {}).items():
            console.print(f"{model}:")
            console.print(f"  Avg inference: {stats['avg_inference_time_ms']:.2f}ms")
            console.print(f"  Avg confidence: {stats['avg_confidence']:.3f}")

def main():
    """Main benchmark execution"""
    console.print("[bold blue]SlopBERT Model Benchmarking Tool[/bold blue]")
    console.print("=" * 50)
    
    # Discover available models
    console.print("[cyan]Discovering available models...[/cyan]")
    models = get_available_models()
    
    if not models:
        console.print("[red]No trained models found! Please train some models first.[/red]")
        return
    
    console.print(f"[green]Found {len(models)} model variants:[/green]")
    for model in models:
        console.print(f"  - {model.name}")
    
    # Fetch recent headlines
    console.print("\n[cyan]Fetching recent headlines...[/cyan]")
    headlines = fetch_recent_headlines(max_headlines=50)
    
    if not headlines:
        console.print("[red]No recent headlines found! Check your RSS feeds.[/red]")
        return
    
    # Run benchmark
    console.print(f"\n[cyan]Running benchmark on {len(headlines)} headlines...[/cyan]")
    results = run_benchmark(headlines, models)
    
    if not results:
        console.print("[red]Benchmark failed to produce results[/red]")
        return
    
    # Analyze results
    console.print("\n[cyan]Analyzing results...[/cyan]")
    analysis = analyze_results(results)
    
    # Save and display results
    save_results(results, analysis)
    console.print("\n")
    display_summary(analysis)
    
    console.print(f"\n[green]Benchmark completed! {len(results)} evaluations performed.[/green]")

if __name__ == "__main__":
    main()