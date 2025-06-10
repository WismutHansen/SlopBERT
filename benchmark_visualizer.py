#!/usr/bin/env python3
"""
Benchmark results visualization and reporting tool.
Generates comprehensive charts and reports from benchmark data.
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

BENCHMARK_RESULTS_DIR = "./benchmark_results"
REPORTS_DIR = "./benchmark_reports"

def load_latest_results() -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the most recent benchmark results and analysis"""
    if not os.path.exists(BENCHMARK_RESULTS_DIR):
        raise FileNotFoundError(f"Results directory {BENCHMARK_RESULTS_DIR} not found")
    
    # Find latest results files
    result_files = [f for f in os.listdir(BENCHMARK_RESULTS_DIR) if f.startswith("benchmark_results_") and f.endswith(".csv")]
    analysis_files = [f for f in os.listdir(BENCHMARK_RESULTS_DIR) if f.startswith("benchmark_analysis_") and f.endswith(".json")]
    
    if not result_files or not analysis_files:
        raise FileNotFoundError("No benchmark results found")
    
    # Get latest files
    latest_results = max(result_files)
    latest_analysis = max(analysis_files)
    
    # Load data
    results_df = pd.read_csv(os.path.join(BENCHMARK_RESULTS_DIR, latest_results))
    
    with open(os.path.join(BENCHMARK_RESULTS_DIR, latest_analysis), 'r') as f:
        analysis = json.load(f)
    
    print(f"Loaded results: {latest_results}")
    print(f"Loaded analysis: {latest_analysis}")
    
    return results_df, analysis

def create_performance_charts(df: pd.DataFrame, output_dir: str):
    """Create performance comparison charts"""
    
    # 1. Inference Time Comparison
    plt.figure(figsize=(12, 6))
    
    # Box plot of inference times by model
    plt.subplot(1, 2, 1)
    model_order = df.groupby('model_type')['inference_time_ms'].median().sort_values().index
    sns.boxplot(data=df, y='model_type', x='inference_time_ms', order=model_order)
    plt.title('Inference Time Distribution by Model')
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Model Type')
    
    # 2. Confidence Distribution
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y='model_type', x='confidence', order=model_order)
    plt.title('Prediction Confidence Distribution by Model')
    plt.xlabel('Confidence Score')
    plt.ylabel('Model Type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_analysis(df: pd.DataFrame, output_dir: str):
    """Create prediction distribution and agreement analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction distribution by model
    ax1 = axes[0, 0]
    prediction_counts = df.groupby(['model_type', 'predicted_label']).size().unstack(fill_value=0)
    prediction_counts.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title('Prediction Distribution by Model')
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Number of Predictions')
    ax1.legend(title='Predicted Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Model agreement heatmap
    ax2 = axes[0, 1]
    # Create agreement matrix
    headlines = df['headline'].unique()
    models = df['model_type'].unique()
    
    agreement_data = []
    for headline in headlines[:20]:  # Limit to first 20 for readability
        headline_preds = df[df['headline'] == headline]
        if len(headline_preds) > 1:
            pred_dict = dict(zip(headline_preds['model_type'], headline_preds['predicted_id']))
            agreement_data.append(pred_dict)
    
    if agreement_data:
        agreement_df = pd.DataFrame(agreement_data).fillna(-1)  # -1 for missing predictions
        sns.heatmap(agreement_df.T, annot=True, fmt='.0f', cmap='viridis', ax=ax2)
        ax2.set_title('Model Predictions by Headline (Sample)')
        ax2.set_xlabel('Headlines (Sample)')
        ax2.set_ylabel('Model Type')
    
    # 3. Confidence vs Prediction scatter
    ax3 = axes[1, 0]
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model]
        ax3.scatter(model_data['predicted_id'], model_data['confidence'], 
                   alpha=0.6, label=model, s=20)
    ax3.set_title('Confidence vs Predicted Class')
    ax3.set_xlabel('Predicted Class ID')
    ax3.set_ylabel('Confidence Score')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Source category breakdown
    ax4 = axes[1, 1]
    category_counts = df['category'].value_counts()
    ax4.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    ax4.set_title('Headlines by Source Category')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_checkpoint_comparison(df: pd.DataFrame, output_dir: str):
    """Create checkpoint evolution analysis for models with multiple checkpoints"""
    
    # Filter for models with checkpoint information
    checkpoint_data = df[df['checkpoint'].notna()]
    
    if checkpoint_data.empty:
        print("No checkpoint data found for comparison")
        return
    
    # Extract checkpoint numbers for sorting
    checkpoint_data = checkpoint_data.copy()
    checkpoint_data['checkpoint_num'] = checkpoint_data['checkpoint'].str.split('-').str[1].astype(int)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Performance evolution by checkpoint
    ax1 = axes[0]
    base_model_types = checkpoint_data['model_type'].str.split(' (', regex=False).str[0].unique()
    for model_type in base_model_types:
        model_checkpoints = checkpoint_data[checkpoint_data['model_type'].str.split(' (', regex=False).str[0] == model_type]
        if len(model_checkpoints) > 1:
            checkpoint_perf = model_checkpoints.groupby('checkpoint_num').agg({
                'inference_time_ms': 'mean',
                'confidence': 'mean'
            }).reset_index()
            
            ax1.plot(checkpoint_perf['checkpoint_num'], checkpoint_perf['inference_time_ms'], 
                    marker='o', label=f'{model_type} - Inference Time')
    
    ax1.set_title('Inference Time Evolution by Checkpoint')
    ax1.set_xlabel('Checkpoint Number')
    ax1.set_ylabel('Average Inference Time (ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence evolution
    ax2 = axes[1]
    for model_type in base_model_types:
        model_checkpoints = checkpoint_data[checkpoint_data['model_type'].str.split(' (', regex=False).str[0] == model_type]
        if len(model_checkpoints) > 1:
            checkpoint_perf = model_checkpoints.groupby('checkpoint_num').agg({
                'confidence': 'mean'
            }).reset_index()
            
            ax2.plot(checkpoint_perf['checkpoint_num'], checkpoint_perf['confidence'], 
                    marker='s', label=f'{model_type} - Confidence')
    
    ax2.set_title('Confidence Evolution by Checkpoint')
    ax2.set_xlabel('Checkpoint Number')
    ax2.set_ylabel('Average Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Prediction stability across checkpoints
    ax3 = axes[2]
    # Calculate prediction variance for each headline across checkpoints
    prediction_stability = []
    for headline in checkpoint_data['headline'].unique():
        headline_data = checkpoint_data[checkpoint_data['headline'] == headline]
        if len(headline_data) > 2:
            pred_variance = headline_data['predicted_id'].var()
            prediction_stability.append(pred_variance)
    
    if prediction_stability:
        ax3.hist(prediction_stability, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_title('Prediction Stability Across Checkpoints')
        ax3.set_xlabel('Prediction Variance')
        ax3.set_ylabel('Number of Headlines')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'checkpoint_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_report(df: pd.DataFrame, analysis: Dict[str, Any], output_dir: str):
    """Create a detailed HTML report"""
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SlopBERT Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 8px; }}
            .section {{ margin: 30px 0; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .chart-container {{ text-align: center; margin: 20px 0; }}
            .chart-container img {{ max-width: 100%; height: auto; }}
            .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>SlopBERT Model Benchmark Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Evaluations:</strong> {analysis.get('summary', {}).get('total_evaluations', 'N/A')}</p>
            <p><strong>Unique Headlines:</strong> {analysis.get('summary', {}).get('unique_headlines', 'N/A')}</p>
            <p><strong>Models Tested:</strong> {analysis.get('summary', {}).get('models_tested', 'N/A')}</p>
        </div>
        
        <div class="section">
            <h2>Performance Summary</h2>
            <table class="metrics-table">
                <tr>
                    <th>Model</th>
                    <th>Avg Inference Time (ms)</th>
                    <th>Median Inference Time (ms)</th>
                    <th>Avg Confidence</th>
                    <th>Median Confidence</th>
                    <th>Evaluations</th>
                </tr>
    """
    
    # Add performance metrics table
    for model, stats in analysis.get('performance', {}).items():
        report_html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{stats.get('avg_inference_time_ms', 0):.2f}</td>
                    <td>{stats.get('median_inference_time_ms', 0):.2f}</td>
                    <td>{stats.get('avg_confidence', 0):.3f}</td>
                    <td>{stats.get('median_confidence', 0):.3f}</td>
                    <td>{stats.get('evaluations', 0)}</td>
                </tr>
        """
    
    report_html += """
            </table>
        </div>
    """
    
    # Add model agreement section
    if 'agreement' in analysis:
        agreement = analysis['agreement']
        report_html += f"""
        <div class="section">
            <h2>Model Agreement Analysis</h2>
            <div class="summary-box">
                <p><strong>Average Agreement:</strong> {agreement.get('average', 0):.1%}</p>
                <p><strong>Median Agreement:</strong> {agreement.get('median', 0):.1%}</p>
                <p><strong>Range:</strong> {agreement.get('min', 0):.1%} - {agreement.get('max', 0):.1%}</p>
            </div>
        </div>
        """
    
    # Add charts
    report_html += """
        <div class="section">
            <h2>Visualizations</h2>
            
            <h3>Performance Comparison</h3>
            <div class="chart-container">
                <img src="performance_comparison.png" alt="Performance Comparison">
            </div>
            
            <h3>Prediction Analysis</h3>
            <div class="chart-container">
                <img src="prediction_analysis.png" alt="Prediction Analysis">
            </div>
            
            <h3>Checkpoint Comparison</h3>
            <div class="chart-container">
                <img src="checkpoint_comparison.png" alt="Checkpoint Comparison">
            </div>
        </div>
    """
    
    # Add prediction distribution section
    report_html += """
        <div class="section">
            <h2>Prediction Distribution</h2>
            <table class="metrics-table">
                <tr>
                    <th>Model</th>
                    <th>Slop</th>
                    <th>Meh</th>
                    <th>OK</th>
                    <th>Not Bad</th>
                    <th>Good Stuff</th>
                    <th>Banger</th>
                </tr>
    """
    
    # Add prediction counts
    for model, predictions in analysis.get('predictions', {}).items():
        report_html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{predictions.get('slop', 0)}</td>
                    <td>{predictions.get('meh', 0)}</td>
                    <td>{predictions.get('ok', 0)}</td>
                    <td>{predictions.get('not bad', 0)}</td>
                    <td>{predictions.get('good stuff', 0)}</td>
                    <td>{predictions.get('banger', 0)}</td>
                </tr>
        """
    
    report_html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Sample Headlines and Predictions</h2>
            <table class="metrics-table">
                <tr>
                    <th>Headline</th>
                    <th>Source</th>
                    <th>Category</th>
                    <th>Model Predictions</th>
                </tr>
    """
    
    # Add sample headlines
    sample_headlines = df['headline'].unique()[:10]
    for headline in sample_headlines:
        headline_data = df[df['headline'] == headline]
        predictions = ', '.join([f"{row['model_type'].split('(')[0].strip()}: {row['predicted_label']}" 
                                for _, row in headline_data.iterrows()])
        
        source = headline_data.iloc[0]['source']
        category = headline_data.iloc[0]['category']
        
        report_html += f"""
                <tr>
                    <td>{headline[:100]}{'...' if len(headline) > 100 else ''}</td>
                    <td>{source}</td>
                    <td>{category}</td>
                    <td>{predictions}</td>
                </tr>
        """
    
    report_html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Methodology</h2>
            <p>This benchmark evaluates multiple SlopBERT model variants on recent headlines fetched from RSS feeds. 
               Each model processes the same set of headlines and generates predictions with confidence scores.</p>
            <ul>
                <li><strong>Models tested:</strong> Full fine-tuned models, LoRA adapters, and quantized variants</li>
                <li><strong>Headlines:</strong> Recent articles from configured RSS feeds</li>
                <li><strong>Metrics:</strong> Inference time, prediction confidence, model agreement</li>
                <li><strong>Labels:</strong> 6-point scale from "slop" (0) to "banger" (5)</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save report
    with open(os.path.join(output_dir, 'benchmark_report.html'), 'w', encoding='utf-8') as f:
        f.write(report_html)

def generate_comparative_summary(df: pd.DataFrame, output_dir: str):
    """Generate a quick comparative summary chart"""
    
    # Create a summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SlopBERT Model Benchmark Summary', fontsize=16, fontweight='bold')
    
    # 1. Speed vs Accuracy scatter
    ax1 = axes[0, 0]
    model_stats = df.groupby('model_type').agg({
        'inference_time_ms': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    scatter = ax1.scatter(model_stats['inference_time_ms'], model_stats['confidence'], 
                         s=100, alpha=0.7, c=range(len(model_stats)), cmap='viridis')
    
    # Add labels for each point
    for idx, row in model_stats.iterrows():
        ax1.annotate(row['model_type'].split('(')[0].strip(), 
                    (row['inference_time_ms'], row['confidence']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Average Inference Time (ms)')
    ax1.set_ylabel('Average Confidence')
    ax1.set_title('Speed vs Confidence Trade-off')
    ax1.grid(True, alpha=0.3)
    
    # 2. Model type distribution
    ax2 = axes[0, 1]
    model_types = df['model_type'].str.split('(', regex=False).str[0].str.strip().value_counts()
    ax2.pie(model_types.values, labels=model_types.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Model Variants Tested')
    
    # 3. Prediction distribution comparison
    ax3 = axes[1, 0]
    prediction_summary = df.groupby(['predicted_label']).size().sort_index()
    bars = ax3.bar(prediction_summary.index, prediction_summary.values, color='skyblue', alpha=0.7)
    ax3.set_title('Overall Prediction Distribution')
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # 4. Source diversity
    ax4 = axes[1, 1]
    source_counts = df['source'].value_counts().head(10)
    ax4.barh(range(len(source_counts)), source_counts.values, color='lightcoral', alpha=0.7)
    ax4.set_yticks(range(len(source_counts)))
    ax4.set_yticklabels(source_counts.index)
    ax4.set_title('Top News Sources')
    ax4.set_xlabel('Number of Headlines')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description='Generate benchmark visualization reports')
    parser.add_argument('--output-dir', default=REPORTS_DIR, 
                       help='Output directory for reports and charts')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        print("Loading benchmark results...")
        df, analysis = load_latest_results()
        
        print(f"Loaded {len(df)} evaluation results")
        print(f"Models: {df['model_type'].nunique()}")
        print(f"Headlines: {df['headline'].nunique()}")
        
        # Generate visualizations
        print("Creating performance charts...")
        create_performance_charts(df, args.output_dir)
        
        print("Creating prediction analysis...")
        create_prediction_analysis(df, args.output_dir)
        
        print("Creating checkpoint comparison...")
        create_checkpoint_comparison(df, args.output_dir)
        
        print("Generating comparative summary...")
        generate_comparative_summary(df, args.output_dir)
        
        print("Creating detailed report...")
        create_detailed_report(df, analysis, args.output_dir)
        
        print(f"\nVisualization complete! Reports saved to: {args.output_dir}")
        print("Generated files:")
        print("  - benchmark_report.html (main report)")
        print("  - performance_comparison.png")
        print("  - prediction_analysis.png") 
        print("  - checkpoint_comparison.png")
        print("  - benchmark_summary.png")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())