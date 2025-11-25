"""
Visualization script for GPT-4o-mini Code Contests analysis results.

Generates charts and visualizations to better understand:
- Success rates by difficulty and source
- Code complexity distributions
- Error patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuration
ANALYSIS_DIR = Path(__file__).parent / "analysis_outputs"
RESULTS_CSV = ANALYSIS_DIR / "analyzed_results.csv"
OUTPUT_DIR = ANALYSIS_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_analyzed_data() -> pd.DataFrame:
    """Load the analyzed results."""
    print(f"Loading analyzed data from {RESULTS_CSV}...")
    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} rows")
    return df


def plot_success_rates_by_difficulty(df: pd.DataFrame):
    """Plot success rates by difficulty level."""
    print("Generating difficulty success rate plot...")
    
    difficulty_stats = df.groupby('difficulty_name').agg({
        'is_valid_syntax': ['count', 'mean']
    }).reset_index()
    difficulty_stats.columns = ['difficulty', 'count', 'success_rate']
    difficulty_stats = difficulty_stats.sort_values('count', ascending=False).head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count by difficulty
    ax1.bar(range(len(difficulty_stats)), difficulty_stats['count'], color='steelblue')
    ax1.set_xticks(range(len(difficulty_stats)))
    ax1.set_xticklabels(difficulty_stats['difficulty'], rotation=45, ha='right')
    ax1.set_xlabel('Difficulty Level')
    ax1.set_ylabel('Number of Problems')
    ax1.set_title('Problem Count by Difficulty')
    ax1.grid(axis='y', alpha=0.3)
    
    # Success rate by difficulty
    colors = ['green' if rate == 1.0 else 'orange' if rate > 0.9 else 'red' 
              for rate in difficulty_stats['success_rate']]
    ax2.bar(range(len(difficulty_stats)), difficulty_stats['success_rate'], color=colors)
    ax2.set_xticks(range(len(difficulty_stats)))
    ax2.set_xticklabels(difficulty_stats['difficulty'], rotation=45, ha='right')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Syntax Validity Rate by Difficulty')
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'difficulty_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved to {OUTPUT_DIR / 'difficulty_analysis.png'}")
    plt.close()


def plot_success_rates_by_source(df: pd.DataFrame):
    """Plot success rates by problem source."""
    print("Generating source success rate plot...")
    
    source_stats = df.groupby('source_name').agg({
        'is_valid_syntax': ['count', 'mean']
    }).reset_index()
    source_stats.columns = ['source', 'count', 'success_rate']
    source_stats = source_stats.sort_values('count', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count by source
    ax1.barh(range(len(source_stats)), source_stats['count'], color='coral')
    ax1.set_yticks(range(len(source_stats)))
    ax1.set_yticklabels(source_stats['source'])
    ax1.set_xlabel('Number of Problems')
    ax1.set_title('Problem Count by Source')
    ax1.grid(axis='x', alpha=0.3)
    
    # Success rate by source
    colors = ['green' if rate == 1.0 else 'orange' if rate > 0.9 else 'red' 
              for rate in source_stats['success_rate']]
    ax2.barh(range(len(source_stats)), source_stats['success_rate'], color=colors)
    ax2.set_yticks(range(len(source_stats)))
    ax2.set_yticklabels(source_stats['source'])
    ax2.set_xlabel('Success Rate')
    ax2.set_title('Syntax Validity Rate by Source')
    ax2.set_xlim([0, 1.05])
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'source_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved to {OUTPUT_DIR / 'source_analysis.png'}")
    plt.close()


def plot_code_complexity_distribution(df: pd.DataFrame):
    """Plot distributions of code complexity metrics."""
    print("Generating code complexity distributions...")
    
    valid_df = df[df['is_valid_syntax']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Lines of code distribution
    axes[0, 0].hist(valid_df['solution_line_count'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Lines of Code')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of Lines of Code (mean={valid_df["solution_line_count"].mean():.1f})')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Character count distribution
    axes[0, 1].hist(valid_df['solution_char_count'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Character Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Distribution of Character Count (mean={valid_df["solution_char_count"].mean():.1f})')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Code features
    features = ['solution_has_imports', 'solution_has_functions', 'solution_has_loops', 
                'solution_has_conditionals']
    feature_labels = ['Imports', 'Functions', 'Loops', 'Conditionals']
    feature_counts = [valid_df[f].sum() for f in features]
    
    axes[1, 0].bar(feature_labels, feature_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    axes[1, 0].set_ylabel('Number of Solutions')
    axes[1, 0].set_title('Presence of Code Features')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, (label, count) in enumerate(zip(feature_labels, feature_counts)):
        axes[1, 0].text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Lines of code by difficulty
    difficulty_order = valid_df.groupby('difficulty_name')['solution_line_count'].mean().sort_values(ascending=False).head(10).index
    data_for_box = [valid_df[valid_df['difficulty_name'] == d]['solution_line_count'].values 
                    for d in difficulty_order]
    
    bp = axes[1, 1].boxplot(data_for_box, labels=difficulty_order, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[1, 1].set_xticklabels(difficulty_order, rotation=45, ha='right')
    axes[1, 1].set_xlabel('Difficulty')
    axes[1, 1].set_ylabel('Lines of Code')
    axes[1, 1].set_title('Code Length by Difficulty (Top 10)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'code_complexity.png', dpi=300, bbox_inches='tight')
    print(f"Saved to {OUTPUT_DIR / 'code_complexity.png'}")
    plt.close()


def plot_overall_summary(df: pd.DataFrame):
    """Create an overall summary visualization."""
    print("Generating overall summary...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPT-4o-mini Code Contests: Overall Summary', fontsize=16, fontweight='bold')
    
    # Success pie chart
    success_counts = df['is_valid_syntax'].value_counts()
    colors_pie = ['#2ecc71', '#e74c3c']
    axes[0, 0].pie(success_counts.values, labels=['Valid Syntax', 'Invalid Syntax'], 
                   autopct='%1.1f%%', colors=colors_pie, startangle=90)
    axes[0, 0].set_title(f'Syntax Validity\n({success_counts[True]} / {len(df)} valid)')
    
    # Error count
    error_count = df['has_error'].sum()
    no_error_count = len(df) - error_count
    axes[0, 1].bar(['No Errors', 'Has Errors'], [no_error_count, error_count], 
                   color=['green', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel('Number of Problems')
    axes[0, 1].set_title('Generation Errors')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate([no_error_count, error_count]):
        axes[0, 1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Top difficulties
    top_difficulties = df['difficulty_name'].value_counts().head(10)
    axes[1, 0].barh(range(len(top_difficulties)), top_difficulties.values, color='steelblue')
    axes[1, 0].set_yticks(range(len(top_difficulties)))
    axes[1, 0].set_yticklabels(top_difficulties.index)
    axes[1, 0].set_xlabel('Number of Problems')
    axes[1, 0].set_title('Top 10 Difficulty Levels')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Code length histogram
    valid_df = df[df['is_valid_syntax']]
    axes[1, 1].hist(valid_df['solution_line_count'], bins=20, color='orange', 
                    edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(valid_df['solution_line_count'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {valid_df["solution_line_count"].mean():.1f}')
    axes[1, 1].axvline(valid_df['solution_line_count'].median(), color='blue', 
                       linestyle='--', linewidth=2, label=f'Median: {valid_df["solution_line_count"].median():.1f}')
    axes[1, 1].set_xlabel('Lines of Code')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Solution Length')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'overall_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved to {OUTPUT_DIR / 'overall_summary.png'}")
    plt.close()


def generate_statistics_table(df: pd.DataFrame):
    """Generate a detailed statistics table as an image."""
    print("Generating statistics table...")
    
    valid_df = df[df['is_valid_syntax']]
    
    stats = [
        ['Metric', 'Value'],
        ['', ''],
        ['Total Problems', f'{len(df)}'],
        ['Valid Syntax', f'{len(valid_df)} ({100*len(valid_df)/len(df):.1f}%)'],
        ['Generation Errors', f'{df["has_error"].sum()} ({100*df["has_error"].sum()/len(df):.1f}%)'],
        ['', ''],
        ['Avg Lines of Code', f'{valid_df["solution_line_count"].mean():.1f}'],
        ['Median Lines of Code', f'{valid_df["solution_line_count"].median():.1f}'],
        ['Max Lines of Code', f'{valid_df["solution_line_count"].max():.0f}'],
        ['Min Lines of Code', f'{valid_df["solution_line_count"].min():.0f}'],
        ['', ''],
        ['Solutions with Imports', f'{valid_df["solution_has_imports"].sum()} ({100*valid_df["solution_has_imports"].mean():.1f}%)'],
        ['Solutions with Functions', f'{valid_df["solution_has_functions"].sum()} ({100*valid_df["solution_has_functions"].mean():.1f}%)'],
        ['Solutions with Classes', f'{valid_df["solution_has_classes"].sum()} ({100*valid_df["solution_has_classes"].mean():.1f}%)'],
        ['Solutions with Loops', f'{valid_df["solution_has_loops"].sum()} ({100*valid_df["solution_has_loops"].mean():.1f}%)'],
        ['Solutions with Conditionals', f'{valid_df["solution_has_conditionals"].sum()} ({100*valid_df["solution_has_conditionals"].mean():.1f}%)'],
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=stats, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(stats)):
        if stats[i][0] == '':
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0')
        else:
            for j in range(2):
                table[(i, j)].set_facecolor('#ffffff' if i % 2 == 0 else '#f9f9f9')
    
    plt.title('GPT-4o-mini Code Contests: Detailed Statistics', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / 'statistics_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved to {OUTPUT_DIR / 'statistics_table.png'}")
    plt.close()


def main():
    """Main visualization pipeline."""
    print("Starting GPT-4o-mini Code Contests Visualization")
    print("=" * 80)
    
    # Load data
    df = load_analyzed_data()
    
    # Generate visualizations
    plot_overall_summary(df)
    plot_success_rates_by_difficulty(df)
    plot_success_rates_by_source(df)
    plot_code_complexity_distribution(df)
    generate_statistics_table(df)
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
