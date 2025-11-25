#!/usr/bin/env python3
"""
Main entry point for analyzing GPT-4o-mini Code Contests results.

This script runs the complete analysis pipeline:
1. Loads and analyzes the results
2. Generates visualizations
3. Creates a summary report

Usage:
    python Analysis/run_full_analysis.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PYTHON = sys.executable


def run_script(script_name: str, description: str):
    """Run a Python script and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    script_path = SCRIPT_DIR / script_name
    result = subprocess.run(
        [PYTHON, str(script_path)],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\n⚠️  Warning: {script_name} exited with code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed successfully!")
    return True


def main():
    """Run the complete analysis pipeline."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              GPT-4o-mini Code Contests - Full Analysis Pipeline            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    
    success_count = 0
    total_count = 3
    
    # Step 1: Statistical Analysis
    if run_script("analyze_gpt4o_mini_results.py", "Statistical Analysis"):
        success_count += 1
    
    # Step 2: Visualizations
    if run_script("visualize_gpt4o_mini_results.py", "Visualization Generation"):
        success_count += 1
    
    # Step 3: Quick Reference
    if run_script("quick_reference.py", "Quick Reference Guide"):
        success_count += 1
    
    # Final Summary
    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {success_count}/{total_count} steps")
    
    if success_count == total_count:
        print("\n✅ Full analysis pipeline completed successfully!")
        print(f"\nResults saved to: {SCRIPT_DIR / 'analysis_outputs'}")
        print("\nGenerated files:")
        print("  📄 analysis_outputs/analysis_report.txt - Text report")
        print("  📊 analysis_outputs/analyzed_results.csv - Enhanced dataset")
        print("  📝 analysis_outputs/success_examples.json - Example successes")
        print("  📈 analysis_outputs/visualizations/ - Charts and graphs")
        print("\nVisualization files:")
        print("  🎨 overall_summary.png - Overview dashboard")
        print("  📊 difficulty_analysis.png - Performance by difficulty")
        print("  🌍 source_analysis.png - Performance by source")
        print("  📏 code_complexity.png - Code metrics")
        print("  📋 statistics_table.png - Statistics summary")
    else:
        print(f"\n⚠️  Pipeline completed with {total_count - success_count} warning(s)")
    
    print(f"{'='*80}\n")
    
    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
