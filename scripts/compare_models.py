#!/usr/bin/env python
"""
Compare results across different model runs.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import compare_model_runs, generate_comparison_report
from src.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple model runs on the same experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  python scripts/compare_models.py \\
    --experiment mouse_opossum \\
    --runs results/mouse_opossum/scgpt_Nov01-14-30 \\
           results/mouse_opossum/seurat_mapquery_Nov01-15-45 \\
    --output results/mouse_opossum/comparisons
  
  # With custom name
  python scripts/compare_models.py \\
    --experiment my_comparison \\
    --runs results/mouse_opossum/scgpt_* \\
    --output results/comparisons
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Name for the comparison experiment'
    )
    
    parser.add_argument(
        '--runs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to model run directories to compare'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for comparison results'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate markdown report'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    run_dirs = [Path(r) for r in args.runs]
    output_dir = Path(args.output)
    
    # Validate run directories
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
    
    # Setup logger
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('comparison', output_dir)
    
    logger.info(f"Comparing {len(run_dirs)} model runs")
    logger.info(f"Experiment name: {args.experiment}")
    
    # Run comparison
    comparison_results = compare_model_runs(
        run_dirs=run_dirs,
        output_dir=output_dir,
        experiment_name=args.experiment,
        logger=logger
    )
    
    # Generate report if requested
    if args.report:
        report_path = generate_comparison_report(
            run_dirs=run_dirs,
            output_dir=output_dir,
            experiment_name=args.experiment,
            logger=logger
        )
        logger.info(f"Report generated: {report_path}")
    
    print(f"\nComparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()