#!/usr/bin/env python
"""
CLI entry point for running cross-species label transfer experiments.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import CrossSpeciesLabelTransferPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-species label transfer experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment with config file
  python scripts/run_experiment.py --config configs/experiments/mouse_to_opossum.yaml
  
  # Specify GPU
  python scripts/run_experiment.py --config configs/experiments/mouse_to_opossum.yaml --gpu 0
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML file'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use (optional)'
    )
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Run pipeline
    print(f"Starting experiment from config: {config_path}")
    pipeline = CrossSpeciesLabelTransferPipeline(config_path)
    pipeline.run()
    
    print(f"\nExperiment complete! Results saved to: {pipeline.save_dir}")


if __name__ == "__main__":
    main()