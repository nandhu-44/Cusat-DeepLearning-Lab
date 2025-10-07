"""Question 5: Dimensionality Reduction and RBM

Main entry point that runs both parts of Question 5:
1. Manifold learning comparison (t-SNE, LLE, UMAP, Isomap)
2. RBM training and visualization

Usage:
    python -m q5.main --run-all
    python -m q5.main --part1-only
    python -m q5.main --part2-only
"""

import argparse
import sys
from pathlib import Path


def run_part1(dataset: str = 'digits', n_samples: int = None, output_dir: str = 'outputs/q5'):
    """Run Part 1: Manifold learning comparison."""
    from q5.part1_manifold_comparison import run_comparison
    
    print("\n" + "="*80)
    print("PART 1: MANIFOLD LEARNING COMPARISON")
    print("="*80)
    
    run_comparison(
        dataset=dataset,
        n_samples=n_samples,
        output_dir=f"{output_dir}/manifold",
        random_state=0,
        verbose=True
    )


def run_part2(dataset: str = 'toy', n_samples: int = None, epochs: int = 100, 
              n_hidden: int = 64, output_dir: str = 'outputs/q5'):
    """Run Part 2: RBM training."""
    from q5.part2_rbm import run_rbm_experiment
    
    print("\n" + "="*80)
    print("PART 2: RESTRICTED BOLTZMANN MACHINE")
    print("="*80)
    
    run_rbm_experiment(
        dataset=dataset,
        n_samples=n_samples,
        n_hidden=n_hidden,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.1,
        output_dir=f"{output_dir}/rbm",
        random_state=0,
        verbose=True
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Q5: Dimensionality Reduction and RBM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both parts with default settings
  python -m q5.main --run-all
  
  # Run only manifold learning on MNIST subset
  python -m q5.main --part1-only --part1-dataset mnist --part1-samples 5000
  
  # Run only RBM on digits dataset
  python -m q5.main --part2-only --part2-dataset digits --epochs 50
  
  # Run both with custom settings
  python -m q5.main --run-all --part1-dataset digits --part2-dataset toy --epochs 100
        """
    )
    
    # Execution mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run-all', action='store_true', default=True,
                     help='Run both Part 1 and Part 2 (default)')
    mode.add_argument('--part1-only', action='store_true',
                     help='Run only Part 1 (manifold learning)')
    mode.add_argument('--part2-only', action='store_true',
                     help='Run only Part 2 (RBM)')
    
    # Part 1 settings
    parser.add_argument('--part1-dataset', type=str, default='digits',
                       choices=['digits', 'mnist'],
                       help='Dataset for Part 1 (default: digits)')
    parser.add_argument('--part1-samples', type=int, default=None,
                       help='Subsample size for Part 1')
    
    # Part 2 settings
    parser.add_argument('--part2-dataset', type=str, default='toy',
                       choices=['toy', 'digits', 'mnist'],
                       help='Dataset for Part 2 (default: toy)')
    parser.add_argument('--part2-samples', type=int, default=None,
                       help='Subsample size for Part 2')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for RBM training (default: 100)')
    parser.add_argument('--n-hidden', type=int, default=64,
                       help='Number of hidden units for RBM (default: 64)')
    
    # General settings
    parser.add_argument('--output-dir', type=str, default='outputs/q5',
                       help='Output directory (default: outputs/q5)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine what to run
    run_part1_flag = args.part1_only or (not args.part2_only and args.run_all)
    run_part2_flag = args.part2_only or (not args.part1_only and args.run_all)
    
    print("="*80)
    print("QUESTION 5: DIMENSIONALITY REDUCTION AND RBM")
    print("="*80)
    
    # Run Part 1 if requested
    if run_part1_flag:
        try:
            run_part1(
                dataset=args.part1_dataset,
                n_samples=args.part1_samples,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"\n✗ Part 1 failed: {e}")
            if args.part1_only:
                sys.exit(1)
    
    # Run Part 2 if requested
    if run_part2_flag:
        try:
            run_part2(
                dataset=args.part2_dataset,
                n_samples=args.part2_samples,
                epochs=args.epochs,
                n_hidden=args.n_hidden,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"\n✗ Part 2 failed: {e}")
            if args.part2_only:
                sys.exit(1)
    
    # Final summary
    print("\n" + "="*80)
    print("Q5 COMPLETE!")
    print("="*80)
    
    output_path = Path(args.output_dir)
    if run_part1_flag:
        print(f"Part 1 outputs: {(output_path / 'manifold').resolve()}")
    if run_part2_flag:
        print(f"Part 2 outputs: {(output_path / 'rbm').resolve()}")
    
    print("\n✓ All tasks completed successfully!")


if __name__ == "__main__":
    main()