#!/usr/bin/env python3
"""Test runner for all three questions with basic parameters."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False


def main():
    """Run all three questions with test parameters."""
    print("Deep Learning Lab - Test Runner")
    print("Running all three questions with small test parameters...")
    
    # Ensure we're in a virtual environment
    venv_python = Path(".venv/Scripts/python.exe")
    if not venv_python.exists():
        print("Virtual environment not found. Please run from project root with .venv activated.")
        sys.exit(1)
    
    python_cmd = str(venv_python)
    
    # Test Q1: CIFAR-10 PPCA (small sample)
    q1_success = run_command([
        python_cmd, "-m", "q1.main",
        "--latents", "16", "32",
        "--sample-size", "500",
        "--ppca-iters", "50",
        "--out-dir", "outputs/test_q1"
    ], "Q1: PPCA Image Compression (CIFAR-10)")
    
    # Test Q2: PPCA Missing Data (small dataset)
    q2_success = run_command([
        python_cmd, "-m", "q2.main",
        "--n-samples", "200",
        "--dim", "20",
        "--latent", "5",
        "--missing-frac", "0.1"
    ], "Q2: PPCA with Missing Data")
    
    # Test Q3: ICA Audio Separation (synthetic)
    q3_success = run_command([
        python_cmd, "-m", "q3.main",
        "--use-synthetic",
        "--duration", "2.0",
        "--output-dir", "outputs/test_q3"
    ], "Q3: ICA Blind Source Separation (Synthetic)")
    
    # Test Q4: PCA vs MDS Comparison
    q4_success = run_command([
        python_cmd, "-m", "q4.main",
        "--n-components", "1", "2", "3", "4",
        "--output-dir", "outputs/test_q4"
    ], "Q4: PCA vs Metric/Non-Metric MDS Comparison")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Q1 (PPCA Image Compression):  {'✓ PASS' if q1_success else '✗ FAIL'}")
    print(f"Q2 (PPCA Missing Data):       {'✓ PASS' if q2_success else '✗ FAIL'}")
    print(f"Q3 (ICA Audio Separation):    {'✓ PASS' if q3_success else '✗ FAIL'}")
    print(f"Q4 (PCA vs MDS Comparison):   {'✓ PASS' if q4_success else '✗ FAIL'}")
    
    total_passed = sum([q1_success, q2_success, q3_success, q4_success])
    print(f"\nOverall: {total_passed}/4 tests passed")
    
    if total_passed == 4:
        print("\n🎉 All tests passed! Lab implementation complete.")
    else:
        print(f"\n⚠️  {4-total_passed} test(s) failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
