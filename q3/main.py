"""Question 3: Blind Source Separation using ICA

Implements:
1. Audio source loading or synthetic signal generation
2. Random mixing of sources
3. FastICA separation
4. Evaluation via correlation and visualization

Usage:
    python -m q3.main --use-synthetic --duration 5
    python -m q3.main --audio-dir data/audio --sources source1.wav source2.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA


def generate_synthetic_sources(duration: float = 5.0, sample_rate: int = 16000, seed: int = 0) -> Tuple[np.ndarray, int]:
    """Generate synthetic audio sources for demonstration."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Source 1: Pure sine wave (musical tone)
    freq1 = 440.0  # A4 note
    source1 = np.sin(2 * np.pi * freq1 * t)
    
    # Source 2: Modulated signal (more complex)
    freq2 = 220.0  # A3 note
    mod_freq = 3.0  # 3 Hz modulation
    source2 = np.sin(2 * np.pi * freq2 * t) * (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))
    
    # Add some noise for realism
    source1 += 0.1 * rng.normal(size=n_samples)
    source2 += 0.1 * rng.normal(size=n_samples)
    
    # Normalize
    source1 = source1 / np.max(np.abs(source1))
    source2 = source2 / np.max(np.abs(source2))
    
    sources = np.vstack([source1, source2])
    return sources, sample_rate


def load_audio_sources(audio_dir: Path, source_files: list[str]) -> Tuple[np.ndarray, int]:
    """Load audio sources from WAV files."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile required for audio loading. Install with: pip install soundfile")
    
    sources = []
    sample_rate = None
    
    for filename in source_files:
        filepath = audio_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        audio, sr = sf.read(filepath)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure consistent sample rate
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: {filepath} has {sr}, expected {sample_rate}")
        
        sources.append(audio)
    
    # Trim to shortest length
    min_length = min(len(s) for s in sources)
    sources = [s[:min_length] for s in sources]
    
    # Normalize each source
    sources = [s / np.max(np.abs(s)) if np.max(np.abs(s)) > 0 else s for s in sources]
    
    return np.vstack(sources), sample_rate


def create_mixing_matrix(n_sources: int, seed: int = 0) -> np.ndarray:
    """Create random mixing matrix."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_sources, n_sources))
    return A


def mix_sources(sources: np.ndarray, mixing_matrix: np.ndarray) -> np.ndarray:
    """Mix sources using the mixing matrix."""
    return mixing_matrix @ sources


def separate_sources(mixed_signals: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Separate mixed signals using FastICA."""
    ica = FastICA(
        n_components=mixed_signals.shape[0],
        random_state=seed,
        whiten='unit-variance',
        max_iter=1000,
        tol=1e-4
    )
    
    # FastICA expects (n_samples, n_features), so transpose
    separated = ica.fit_transform(mixed_signals.T).T
    unmixing_matrix = ica.mixing_.T  # Unmixing matrix
    
    return separated, unmixing_matrix


def evaluate_separation(original: np.ndarray, separated: np.ndarray) -> np.ndarray:
    """Evaluate separation quality using correlation matrix."""
    n_sources = original.shape[0]
    correlation_matrix = np.zeros((n_sources, n_sources))
    
    for i in range(n_sources):
        for j in range(n_sources):
            correlation_matrix[i, j] = np.corrcoef(original[i], separated[j])[0, 1]
    
    return correlation_matrix


def plot_signals(original: np.ndarray, mixed: np.ndarray, separated: np.ndarray, 
                sample_rate: int, output_path: Optional[Path] = None):
    """Plot original, mixed, and separated signals."""
    n_sources = original.shape[0]
    n_samples = original.shape[1]
    time = np.arange(n_samples) / sample_rate
    
    # Show only first few seconds for clarity
    max_time = 3.0
    max_samples = int(max_time * sample_rate)
    if n_samples > max_samples:
        time = time[:max_samples]
        original = original[:, :max_samples]
        mixed = mixed[:, :max_samples]
        separated = separated[:, :max_samples]
    
    fig, axes = plt.subplots(3, n_sources, figsize=(12, 8))
    if n_sources == 1:
        axes = axes.reshape(3, 1)
    
    # Plot original sources
    for i in range(n_sources):
        axes[0, i].plot(time, original[i])
        axes[0, i].set_title(f'Original Source {i+1}')
        axes[0, i].set_ylabel('Amplitude')
        if i == 0:
            axes[0, i].set_ylabel('Original\nAmplitude')
    
    # Plot mixed signals
    for i in range(n_sources):
        axes[1, i].plot(time, mixed[i])
        axes[1, i].set_title(f'Mixed Signal {i+1}')
        if i == 0:
            axes[1, i].set_ylabel('Mixed\nAmplitude')
    
    # Plot separated signals
    for i in range(n_sources):
        axes[2, i].plot(time, separated[i])
        axes[2, i].set_title(f'Separated Source {i+1}')
        axes[2, i].set_xlabel('Time (s)')
        if i == 0:
            axes[2, i].set_ylabel('Separated\nAmplitude')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_audio_sources(sources: np.ndarray, sample_rate: int, output_dir: Path, prefix: str = "separated"):
    """Save separated sources as WAV files."""
    try:
        import soundfile as sf
    except ImportError:
        print("Warning: soundfile not available, skipping audio file output")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, source in enumerate(sources):
        # Normalize to prevent clipping
        normalized = source / np.max(np.abs(source)) if np.max(np.abs(source)) > 0 else source
        
        filename = output_dir / f"{prefix}_source_{i+1}.wav"
        sf.write(filename, normalized, sample_rate)
        print(f"Saved: {filename}")


def run_experiment(use_synthetic: bool = True, audio_dir: Optional[str] = None, 
                  source_files: Optional[list[str]] = None, duration: float = 5.0,
                  output_dir: str = "outputs/q3_ica", seed: int = 0):
    """Run the complete ICA experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or generate sources
    if use_synthetic:
        print(f"Generating synthetic sources ({duration}s)")
        sources, sample_rate = generate_synthetic_sources(duration, seed=seed)
    else:
        if not audio_dir or not source_files:
            raise ValueError("audio_dir and source_files required when not using synthetic data")
        
        print(f"Loading audio sources from {audio_dir}")
        sources, sample_rate = load_audio_sources(Path(audio_dir), source_files)
    
    n_sources = sources.shape[0]
    print(f"Loaded {n_sources} sources, sample rate: {sample_rate} Hz")
    
    # Create mixing matrix and mix sources
    print("Creating random mixing matrix and mixing sources...")
    mixing_matrix = create_mixing_matrix(n_sources, seed)
    mixed_signals = mix_sources(sources, mixing_matrix)
    
    print("Mixing matrix:")
    print(mixing_matrix)
    
    # Separate using FastICA
    print("Separating sources using FastICA...")
    separated_sources, unmixing_matrix = separate_sources(mixed_signals, seed)
    
    # Evaluate separation quality
    correlation_matrix = evaluate_separation(sources, separated_sources)
    
    print("\nSeparation Results:")
    print("Correlation matrix (original vs separated):")
    print(correlation_matrix)
    print(f"Absolute correlation matrix:")
    print(np.abs(correlation_matrix))
    
    # Find best permutation based on absolute correlations
    abs_corr = np.abs(correlation_matrix)
    max_corr_per_source = np.max(abs_corr, axis=1)
    mean_max_correlation = np.mean(max_corr_per_source)
    
    print(f"Mean maximum absolute correlation: {mean_max_correlation:.3f}")
    
    # Plot results
    plot_path = output_path / "separation_results.png"
    plot_signals(sources, mixed_signals, separated_sources, sample_rate, plot_path)
    print(f"Saved plot: {plot_path}")
    
    # Save separated audio files
    save_audio_sources(separated_sources, sample_rate, output_path)
    
    return {
        'correlation_matrix': correlation_matrix,
        'mean_max_correlation': mean_max_correlation,
        'mixing_matrix': mixing_matrix,
        'unmixing_matrix': unmixing_matrix
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Blind Source Separation using ICA")
    parser.add_argument('--use-synthetic', action='store_true', default=True,
                       help='Use synthetic sources instead of audio files')
    parser.add_argument('--audio-dir', type=str, default='data/audio',
                       help='Directory containing audio files')
    parser.add_argument('--sources', nargs='+', default=['source1.wav', 'source2.wav'],
                       help='Audio source filenames')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duration for synthetic sources (seconds)')
    parser.add_argument('--output-dir', type=str, default='outputs/q3_ica',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        results = run_experiment(
            use_synthetic=args.use_synthetic,
            audio_dir=args.audio_dir if not args.use_synthetic else None,
            source_files=args.sources if not args.use_synthetic else None,
            duration=args.duration,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {Path(args.output_dir).resolve()}")
        
    except Exception as e:
        print(f"Error: {e}")
        if not args.use_synthetic:
            print("\nTip: Try running with --use-synthetic flag to use generated signals instead of audio files")


if __name__ == "__main__":
    main()
