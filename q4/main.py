import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)

print("=" * 80)
print("PART 1: PCA vs Metric MDS vs Non-Metric MDS on Wine Dataset")
print("=" * 80)

# Load Wine dataset (13 dimensional features, 3 classes)
wine = load_wine()
X_wine = wine.data
y_wine = wine.target
wine_names = wine.target_names

print(f"\nWine Dataset Info:")
print(f"- Shape: {X_wine.shape}")
print(f"- Features: {wine.feature_names}")
print(f"- Classes: {wine_names}")
print(f"- Class distribution: {np.bincount(y_wine)}")

# Standardize the data (important for distance-based methods)
scaler = StandardScaler()
X_wine_scaled = scaler.fit_transform(X_wine)

print("\n" + "-" * 80)
print("Applying Dimensionality Reduction Methods...")
print("-" * 80)

# 1. PCA (Principal Component Analysis)
print("\n1. PCA - Linear dimensionality reduction")
print("   - Preserves global variance structure")
print("   - Fast computation")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_wine_scaled)
explained_var = pca.explained_variance_ratio_
print(f"   - Explained variance: {explained_var[0]:.3f} + {explained_var[1]:.3f} = {sum(explained_var):.3f}")

# 2. Metric MDS (Classical MDS / PCoA)
print("\n2. Metric MDS - Preserves exact distances")
print("   - Uses Euclidean distances")
print("   - Maintains metric properties")
mds_metric = MDS(n_components=2, metric=True, dissimilarity='euclidean', 
                 random_state=42, n_init=10, max_iter=300)
X_mds_metric = mds_metric.fit_transform(X_wine_scaled)
stress_metric = mds_metric.stress_
print(f"   - Stress value: {stress_metric:.2f}")

# 3. Non-Metric MDS (Preserves rank order of distances)
print("\n3. Non-Metric MDS - Preserves rank order of distances")
print("   - More flexible than metric MDS")
print("   - Focuses on ordinal relationships")
mds_nonmetric = MDS(n_components=2, metric=False, dissimilarity='euclidean',
                    random_state=42, n_init=10, max_iter=300)
X_mds_nonmetric = mds_nonmetric.fit_transform(X_wine_scaled)
stress_nonmetric = mds_nonmetric.stress_
print(f"   - Stress value: {stress_nonmetric:.2f}")

# Plot all three methods side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', '^']

methods = [
    (X_pca, 'PCA', f'Explained Var: {sum(explained_var):.1%}'),
    (X_mds_metric, 'Metric MDS', f'Stress: {stress_metric:.2f}'),
    (X_mds_nonmetric, 'Non-Metric MDS', f'Stress: {stress_nonmetric:.2f}')
]

for idx, (X_transformed, title, subtitle) in enumerate(methods):
    ax = axes[idx]
    
    for i, (class_idx, class_name) in enumerate(zip(range(3), wine_names)):
        mask = y_wine == class_idx
        ax.scatter(X_transformed[mask, 0], X_transformed[mask, 1],
                  c=colors[i], label=class_name, marker=markers[i],
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Component 2', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/pca_vs_mds_wine.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to ./outputs/pca_vs_mds_wine.png")

# Calculate separation metrics
def calculate_separation(X, y):
    """Calculate between-class and within-class distances"""
    between_dists = []
    within_dists = []
    
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            dist = np.linalg.norm(X[i] - X[j])
            if y[i] == y[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)
    
    return np.mean(between_dists), np.mean(within_dists)

print("\n" + "-" * 80)
print("CLUSTER SEPARATION ANALYSIS")
print("-" * 80)

for X_transformed, method_name in [(X_pca, 'PCA'), 
                                    (X_mds_metric, 'Metric MDS'), 
                                    (X_mds_nonmetric, 'Non-Metric MDS')]:
    between, within = calculate_separation(X_transformed, y_wine)
    ratio = between / within
    print(f"\n{method_name}:")
    print(f"  - Between-class distance: {between:.3f}")
    print(f"  - Within-class distance:  {within:.3f}")
    print(f"  - Separation ratio:       {ratio:.3f} (higher = better separation)")

print("\n" + "=" * 80)
print("PART 2: Stress Function Analysis on Iris Dataset")
print("=" * 80)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"\nIris Dataset Info:")
print(f"- Shape: {X_iris.shape}")
print(f"- Original dimensions: {X_iris.shape[1]}")

# Standardize
X_iris_scaled = scaler.fit_transform(X_iris)

# Calculate stress for different dimensions
dimensions = range(1, 5)
stress_values = []

print("\nCalculating stress for dimensions 1-4...")
for n_dim in dimensions:
    mds = MDS(n_components=n_dim, metric=False, dissimilarity='euclidean',
              random_state=42, n_init=10, max_iter=300)
    mds.fit(X_iris_scaled)
    stress_values.append(mds.stress_)
    print(f"  Dimensions: {n_dim}, Stress: {mds.stress_:.4f}")

# Plot stress vs dimensions
plt.figure(figsize=(10, 6))
plt.plot(dimensions, stress_values, 'o-', linewidth=2, markersize=10, 
         color='#E74C3C', markerfacecolor='white', markeredgewidth=2)

for i, (dim, stress) in enumerate(zip(dimensions, stress_values)):
    plt.annotate(f'{stress:.2f}', 
                xy=(dim, stress), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=11,
                fontweight='bold')

plt.xlabel('Number of Dimensions', fontsize=14, fontweight='bold')
plt.ylabel('Stress Value', fontsize=14, fontweight='bold')
plt.title('Non-Metric MDS: Stress vs Number of Dimensions\n(Iris Dataset)', 
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(dimensions)

# Add shaded region to show improvement
plt.fill_between(dimensions, stress_values, alpha=0.2, color='#E74C3C')

plt.tight_layout()
plt.savefig('./outputs/stress_vs_dimensions.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to ./outputs/stress_vs_dimensions.png")

# Analysis and conclusions
print("\n" + "=" * 80)
print("CONCLUSIONS AND ANALYSIS")
print("=" * 80)

print("\n📊 WHICH METHOD SHOWS CLEARER SEPARATION?")
print("-" * 80)

# Calculate which method has best separation
methods_data = [
    ('PCA', X_pca),
    ('Metric MDS', X_mds_metric),
    ('Non-Metric MDS', X_mds_nonmetric)
]

best_method = None
best_ratio = 0

for name, X_transformed in methods_data:
    between, within = calculate_separation(X_transformed, y_wine)
    ratio = between / within
    if ratio > best_ratio:
        best_ratio = ratio
        best_method = name

print(f"\n🏆 WINNER: {best_method} (Separation Ratio: {best_ratio:.3f})")
