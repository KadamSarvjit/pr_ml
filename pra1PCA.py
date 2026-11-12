import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv"
df = pd.read_csv(url)

print("First five rows of the dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

y = df.iloc[:, 0]
X = df.iloc[:, 1:]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

result = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
result["Class"] = y.to_numpy()

print("\nTransformed data:")
print(result.head())

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

total_variance = sum(pca.explained_variance_ratio_) * 100
print("Total variance captured:", round(total_variance, 2), "%")

plt.figure(figsize=(8, 6))

for label in result["Class"].unique():
    subset = result[result["Class"] == label]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Class {label}",
        alpha=0.7
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_ * 100
)
plt.xlabel("Principal Components")
plt.ylabel("Variance Explained (%)")
plt.title("PCA Explained Variance")
plt.tight_layout()
plt.show()

print("\nPCA analysis completed successfully!")
