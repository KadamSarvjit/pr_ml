
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load the dataset
df = pd.read_csv("iris.csv")

print("First five rows of the dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

# Step 2: Select numerical features
columns_to_drop = [
    col for col in ["Id", "ID", "Species"]
    if col in df.columns
]

X = df.drop(columns=columns_to_drop)
X = X.select_dtypes(include="number")
X = X.fillna(X.mean())

print("\nFeatures used for clustering:")
print(X.head())

if X.shape[1] < 4:
    raise ValueError("At least four numerical features are required.")

# Step 3: Apply the Elbow Method
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    model.fit(X)
    wcss.append(model.inertia_)

# Step 4: Plot the Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Step 5: Apply K-Means with optimal K = 3
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(X)

# Step 6: Display cluster information
print("\nCluster assignments:")
print(df[["Cluster"]].head(10))

print("\nNumber of samples in each cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster centroids:")
print(pd.DataFrame(
    kmeans.cluster_centers_,
    columns=X.columns
))

# Step 7: Visualize clusters using petal measurements
plt.figure(figsize=(8, 5))

plt.scatter(
    X.iloc[:, 2],
    X.iloc[:, 3],
    c=df["Cluster"],
    cmap="viridis",
    s=50
)

plt.scatter(
    kmeans.cluster_centers_[:, 2],
    kmeans.cluster_centers_[:, 3],
    marker="X",
    s=200,
    c="red",
    label="Centroids"
)

plt.title("K-Means Clustering of Iris Dataset")
plt.xlabel(X.columns[2])
plt.ylabel(X.columns[3])
plt.legend()
plt.grid(True)
plt.show()

print("\nFinal WCSS:", kmeans.inertia_)
print("\nK-Means Clustering Completed Successfully!")
