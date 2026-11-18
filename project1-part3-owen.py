# Owen Catron
# CSCI 270 PROJECT 1 PART 3 SUBMISSION

# Import necessary libraries
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import os

# Acquire directory to identify where the CSV is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Mall_Customers.csv')

# initialize variables to be used for iterative K values on both clusters
K_range = range(2, 11)

# 1. Age vs. Spending Score (1-100)

inertias1 = [] # for the elbow method in the first clustering example
kmeans1_k5 = None # initializes the variable to save for later when graphing the best K

# reads CSV and assigns to a variable
df1 = pd.read_csv(csv_path, usecols=['Age', 'Spending Score (1-100)'])
X = df1

print(f"For the Age vs. Spending Score model, here are the k values 2-10.")

for k in K_range:
    # initializes K-Means model
    kmeans1 = KMeans(n_clusters=k, random_state=42)
    kmeans1.fit(X)

    inertias1.append(kmeans1.inertia_) # tracking inertia for the elbow curve

    score1 = silhouette_score(X, kmeans1.labels_) # calculates silhouette score

    print(f"Cluster centroids:\n{kmeans1.cluster_centers_}")
    print(f"K-value: {k}")
    print(f"Silhouette Score: {score1:.3f}")

    if k == 5:
        kmeans1_k5 = kmeans1 # recalls the variable and assigns to optimal K of 5
        silhouette1_k5 = score1 # saves for silhouette comparison

    if k == 6:
        silhouette1_k6 = score1 # saves for silhouette comparison

# sets up first window to display scatter (K-Means) and elbow method (inertia vs K)
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5)) 
fig1.canvas.manager.set_window_title("Figure 1: Age vs. Spending Score")

# Scatter of first clustering
axes1[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans1_k5.labels_, cmap='rainbow')
axes1[0].scatter(kmeans1_k5.cluster_centers_[:, 0], 
kmeans1_k5.cluster_centers_[:, 1], s=300, marker='X', c='black', label='Centroids')
axes1[0].set_title("K-Means Clustering (K=5): Age vs. Spending Score")
axes1[0].set_xlabel('Age')
axes1[0].set_ylabel('Spending Score (1-100)')
axes1[0].legend()

# Elbow method for first clustering
axes1[1].plot(K_range, inertias1, 'bo-')
axes1[1].set_title('Elbow Method for Optimal K in Age vs. Spending Score')
axes1[1].set_xlabel('Number of clusters (K)')
axes1[1].set_ylabel('Inertia')
axes1[1].grid(True)

plt.tight_layout()
plt.show() 

# 2. Annual Income (k$) vs. Spending Score (1-100)

inertias2 = [] # for the elbow method in the second clustering example
kmeans2_k5 = None # initializes the variable to save for later when graphing the best K

# reads CSV and assigns to a variable
df2 = pd.read_csv(csv_path, usecols=['Annual Income (k$)', 'Spending Score (1-100)'])
y = df2

print(f"\nFor the Annual Income vs. Spending Score model, here are the k values 2-10.")

for k in K_range:
    # initializes second K-Means model
    kmeans2 = KMeans(n_clusters=k, random_state=42)
    kmeans2.fit(y)

    inertias2.append(kmeans2.inertia_) # tracking inertia for the elbow curve

    score2 = silhouette_score(y, kmeans2.labels_) # calculates silhouette score

    print(f"Cluster centroids:\n{kmeans2.cluster_centers_}")
    print(f"K-value: {k}")
    print(f"Silhouette Score: {score2:.3f}")
    
    if k == 5:
        kmeans2_k5 = kmeans2 # recalls variable and assigns to optimal K of 5
        silhouette2_k5 = score2 # saves for silhouette comparison

    if k == 6:
        silhouette2_k6 = score2 # saves for silhouette comparison

# sets up second window to display scatter (K-Means) and elbow method (inertia vs K)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.canvas.manager.set_window_title("Figure 2: Annual Income vs. Spending Score")

# Scatter of second clustering
axes2[0].scatter(y.iloc[:, 0], y.iloc[:, 1], c=kmeans2_k5.labels_, cmap='rainbow')
axes2[0].scatter(kmeans2_k5.cluster_centers_[:, 0], 
kmeans2_k5.cluster_centers_[:, 1], s=300, marker='X', c='black', label='Centroids')
axes2[0].set_title("K-Means Clustering (K=5): Annual Income vs. Spending Score")
axes2[0].set_xlabel('Annual Income (k$)')
axes2[0].set_ylabel('Spending Score (1-100)')
axes2[0].legend()

# Elbow method for second clustering
axes2[1].plot(K_range, inertias2, 'bo-')
axes2[1].set_title('Elbow Method for Optimal K in Annual Income vs. Spending Score')
axes2[1].set_xlabel('Number of clusters (K)')
axes2[1].set_ylabel('Inertia')
axes2[1].grid(True)

plt.tight_layout()
plt.show() 

# 3. Silhouette Score, Optimal K Value, and Business Pitch Justifications

# definition and formula
print(f"\n--- Silhouette Score Definition ---")
print(f"\nThe silhouette score determines how dense and correllated the data points are for each cluster, "
    f"and also measures how well each cluster is separated from its nearest neighbors.")
print(f"The formula is given by the difference between separation and cohesion "
    f"over the maximum of the two values.")

# importance of comparison
print(f"\n--- Importance of Silhouette Comparisons ---")
print(f"\nWhen comparing the silhouette scores, it is relevant to consider the higher value "
    f"of each K-Means integer when comparing against K = 5 and K = 6.")
print(f"The higher the silhouette score decimal (closest to 1) indicates how strongly the clusters "
    f"are separated, and how common it is for clusters to overlap with other clusters.")

# identifying each silhouette score value to determine which clustering is ideal
print(f"\n--- Silhouette Score Value Comparisons ---")
print(f"\nFor K = 5 in the Age vs. Spending Score model, the silhouette score is: {silhouette1_k5:.3f}"
    f"\nFor K = 6 in the Age vs. Spending Score model, the silhouette score is: {silhouette1_k6:.3f}")
print(f"For K = 5 in the Annual Income vs. Spending Score model, the silhouette score is: {silhouette2_k5:.3f}"
    f"\nFor K = 6 in the Annual Income vs. Spending Score model, the silhouette score is: {silhouette2_k6:.3f}")

# technical justification of K = 5 vs K = 6 for both clustering samples
print(f"\n--- Selected Optimized K Value Technical Justification ---")
print(f"\nWe chose K = 5 for both the Age vs. Spending Score and the Annual Income vs. Spending Score clustering models.")
print(f"As shown in the comparison for both clustering models, none of the colored data clusters"
f" have significantly noticeable overlap with one another.")
print(f"The silhouette scores for both clustering models are higher for K = 5 when compared to K = 6.")
print(f"Both are close to being reasonable for explaining the distribution of their clustering, "
f"although the borders in each model are visually narrow. Even still, there is no overlap, unlike minor outliers or misclassification in K = 6.")
print(f"The most important observation made comes with looking at the elbow method curve in Annual Income vs. Spending Score.")
print(f"This curve has steep rates of decline between integers K = 2 and K = 5, until the ranges of K = 5 to K = 10.")
print(f"This is the purpose of the elbow curve, to locate where"
" the rate of decline begins to diminish, which indicates the data point with the optimal K value.")
print(f"While the Age vs. Spending Score model observes an optimal K of 4 or 6, the stark falloff "
f"between Income and Score in terms of inertia vs. K-values is more defined.")
print(f"This constitutes for an argumentative technical analysis for either K = 5 or K = 6, although the "
    f"insights from being a mall owner have greater support to K = 5 for its narrowly defined business strategy.")

# BUSINESS PITCH: explaining the K = 5 decision on number of clusters for models
print(f"\n--- BUSINESS DEFENSE PITCH: K = 5 CLUSTERS ---")
print(f"\nWhen considering modeling the data for Age and Annual Income, each against Spending Score,"
    f" we need to recognize the target demographics for the marketing team to develop a competitive strategy.")
print(f"Viewing the Age vs. Spending Score graph at K = 5 clusters helps us, the analysts, realize portions "
    f"of the data whose locations have feature similarities by concise sectioning in the scatter plot.")
print(f"The red, blue, yellow, purple, and green clusters have centroids that can easily categorize "
    f"and have easier interpretability of this data. Having any additional cluster counts (e.g. 6)"
    f" would be a broader model of analyzing the correlation between two features, which are more likely "
    f"to contain overlap, meaning an unclear grouping of characteristics. This means a harder time appealing customers in each group.")
print(f"Notably, for the centroids of the Annual Income vs. Spending score graph, a vast majority have "
    f"greater point distance variations, with the exception of the purple cluster.")
print(f"By this example, it is important for the marketing team to identify closely related variables for their customer bases,"
    f" which is why it can be determined for them that most customers of a salary of $40,000 to approx. $65,000 "
    f"would have a spending score of 40 to approx. 60. As the mall owner, we need to trust that the marketing team "
    f"will be able to notice patterns in the data for segmentation and advertising purposes, like with this example.")
print(f"To give another example, on the Age. vs. Spending Score model, the general patterns seem to outline how "
    f"the younger demographics tend to have a higher spending score, while older generations would be less "
    f"inclined to spend, thus having a lower spending score. This becomes important to note for determining frequent mall visitors.")
print(f"The scatter data, as it stands, have easy to distinguish clusters at a K score of 5.")
print(f"More clusters would correspond to the marketing team having to focus on additional, possibly redundant "
    f"generalizations of features, leading to a potentially less effective competitive strategy.")
print(f"While not shown, when analyzing the K = 6 scatter plots, the data points already begin to become potentially mismatched - "
    f"not only having overlap, but miscolored points as well, only to justify the need for a specific number of circular clusters.")
print(f"From a business viewpoint, K = 6 leaves the data harder to distinguish.\n")

print(f"\nIn short, K = 5 is a more valuable representation for a marketing team to develop a business strategy.")
print(f"The 5 clusters are articulated so that we can outline their demographics as well as their quantity of visitors, "
    f"so the outcomes for the strategy (advertising, coupons, pricing of vendors or stores) can ultimately attract more customers,"
    f" based on the data in the scatter plots that have been gathered so far.")