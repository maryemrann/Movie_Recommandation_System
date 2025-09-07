# visualize_recommendations.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(
    r"E:\maryam\Elevvo\Task2\ml-100k\u.data",
    sep='\t',
    names=column_names
)

# Create user-item matrix
ratings = ratings.pivot(index='user_id', columns='movie_id', values='rating')

# Fill missing values with 0 for similarity calculation
ratings_filled = ratings.fillna(0)

# -------------------------------
# 2️⃣ Compute user similarity
# -------------------------------
user_sim = cosine_similarity(ratings_filled)
user_sim_df = pd.DataFrame(user_sim, index=ratings.index, columns=ratings.index)

# Plot heatmap of user similarities (subset of first 20 users)
plt.figure(figsize=(10,8))
sns.heatmap(user_sim_df.iloc[:20, :20], annot=True, fmt=".2f", cmap="coolwarm")
plt.title("User-User Similarity (subset of 20 users)")
plt.show()

# -------------------------------
# 3️⃣ Top-10 recommendations for user 42
# -------------------------------
# Replace this list with actual recommendations from your mr.py script
top_10 = pd.DataFrame({
    "Movie": [
        "Wallace & Gromit", "Manon des sources", "Ace Ventura", "Chasing Amy",
        "Man Who Knew Too Little", "The Women", "In Love and War",
        "To Be or Not to Be", "Paris Is Burning", "The Third Man"
    ],
    "Score": [5,5,5,5,5,5,5,5,5,5]
})

plt.figure(figsize=(8,6))
top_10.plot(kind="barh", x="Movie", y="Score", color="skyblue", legend=False)
plt.xlabel("Predicted Score")
plt.title("Top-10 Movie Recommendations for User 42")
plt.gca().invert_yaxis()  # Highest score on top
plt.show()
