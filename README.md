Movie Recommendation System | MovieLens 100K

A recommendation system that predicts and suggests movies to users based on their preferences and similarity to other users.
Built with collaborative filtering techniques and evaluated on the MovieLens 100K dataset.

📂 Dataset

MovieLens 100K dataset containing:

100,000 ratings (scale 1–5)

943 users

1,682 movies

Official source: GroupLens MovieLens Dataset

Files used:

u.data → ratings (user_id, item_id, rating, timestamp)

u.item → movie details (id, title, genres)

u1.base / u1.test → predefined train/test splits

⚙️ Features

✔️ User-based Collaborative Filtering (cosine similarity between users)
✔️ Item-based Collaborative Filtering (cosine similarity between items)
✔️ Matrix Factorization with SVD (latent feature approach)
✔️ Evaluation using Precision@K metric
✔️ Command-line interface for flexible experimentation

🛠️ Tools & Libraries

Python 3.x

NumPy

Pandas

Scikit-learn
