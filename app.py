# ==========================================================
# AI-Powered Recommendation System (Single File Version)
# Author: Your Name
# Description: Content-Based Recommendation System using
# TF-IDF and Cosine Similarity with Flask Web Interface
# ==========================================================

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# Sample Product Dataset
# ----------------------------------------------------------

data = pd.DataFrame({
    "id": [1,2,3,4,5,6,7,8,9,10],
    "title": [
        "Wireless Mouse",
        "Gaming Keyboard",
        "Running Shoes",
        "Cotton T-Shirt",
        "Smartphone",
        "Water Bottle",
        "Yoga Mat",
        "Headphones",
        "Backpack",
        "Sunglasses"
    ],
    "description": [
        "Ergonomic wireless mouse with USB receiver",
        "Mechanical keyboard with RGB lighting",
        "Lightweight running shoes for daily workouts",
        "Comfortable cotton t-shirt for casual wear",
        "Android smartphone with 128GB storage",
        "Insulated stainless steel water bottle",
        "Non-slip yoga mat for home workouts",
        "Noise cancelling over ear headphones",
        "Durable travel backpack with compartments",
        "UV protection stylish sunglasses"
    ],
    "category": [
        "Electronics",
        "Electronics",
        "Fashion",
        "Fashion",
        "Electronics",
        "Sports",
        "Sports",
        "Electronics",
        "Accessories",
        "Accessories"
    ]
})

# ----------------------------------------------------------
# Recommendation Model (TF-IDF + Cosine Similarity)
# ----------------------------------------------------------

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["description"] + " " + data["category"])
similarity_matrix = cosine_similarity(tfidf_matrix)

def get_recommendations(product_title, top_n=5):
    if product_title not in data["title"].values:
        return []

    idx = data[data["title"] == product_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n+1]

    recommended_indices = [i[0] for i in similarity_scores]
    return data.iloc[recommended_indices][["title", "category"]].to_dict("records")

# ----------------------------------------------------------
# Flask Application
# ----------------------------------------------------------

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Recommendation System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f9;
            text-align: center;
            padding-top: 50px;
        }
        select, button {
            padding: 10px;
            margin: 10px;
        }
        .card {
            background: white;
            padding: 15px;
            margin: 10px auto;
            width: 300px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <h1>AI-Powered Product Recommendation</h1>

    <select id="productSelect">
        <option value="">Select a product</option>
        {% for product in products %}
            <option value="{{ product }}">{{ product }}</option>
        {% endfor %}
    </select>

    <button onclick="getRecommendations()">Get Recommendations</button>

    <div id="results"></div>

    <script>
        function getRecommendations() {
            const product = document.getElementById("productSelect").value;

            if (!product) {
                alert("Please select a product.");
                return;
            }

            fetch("/recommend", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ product: product })
            })
            .then(response => response.json())
            .then(data => {
                const resultsDiv = document.getElementById("results");
                resultsDiv.innerHTML = "<h3>Recommended Products:</h3>";

                data.forEach(item => {
                    resultsDiv.innerHTML += `
                        <div class="card">
                            <strong>${item.title}</strong><br>
                            Category: ${item.category}
                        </div>
                    `;
                });
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    products = data["title"].tolist()
    return render_template_string(HTML_TEMPLATE, products=products)

@app.route("/recommend", methods=["POST"])
def recommend():
    product_name = request.json["product"]
    recommendations = get_recommendations(product_name)
    return jsonify(recommendations)

# ----------------------------------------------------------
# Run Application
# ----------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
