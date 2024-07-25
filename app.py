from flask import Flask, render_template, request
import pandas as pd
from src.recommendation import recommend_content_based, recommend_collaborative

app = Flask(__name__)

# Load dataset globally
data = pd.read_csv('data/dataset.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Use the globally defined data variable
    recommendations = recommend_content_based(data)  # or recommend_collaborative(data)
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
