import os
import threading
import requests
import pandas as pd
import time
import re
import string

from flask_cors import CORS
from flask import Flask, jsonify, request
from googleapiclient.discovery import build

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import fasttext
import nltk

# AraBERT Embedding
from transformers import AutoTokenizer, AutoModel
import torch
from multiprocessing import Pool

# FastText Embedding
from sklearn.decomposition import PCA
from huggingface_hub import hf_hub_download
import numpy as np

# Naive Bayes Modeling
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Define the port number
port_number = int(os.environ.get("PORT", 5000))

# Function to terminate any process using the port
def terminate_port(port):
    try:
        # Find the process using the port and terminate it
        result = os.popen(f"lsof -t -i:{port}").read().strip()
        if result:
            os.system(f"kill -9 {result}")
            print(f"Terminated process on port {port}.")
        else:
            print(f"No process found running on port {port}.")
    except Exception as e:
        print(f"Error while terminating port {port}: {e}")

# Terminate any process running on the specified port
terminate_port(port_number)

app = Flask(__name__)

# Enable CORS for all routes (you can also limit this to specific routes if needed)
CORS(app)

# Enable debug mode
app.debug = True

# Initialize YouTube API client
def get_youtube_client(api_key):
    return build("youtube", "v3", developerKey=api_key)

# Utility to split list into chunks
def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
# Search for videos
def search_videos(youtube, query, max_results):
    videos = []
    video_ids = []
    try:
        request = youtube.search().list(part="id,snippet", q=query, maxResults=min(max_results, 50), type="video")
        while request and len(videos) < max_results:
            response = request.execute()
            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                video_title = item["snippet"]["title"]
                video_ids.append(video_id)
                videos.append({"id": video_id, "title": video_title, "category_name": "Fetching..."})
                if len(videos) >= max_results:
                    break
            request = youtube.search().list_next(request, response)
    except Exception as e:
        print(f"Error searching videos: {e}")
    return videos, video_ids

# Fetch video details
def fetch_video_details(youtube, video_ids):
    video_details = []
    for chunk in split_list(video_ids, 50):
        response = youtube.videos().list(part="snippet", id=",".join(chunk)).execute()
        video_details.extend(response.get("items", []))
    return video_details

# Fetch comments
def fetch_comments(youtube, video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
        while request:
            response = request.execute()
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "comment_id": item["id"],
                    "text": snippet["textDisplay"],
                    "author": snippet["authorDisplayName"],
                    "published_at": snippet["publishedAt"],
                    "like_count": snippet["likeCount"],
                    "reply_count": item["snippet"]["totalReplyCount"]
                })
            request = youtube.commentThreads().list_next(request, response)
            time.sleep(1)  # Avoid hitting API rate limits
    except Exception as e:
        print(f"Error fetching comments for video {video_id}: {e}")
    return comments

# Function to clean the data as per the script
def clean_data(data):
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Function to check if a string contains only Arabic characters
    def non_arabic_without_punctuation(text):
        # Arabic Unicode range: \u0600-\u06FF
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)

        # Remove Arabic punctuation (commonly used in Arabic script)
        arabic_punctuation = r'[،؛؟\u060C\u061B\u066B\u066F]'  # Arabic comma, semicolon, question mark, etc.
        text = re.sub(arabic_punctuation, '', text)

        # Remove all other punctuation marks from any language
        text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)  # Remove punctuation

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text


    # Step 1: Remove non-Arabic characters and specific words
    word_to_remove = "و"  # Replace this with any word you want to remove
    df["Comment"] = df['Comment'].apply(non_arabic_without_punctuation)
    df['Comment'] = df['Comment'].str.replace(
        r'\b' + re.escape(word_to_remove) + r'\b', '', regex=True
    )

    # Step 2: Remove rows with empty 'Comment' or 'Label' columns or rows with only whitespace in 'Comment' or 'Label'
    df = df[
        (df['Comment'].notnull()) &
        (df['Label'].notnull()) &
        (df['Comment'].str.strip() != '') &
        (df['Label'].str.strip() != '')
    ]

    return df.to_dict(orient="records")  # Convert cleaned DataFrame back to dictionary

# Tokenization function
def tokenize_data(data):
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)

    # Tokenize the 'Comment' column by splitting each comment into words
    df['Comment'] = df['Comment'].str.split()

    return df.to_dict(orient="records")  # Convert tokenized DataFrame back to dictionary

# Load the model once at startup For AraBERT
MODEL_NAME_BERT = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)
model_bert = AutoModel.from_pretrained(MODEL_NAME_BERT)
model_bert.eval()  # Set to evaluation mode
def generate_embeddings_batch_bert(batch, dim):
    """Generate embeddings for a batch of comments using BERT."""
    embeddings = []
    for comment in batch:
        try:
            # Tokenize the comment and pad/truncate it to fit the model's max length
            inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                # Get the model's last hidden state and pool it to get sentence embedding
                output = model_bert(**inputs)
                embedding = output.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling
                embeddings.append(embedding[:dim])  # Trim to requested dimensions
        except Exception as e:
            embeddings.append([0] * dim)  # Append zeros if an error occurs
    return embeddings


# Load the FastText model at startup
model_path = hf_hub_download(repo_id="facebook/fasttext-ar-vectors", filename="model.bin")
model = fasttext.load_model(model_path)

# # Route to check if the app is working

# Home route
@app.route('/')
def home():
    return "Welcome to ScrapeSense API!"

# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    print("Health check endpoint called.")
    return jsonify({"status": "running"}), 200

# Scrape route
@app.route("/scrape", methods=["POST"])
def scrape_youtube():
    data = request.get_json()
    api_key = data.get("apiKey")

    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    # YouTube API request
    ss_url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}"

    try:
        ss_response = requests.get(ss_url).json()

        if 'error' in ss_response:
            return jsonify({"error": ss_response['error']}), 400

        return jsonify(ss_response), 200  # Return the data from YouTube API

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching data from YouTube API: {str(e)}"}), 500

# Search Videos Route
@app.route("/searchVideos", methods=["POST"])
def search_videos_route():
    try:
        # Parse request JSON
        data = request.get_json()
        query = data.get("query")
        max_results = data.get("max_results", 50)
        developer_key = data.get("developerKey")

        # Initialize YouTube API client
        youtube = get_youtube_client(developer_key)

        # Search videos
        videos, video_ids = search_videos(youtube, query, int(max_results))

        # Fetch comments
        all_comments = []
        for video_id in video_ids:
            all_comments.extend(fetch_comments(youtube, video_id))

        # Convert data to Pandas DataFrame and return as JSON
        df_comments = pd.DataFrame(all_comments)
        return jsonify({
            "videos": videos,
            "comments": df_comments.to_dict(orient="records")
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Clean Data Route
@app.route("/clean", methods=["POST"])
def clean():
    try:
        data = request.get_json()  # Get JSON payload

        if not data or not isinstance(data, list):
            return jsonify({"error": "Valid data array is required"}), 400

        # Clean the data
        cleaned_data = clean_data(data)

        return jsonify({"cleanedData": cleaned_data}), 200  # Return cleaned data

    except Exception as e:
        return jsonify({"error": f"Error cleaning data: {str(e)}"}), 500

# Tokenize Data Route
@app.route("/tokenize", methods=["POST"])
def tokenize():
    try:
        data = request.get_json()  # Get JSON payload

        if not data or not isinstance(data, list):
            return jsonify({"error": "Valid data array is required"}), 400

        # Tokenize the data
        tokenized_data = tokenize_data(data)

        return jsonify({"tokenizedData": tokenized_data}), 200  # Return tokenized data

    except Exception as e:
        return jsonify({"error": f"Error tokenizing data: {str(e)}"}), 500

# Embedding Word Route
# TF-IDF Route
@app.route('/embedding-tfidf', methods=['POST'])
def embedding_tfidf():
    try:
        # Get data from the request
        data = request.get_json()
        method = data.get("method")
        max_features = int(data.get("maxFeatures"))
        min_df = float(data.get("minDf")) / 100  # Convert percentage to a value between 0 and 1
        max_df = float(data.get("maxDf")) / 100  # Convert percentage to a value between 0 and 1
        tidy_features = data.get("tidyFeatures")

        # Prepare the text data for TF-IDF Vectorizer by joining tokenized words into a single string
        comments = [' '.join(item["Comment"]) for item in tidy_features]  # Join the list of tokens into a string for each Comment

        # Ensure the necessary NLTK resources are available
        nltk.download('stopwords')
        # Initialize the TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=nltk.corpus.stopwords.words('arabic')  # Using Arabic stop words as per your case
        )

        # Transform the comments to get the TF-IDF features
        features = vectorizer.fit_transform(comments)

        # Convert the features to a list of dictionaries
        features_list = features.toarray().tolist()

        # Prepare the response data
        response = {
            "features": features_list,
            "feature_names": vectorizer.get_feature_names_out().tolist(),
        }

        return jsonify(response)


    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Embedding AraBERT Route
@app.route('/embedding-bert', methods=['POST'])
def embedding_bert():
    try:
        # Extract comments and dim from the request
        request_data = request.get_json()
        comments = request_data.get('comments', [])
        dim = request_data.get('dim', 64)  # Default to 64 if dim is not provided

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Split comments into batches for better performance
        batch_size = 32  # Adjust the batch size as needed
        batches = [comments[i:i + batch_size] for i in range(0, len(comments), batch_size)]

        # Use multiprocessing to process batches in parallel
        with Pool(processes=4) as pool:
            embeddings = pool.starmap(generate_embeddings_batch_bert, [(batch, dim) for batch in batches])

        # Flatten the result (list of lists) into a single list
        embeddings = [embedding for batch in embeddings for embedding in batch]

        # Return the embeddings as JSON
        return jsonify({"embeddings": embeddings}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Embedding FastText Route
@app.route('/embedding-fasttext', methods=['POST'])
def embedding_fasttext():
    try:
        # Extract the request data
        request_data = request.get_json()
        comments = request_data.get('comments', [])
        batch_size = request_data.get('batch_size', 1000)  # Default batch size if not provided
        min_word_count = request_data.get('min_word_count', 0)  # Min word count filter
        embedding_method = request_data.get('embedding_method', 'average')  # Default to averaging word embeddings

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        embeddings = []

        for comment in comments:
            words = comment.split()
            word_vectors = [model[word] for word in words if word in model]

            # Optionally filter words by min_word_count (if it's a valid model parameter)
            if min_word_count > 0:
                word_vectors = [wv for word, wv in zip(words, word_vectors) if words.count(word) >= min_word_count]

            if word_vectors:
                if embedding_method == 'average':
                    # Mean of word embeddings
                    comment_embedding = sum(word_vectors) / len(word_vectors)

                elif embedding_method == 'weighted_average':
                    # Weighted average of word embeddings
                    weights = [1 / (i + 1) for i in range(len(word_vectors))]  # Example weighting scheme
                    weighted_sum = sum(w * vec for w, vec in zip(weights, word_vectors))
                    comment_embedding = weighted_sum / sum(weights)

                elif embedding_method == 'max':
                    # Max pooling of word embeddings
                    comment_embedding = np.max(word_vectors, axis=0)

                else:
                    # Raise an error for unsupported methods
                    raise ValueError(f"Unsupported embedding method: {embedding_method}")

                embeddings.append(comment_embedding.tolist())
            else:
                # Handle comments with no valid words in the model
                embeddings.append([])

        return jsonify({"embeddings": embeddings}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Modeling Tf-IDf Route
@app.route('/train-tfidf', methods=['POST'])
def train_tfidf_model():
    try:
        # Parse the request data
        request_data = request.get_json()
        embedded_data = request_data['embeddedData']
        cleaned_data = request_data['cleanedData']
        partition = request_data['partition']  # e.g., "80-20" or "70-30"

        # Determine test size from partition
        partition_mapping = {
            "80-20": 0.2,
            "70-30": 0.3
        }
        test_size = partition_mapping.get(partition)
        if test_size is None:
            return jsonify({"error": "Invalid partition value"}), 400

        # Convert data to DataFrame
        df = pd.DataFrame(cleaned_data)  # cleanedData should contain 'Comment' and 'Label'
        X = pd.DataFrame(embedded_data['features'])  # embeddedData['features'] should be an array of arrays
        y = df['Label']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        #  Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_labels = {
            "labels": model.classes_.tolist(),
            "matrix": conf_matrix.tolist()  # Convert to list for JSON serialization
        }

        # Return results
        return jsonify({
             "X": X.to_json(orient="split"),
            "Y": y.to_json(orient="split"),
            "accuracy": accuracy,
            "classificationReport": report,
            "confusionMatrix": conf_matrix_labels
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port_number)