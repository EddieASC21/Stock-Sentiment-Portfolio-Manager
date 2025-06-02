# backend/app.py

from flask import Flask, request, jsonify
from sentiment_service import get_sentiment_for_ticker
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Flask is running. Use /api/sentiment?ticker=…"

@app.route("/api/sentiment", methods=["GET"])
def sentiment_endpoint():
    ticker = request.args.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "You must supply a `ticker` query parameter."}), 400

    n = request.args.get("n", default=10, type=int)
    try:
        result = get_sentiment_for_ticker(ticker, n=n)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Disable Flask reloader so we don’t double‐start Spark
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
