# backend/sentiment_service.py

from pyspark.sql import SparkSession
import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StringType, IntegerType
import requests
import time
import random

# ─────────────────────────────────────────────────────────────────
# 1) Remove import‐time Spark creation:
#    We will initialize SparkSession and the NLP pipeline lazily.
# ─────────────────────────────────────────────────────────────────
spark = None
nlp_model = None
hashingTF = None
idf = None
convert_to_string_array = udf(lambda x: x, ArrayType(StringType()))
random_label_udf  = udf(lambda: random.randint(0, 1), IntegerType())

# ─────────────────────────────────────────────────────────────────
# 2) A helper function to lazily initialize Spark and all pipeline stages
# ─────────────────────────────────────────────────────────────────
def _init_spark_and_pipeline():
    global spark, nlp_model, hashingTF, idf

    if spark is None:
        # Start Spark only once, on the driver
        spark = sparknlp.start()

        # Create a tiny “dummy” DataFrame so we can fit the NLP pipeline
        dummy_data = [("DUMMY", "This is placeholder text for pipeline fitting.")]
        df_dummy = spark.createDataFrame(dummy_data, schema=["ticker", "text"])

        # Build Spark NLP pipeline
        document_assembler = (
            DocumentAssembler().setInputCol("text").setOutputCol("document")
        )
        tokenizer = (
            Tokenizer().setInputCols(["document"]).setOutputCol("token")
        )
        normalizer = (
            Normalizer().setInputCols(["token"]).setOutputCol("normalized")
        )
        stop_words_cleaner = (
            StopWordsCleaner().setInputCols(["normalized"]).setOutputCol("cleanTokens")
        )
        lemmatizer = (
            LemmatizerModel
            .pretrained("lemma_antbnc", lang="en")
            .setInputCols(["cleanTokens"])
            .setOutputCol("lemmas")
        )

        pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            normalizer,
            stop_words_cleaner,
            lemmatizer
        ])

        # Fit the NLP pipeline on dummy data
        nlp_model = pipeline.fit(df_dummy)

        # Initialize TF‐IDF objects (HashingTF doesn’t need a “fit”)
        hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures")
        idf      = IDF(inputCol="rawFeatures", outputCol="features")


# ─────────────────────────────────────────────────────────────────
# 3) The core function that fetches articles, processes them, and returns sentiment
# ─────────────────────────────────────────────────────────────────
def get_sentiment_for_ticker(ticker, n=10):
    from pyspark.ml.classification import NaiveBayes  # import where spark exists

    # 3a) Make sure Spark and the NLP pipeline are initialized
    _init_spark_and_pipeline()

    # 3b) Fetch the latest “n” articles for this ticker
    news_data = fetch_news_from_ticker_tick([ticker], n=n)
    if not news_data:
        return {"error": f"No data available for ticker {ticker}"}

    # 3c) Create a DataFrame with schema ["ticker", "text"]
    df = spark.createDataFrame(news_data, schema=["ticker", "text"])

    # 3d) Run the Spark NLP pipeline on the “text” column
    processed = nlp_model.transform(df)  # adds “lemmas” column

    # 3e) Extract tokens (lemmatized words) into a String array column
    processed = processed.withColumn("tokens", convert_to_string_array(col("lemmas.result")))

    # 3f) Apply HashingTF → rawFeatures
    featured = hashingTF.transform(processed)

    # 3g) Fit an IDF model on these features (rescale)
    idfModel = idf.fit(featured)
    rescaled = idfModel.transform(featured)

    # 3h) Assign random labels (for demo) and train NaiveBayes
    rescaled    = rescaled.withColumn("label", random_label_udf())
    train, test = rescaled.randomSplit([0.8, 0.2], seed=12345)
    nb          = NaiveBayes(featuresCol="features", labelCol="label")
    nbModel     = nb.fit(train)

    # 3i) Evaluate on the test set (optional)
    predictions = nbModel.transform(test)
    evaluator   = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    nbAccuracy = evaluator.evaluate(predictions)

    # 3j) Predict on all data, classify each article’s sentiment
    final_preds = nbModel.transform(rescaled) \
                         .withColumn("sentiment", classify_sentiment_udf(col("probability")))

    # 3k) Aggregate counts by sentiment
    sentiment_counts = final_preds.groupBy("sentiment").count().collect()
    total_articles   = sum(row["count"] for row in sentiment_counts)
    positive_articles = next(
        (row["count"] for row in sentiment_counts if "Positive" in row["sentiment"]),
        0
    )

    sentiment_summary = {
        row["sentiment"]: f"{row['count']} articles, {row['sentiment'].split(' - ')[1]}"
        for row in sentiment_counts
    }

    opinion_score = (
        positive_articles / total_articles
        if total_articles > 0 else 0
    )

    return {
        "sentiment_summary": sentiment_summary,
        "opinion_score": f"{opinion_score:.2f} (Positive: {positive_articles} / {total_articles})",
        "model_accuracy": f"{nbAccuracy:.2f}"
    }


# ─────────────────────────────────────────────────────────────────
# 4) The helper to fetch news from TickerTick (unchanged)
# ─────────────────────────────────────────────────────────────────
def fetch_news_from_ticker_tick(tickers, n=10):
    api_url = "https://api.tickertick.com/feed"
    news_data = []
    for t in tickers:
        params = {"q": f"tt:{t}", "n": n}
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            for story in data.get("stories", []):
                news_data.append((t, story["title"]))
        else:
            print(f"Failed to fetch data for {t}, status {response.status_code}")
        time.sleep(1)
    return news_data

# ─────────────────────────────────────────────────────────────────
# 5) The UDF/classify function remains at module scope
# ─────────────────────────────────────────────────────────────────
def classify_sentiment(probabilities):
    positive_prob = probabilities[1]
    if positive_prob > 0.6:
        return "Positive - News is likely viewed favorably."
    elif positive_prob < 0.4:
        return "Negative - News is likely viewed unfavorably."
    else:
        return "Neutral - News is seen as neither clearly good nor bad."

classify_sentiment_udf = udf(classify_sentiment, StringType())
