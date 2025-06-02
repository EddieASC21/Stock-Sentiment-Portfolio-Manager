// src/api/sentimentApi.ts
import axios from "axios";

export interface SentimentResponse {
  sentiment_summary: Record<string, string>;
  opinion_score: string;
  model_accuracy?: string;
  error?: string;
}

// Base URL for your Flask backend
const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export async function fetchSentiment(
  ticker: string,
  n: number = 10
): Promise<SentimentResponse> {
  const response = await axios.get<SentimentResponse>(
    `${BASE_URL}/api/sentiment`,
    {
      params: { ticker, n },
    }
  );
  return response.data;
}
