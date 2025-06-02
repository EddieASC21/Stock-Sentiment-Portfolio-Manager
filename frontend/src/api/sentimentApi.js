import axios from "axios";

// If you later deploy Flask elsewhere, set REACT_APP_API_URL in .env
const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export async function fetchSentiment(ticker, n = 10) {
  const response = await axios.get(`${BASE_URL}/api/sentiment`, {
    params: { ticker, n },
  });
  return response.data;
}
