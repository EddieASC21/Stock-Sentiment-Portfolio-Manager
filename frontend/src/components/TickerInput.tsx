// src/components/TickerInput.tsx

import React, { useState, KeyboardEvent } from "react";
import { fetchSentiment } from "../api/sentimentApi";
import ReactSpeedometer from "react-d3-speedometer";

interface SentimentResponse {
  sentiment_summary: Record<string, string>;
  opinion_score: string;
  model_accuracy?: string;
  error?: string;
}

function TickerInput() {
  const [inputValue, setInputValue] = useState<string>("");
  const [displayValue, setDisplayValue] = useState<string>("");
  const [sentimentScore, setSentimentScore] = useState<number>(0.5); // 0–1 range, 0.5 = neutral
  const [sentimentLabel, setSentimentLabel] = useState<string>("Neutral");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
    setError("");
  };

  const handleKeyPress = async (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      setError("");
      setDisplayValue("");
      setLoading(true);

      try {
        // 1) Call the Flask endpoint (via our TS helper)
        const result: SentimentResponse = await fetchSentiment(
          inputValue.toUpperCase(),
          10
        );

        // 2) If Flask returned an error, show it
        if (result.error) {
          setError(result.error);
          setLoading(false);
          return;
        }

        // 3) Extract the numeric part of opinion_score (e.g. "0.50 (...)" → 0.5)
        //    We split on space, parseFloat on the first token
        const rawScore = result.opinion_score.split(" ")[0];
        const parsed = parseFloat(rawScore); // should be in [0,1]

        // 4) Determine sentimentLabel based on parsed score
        let label = "Neutral";
        if (parsed < 0.3) {
          label = "Bearish";
        } else if (parsed >= 0.30 && parsed < 0.45) {
          label = "Somewhat Bearish";
        } else if (parsed >= 0.45 && parsed < 0.55) {
          label = "Neutral";
        } else if (parsed >= 0.55 && parsed < 0.70) {
          label = "Somewhat Bullish";
        } else {
          label = "Bullish";
        }

        // 5) Update state
        setSentimentScore(parsed);
        setSentimentLabel(label);

        // 6) Build a display string. Feel free to format however you like:
        const summaryLines = Object.entries(result.sentiment_summary)
          .map(([sent, desc]) => `${sent}: ${desc}`)
          .join(" | ");

        setDisplayValue(() => {
          return (
            `Company: ${inputValue.toUpperCase()}  ` +
            `Opinion Score: ${parsed.toFixed(3)}  ` +
            `(Label: ${label})\n\n` +
            `Details → ${summaryLines}`
          );
        });
      } catch (err: any) {
        console.error(err);
        setError(
          err.response?.data?.error ||
            "An unexpected error occurred while fetching sentiment."
        );
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      {/* 1) Speedometer / Gauge */}
      <div className="mb-2">
        <ReactSpeedometer
          value={sentimentScore * 100} // convert [0,1] → [0,100]
          maxValue={100}
          minValue={0}
          segments={5}
          needleColor="gray"
          segmentColors={[
            "#5BE12C", // very green
            "#AED51F", // yellow‐green
            "#FFC914", // yellow
            "#FF6A15", // orange
            "#EA4228", // red
          ]}
          valueTextFontSize="0px" // hide the numeric text inside gauge
          width={300}
          height={200}
        />
      </div>

      {/* 2) Input Box */}
      <div className="form-control w-full max-w-xs mb-6">
        <label className="label">
          <span className="label-text">Sentiment Analyzer</span>
          <span className="label-text-alt">Input Ticker</span>
        </label>
        <input
          type="text"
          placeholder="Type ticker, then press Enter"
          className="input input-bordered w-full max-w-xs"
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          value={inputValue}
        />
        <label className="label">
          <span className="label-text-alt">
            Any company listed on a public stock exchange
          </span>
        </label>
      </div>

      {/* 3) Loading / Error / Results */}
      {loading && <p className="mb-4">Loading sentiment…</p>}
      {error && <p className="text-red-600 mb-4">{error}</p>}

      {displayValue && (
        <pre className="whitespace-pre-wrap bg-gray-100 p-4 rounded mb-4">
          {displayValue}
        </pre>
      )}

      {/* 4) Legend */}
      <div className="sentiment-legend flex flex-wrap gap-4 p-4 border border-gray-300 rounded-lg">
        <div>
          <strong>Bearish:</strong> [0.00 – 0.30)
        </div>
        <div>
          <strong>Somewhat Bearish:</strong> [0.30 – 0.45)
        </div>
        <div>
          <strong>Neutral:</strong> [0.45 – 0.55)
        </div>
        <div>
          <strong>Somewhat Bullish:</strong> [0.55 – 0.70)
        </div>
        <div>
          <strong>Bullish:</strong> [0.70 – 1.00]
        </div>
      </div>
    </div>
  );
}

export default TickerInput;
