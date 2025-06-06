# Stock Sentiment Portfolio Manager

## Introduction
This tool is built using the React (TypeScript) stack and Yarn package manager for the frontend, and a Python/Flask + PySpark backend, to help you analyze stock sentiment using various data sources.


## Getting Started
To get started with this project, follow these steps:

1. **Clone the Repository:** Start by cloning this repository to your local machine using Git.
```
git clone <repository-url>
```
2. Navigate to the Project Directory: Change your current directory to the project folder.
```
cd CFTstocksentiment
cd frontend
```
3. Backend Setup
```
cd backend
cd frontend
```
4. Install Dependencies: Install the project dependencies using Yarn.
```
yarn install
pip install flask flask-cors pyspark sparknlp requests
```
5. Run the Development Server: Start the development server to run the project locally.
```
yarn start
python app.py
```
6. Access the Application: Open your web browser and navigate to http://localhost:3000 to access the application.

## Project Structure
The project is structured as follows:

- `frontend/`: This directory contains the frontend code for the application.
- `backend/`: This directory contains the backend code for the application.
- `package.json`: This file contains project metadata and a list of dependencies.
- `yarn.lock`: A lock file that specifies exact versions of dependencies.

## Features

### 1. Real-time Sentiment Analysis
- Analyze sentiment from various sources in real-time to provide up-to-date insights.

### 2. Customizable Dashboard
- Customize the dashboard to display sentiment trends, sentiment distribution, and relevant news articles.

### 3. Historical Analysis
- View historical sentiment data to identify trends and patterns over time.
