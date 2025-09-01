# Stroke Risk Prediction API

A simple Flask backend wrapped in Docker for predicting stroke risk.

## How to Run (Locally)
```bash
docker build -t stroke-api .
docker run -p 7860:7860 stroke-api
