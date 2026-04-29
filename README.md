# Basic End-to-End Demo Pipeline for Retraining Model and Running Final Prediction Model
## - By Michael Acheampong

This repository is designed as a **teaching project for students learning end-to-end machine learning (MLOps)**. It walks through the full lifecycle of a model:

* Retraining an existing model with new data
* Archiving old model versions
* Running predictions locally
* Serving predictions via an API
* Containerizing the application with Docker
* Automating retraining with CI

The project uses the classic Iris dataset to keep concepts simple while demonstrating real-world workflows.

---

## 📁 Repository Structure

```
.
├── app.py                  # Flask API for serving predictions
├── final_iris_model.keras  # Current production model
├── final_iris_model.py     # CLI prediction script
├── iris_scaler.pkl         # Feature scaler
├── retrain_iris_model.py   # Retraining + model versioning
├── Dockerfile              # Container setup
├── requirements.txt        # Dependencies
├── .github/workflows/ci.yml# CI pipeline
└── README.md
```

---

## 🎯 Learning Objectives

By working through this repo, students will learn how to:

* Retrain a machine learning model using new data
* Version and archive models
* Run predictions from a trained model
* Deploy a model behind an API
* Containerize ML applications using Docker
* Automate workflows using CI/CD

---

## 🔁 Part 1: Retrain the Model (with Versioning)

This step simulates updating a production model with new data.

### What happens:

* The current model (`final_iris_model.keras`) is **archived with a timestamp**
* The model is **retrained on the dataset**
* A **new updated model** is saved with the original name

### Run:

```bash
python retrain_iris_model.py
```

### Example output:

```
Archived old model as: final_iris_model_20260429.keras
```

📌 This ensures **model history is preserved**, which is critical in real-world ML systems.

---

## 🔮 Part 2: Run Predictions Locally

You can test the trained model directly from the command line.

### Input format:

```json
{
  "sepal_length": 10,
  "sepal_width": 7,
  "petal_length": 8,
  "petal_width": 10
}
```

### Run with inline JSON:

```bash
python final_iris_model.py --input_json '{"sepal_length":10,"sepal_width":7,"petal_length":8,"petal_width":10}'
```

### Or using a file:

```bash
python final_iris_model.py --input_file sample.json
```

### Output:

```
Prediction: virginica
```

---

## 🌐 Part 3: Serve the Model with Flask API

The model is exposed via a simple Flask application.

### Start the server:

```bash
python app.py
```

Server runs on:

```
http://127.0.0.1:5001
```

### Test the API:

```bash
curl -X POST "http://127.0.0.1:5001/api/flower" \
-H "Content-Type: application/json" \
-d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Response:

```json
"setosa"
```

---

## 🐳 Part 4: Dockerize the Application

A simple multi-stage Dockerfile is included to package the Flask API.

### Build the image:

```bash
docker build -t iris-ml-app .
```

### Run the container:

```bash
docker run -p 5001:5001 iris-ml-app
```

Now access the API at:

```
http://localhost:5001/api/flower
```

---

## ⚙️ Part 5: CI/CD Pipeline

The repository includes a GitHub Actions workflow:

```
.github/workflows/ci.yml
```

### What it does:

* Installs dependencies
* Retrains the model
* Uploads all artifacts (model, scaler, etc.)

### Triggered on:

* Push to `main`
* Pull requests to `main`

This demonstrates how **ML training can be automated in CI pipelines**.

---

## 🚀 Quick Start for Students

Follow these steps to go through the full pipeline:

### 1. Setup environment

```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
# .venv\Scripts\activate    # (Windows)

pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Retrain the model

```bash
python retrain_iris_model.py
```

---

### 3. Run a local prediction

```bash
python final_iris_model.py --input_json '{"sepal_length":6,"sepal_width":3,"petal_length":5,"petal_width":2}'
```

---

### 4. Start the API

```bash
python app.py
```

---

### 5. Test the API

```bash
curl -X POST "http://127.0.0.1:5001/api/flower" \
-H "Content-Type: application/json" \
-d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

---

### 6. Run with Docker

```bash
docker build -t iris-ml-app .
docker run -p 5001:5001 iris-ml-app
```

---

## 🧠 Key Concepts Demonstrated

* Model retraining and lifecycle management
* Model versioning (archiving old models)
* Feature preprocessing with scalers
* CLI-based inference
* REST API deployment with Flask
* Containerization with Docker
* CI/CD for ML workflows

---

## 📌 Notes for Students

* Always version your models before retraining
* Keep preprocessing (like scalers) consistent
* Test locally before deploying
* Automate repetitive tasks (CI/CD)
* Think of ML as a **pipeline**, not just a notebook

