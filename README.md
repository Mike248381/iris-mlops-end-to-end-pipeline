# End-to-End MLOps Pipeline: Retraining, Versioning, and Model Serving (Iris Dataset)

### Michael Acheampong, PhD

This repository is a **teaching-focused, end-to-end MLOps project** designed to help students understand how machine learning systems move from training to deployment.

It demonstrates a complete workflow using a TensorFlow model, including:

* Retraining an existing model with new data
* Archiving and versioning previous model artifacts
* Running predictions via a command-line interface
* Serving predictions through a Flask API
* Containerizing the application with Docker
* Automating retraining with CI/CD pipelines

While the dataset (Iris) is simple, the **engineering workflow reflects real-world ML systems**.

---

## 📁 Repository Structure

```
.
├── app.py                  # Flask API for serving predictions
├── final_iris_model.keras  # Current TensorFlow production model
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

By working through this repository, students will learn how to:

* Retrain and update an existing machine learning model
* Implement basic model versioning and artifact management
* Perform inference via CLI and API interfaces
* Deploy a model using Flask
* Containerize applications with Docker
* Automate ML workflows using CI/CD

---

## 🔁 Part 1: Retrain the Model (with Versioning)

This step simulates updating a production model with new data.

### What happens:

* The current model (`final_iris_model.keras`) is **archived with a timestamp**
* The archived model is **reloaded and retrained**
* A **new production model** is saved using the original filename

### Run:

```bash
python retrain_iris_model.py
```

### Example output:

```
Archived old model as: final_iris_model_20260429.keras
```

📌 This demonstrates a simple but important concept: **never overwrite models without keeping history**.

---

## 🔮 Part 2: Run Predictions Locally

Run inference directly from the command line using the trained model.

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

## 🌐 Part 3: Serve the Model with a Flask API

The trained model is exposed through a lightweight REST API.

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

A multi-stage Dockerfile is included to package the Flask API for deployment.

### Build the image:

```bash
docker build -t iris-ml-app .
```

### Run the container:

```bash
docker run -p 5001:5001 iris-ml-app
```

Access the API at:

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
* Uploads model artifacts

### Triggered on:

* Push to `main`
* Pull requests to `main`

This demonstrates how **model retraining can be automated in a CI pipeline**.

---

## 🚀 Quick Start for Students

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

### 3. Run a prediction

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
* Model versioning via artifact archiving
* Feature preprocessing consistency
* CLI-based inference workflows
* REST API deployment with Flask
* Containerization using Docker
* CI/CD for machine learning pipelines

---

## 📌 Notes for Students

* Always version models before retraining
* Keep preprocessing steps consistent between training and inference
* Validate locally before deploying
* Automate repeatable workflows
* Treat machine learning as a **system**, not just a model