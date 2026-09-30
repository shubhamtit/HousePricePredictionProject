# 🏠 House Price Prediction — Linear Regression + FastAPI + Docker

A machine learning project built while learning **Linear Regression** and the basics of **ML deployment/MLOps**.

The project predicts house prices based on:

* Area in square feet
* Number of bedrooms
* Number of bathrooms
* Age of the house

The trained machine learning model is exposed through a **FastAPI REST API** and packaged into a **Docker image** so that it can be easily run on another machine.

---

## 🚀 Project Overview

The project follows this workflow:

```text
Dataset
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Linear Regression
   ↓
Model Evaluation
   ↓
Save Trained Model
   ↓
FastAPI
   ↓
Docker
   ↓
Docker Hub
   ↓
Run API Locally
```

---

## 🧠 What I Learned

Through this project, I practiced the complete basic machine learning workflow:

* Loading and understanding a dataset
* Exploratory Data Analysis (EDA)
* Understanding features and target variables
* Correlation analysis
* Splitting data into training and testing sets
* Training a Linear Regression model
* Understanding model coefficients and intercept
* Making predictions
* Evaluating a regression model
* Saving a trained ML model with Joblib
* Creating a REST API using FastAPI
* Containerizing the ML application with Docker
* Publishing the Docker image to Docker Hub

---

# 📊 Dataset

The dataset contains the following features:

| Feature     | Description                      |
| ----------- | -------------------------------- |
| `area_sqft` | Area of the house in square feet |
| `bedrooms`  | Number of bedrooms               |
| `bathrooms` | Number of bathrooms              |
| `age_years` | Age of the house                 |
| `price`     | House price — target variable    |

The dataset was generated for learning purposes.

---

# 🔍 Exploratory Data Analysis

Before training the model, I performed EDA to understand the dataset.

Some of the things checked were:

### Dataset shape

```python
data.shape
```

This tells us the number of rows and columns.

### Dataset information

```python
data.info()
```

This helps understand:

* Data types
* Number of entries
* Missing values

### Statistical summary

```python
data.describe()
```

This provides:

* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

### Missing values

```python
data.isnull().sum()
```

This checks whether any columns contain missing values.

### Correlation

```python
data.corr()
```

Correlation helped understand how strongly the different features were related to the target price.

---

# 🤖 Linear Regression

The model uses multiple linear regression.

The basic equation is:

```text
price =
    b0
    + b1 × area_sqft
    + b2 × bedrooms
    + b3 × bathrooms
    + b4 × age_years
```

Where:

* `b0` = intercept
* `b1...b4` = learned coefficients

The model learns these values from the training data.

---

# ✂️ Train/Test Split

The dataset was divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The model uses the training data to learn.

The test data is kept separate and used later to evaluate how well the model performs on unseen data.

---

# 🏋️ Training the Model

The model was created using scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

After training, the model can make predictions:

```python
predictions = model.predict(X_test)
```

---

# 📏 Model Evaluation

Several regression metrics were used to understand model performance.

## MAE — Mean Absolute Error

```text
MAE = average(|actual - predicted|)
```

MAE tells us the average absolute difference between the actual and predicted values.

It is expressed in the same unit as the target.

For example, if MAE is ₹4 lakh, the model's predictions differ from the actual prices by about ₹4 lakh on average in absolute terms.

---

## MSE — Mean Squared Error

```text
MSE = average((actual - predicted)²)
```

MSE squares the prediction errors.

Because the errors are squared, larger errors receive more penalty.

The drawback is that MSE is expressed in squared target units, which makes it less intuitive to interpret directly.

---

## RMSE — Root Mean Squared Error

```text
RMSE = √MSE
```

RMSE converts the error back to the same unit as the target.

It also gives more weight to larger errors than MAE.

---

## R² — R-squared

R² measures how much of the variation in the target variable is explained by the model on the evaluation data.

A value closer to `1` indicates that the model explains more of the variation in the evaluation set.

R² should not be interpreted simply as "model accuracy percentage."

---

# 📈 Actual vs Predicted

I also compared the actual house prices with the model's predicted prices.

```text
Actual Price
      │
      │       •
      │    •
      │  •
      │•
      └──────────────── Predicted Price
```

A prediction equal to the actual value would lie on the ideal diagonal line.

This visualization helps identify how far predictions are from the actual values.

---

# 💾 Saving the Model

After training, the model was saved using Joblib:

```python
import joblib

joblib.dump(model, "house_price_model.pkl")
```

This allows the trained model to be loaded later without retraining it.

---

# 🌐 FastAPI

The trained model was exposed through a REST API using FastAPI.

Example request:

```json
{
    "area_sqft": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 5
}
```

The API sends the input to the trained model and returns the predicted price.

Example endpoint:

```text
POST /prediction
```

FastAPI also provides interactive API documentation through Swagger UI:

```text
http://localhost:8000/docs
```

---

# 🐳 Docker

The FastAPI application and trained model were packaged into a Docker image.

The Docker image contains the required application environment, including:

* Python
* FastAPI
* Uvicorn
* scikit-learn
* Joblib
* NumPy
* FastAPI application
* Trained ML model

This makes the application easier to run consistently on another machine.

---

# 📦 Docker Image

The Docker image is published on Docker Hub.

**Docker Hub:**

https://hub.docker.com/r/shubhamsinghtit/housepriseprediction

To download the image:

```bash
docker pull YOUR_DOCKER_USERNAME/house-price-api:latest
```

Run the container:

```bash
docker run -p 8000:8000 YOUR_DOCKER_USERNAME/house-price-api:latest
```

Then open:

```text
http://localhost:8000/docs
```

---

# 💻 Running the Project Locally

## Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd Project1
```

## Create a virtual environment

```bash
python3 -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run FastAPI

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

# 🗂️ Project Structure

```text
Project1/
│
├── main.py
├── house_price_model.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib
* Docker
* Docker Hub

---

# 🎯 What This Project Represents

This project was created as a learning project to understand not only how to train a machine learning model, but also how to take that model beyond a notebook/script and expose it as a usable service.

The main progression was:

```text
Machine Learning
       ↓
Model Evaluation
       ↓
Model Persistence
       ↓
REST API
       ↓
Containerization
       ↓
Docker Image
       ↓
Docker Hub
```

This is my first step toward learning **MLOps and ML deployment**.

---

## 🔗 Links

### GitHub

https://github.com/shubhamtit/HousePricePredictionProject

### Docker Hub

https://hub.docker.com/r/shubhamsinghtit/housepriseprediction
