# Customer Purchase Behaviour — Spending Prediction App

Predict customer spending based on behavioural and financial attributes using a
Linear Regression model trained in Python and deployed with **Gradio** and **Hugging Face Spaces**.

**Live Demo:** https://huggingface.co/spaces/yashuchouhan/Customerpurchasebehaviour

## Objective

Businesses need insights into how much customers are likely to spend so they can:

- target the right customers
- personalize offers
- optimize marketing budgets
- improve customer retention

This project builds a machine-learning model that predicts **purchase amount**
using customer behaviour features.

##  Project Overview

- Data preprocessing and cleaning
- Feature selection
- Linear Regression model
- Model evaluation (R², RMSE)
- Saved model using `joblib`
- Interactive UI using **Gradio**
- Deployment to **Hugging Face Spaces**

#  Model Details
**Algorithm:** Linear Regression 
- simple and interpretable
- shows how each feature influences spending
- works well for numeric relationships

##  Features Used

- **Purchase Frequency** – number of purchases made within a time period  
- **Loyalty Score** – measure of customer engagement and loyalty  
- **Annual Income** – customer's yearly income  

## Tech Stack

Python

Pandas

Scikit-learn

Joblib

Gradio

Hugging Face Spaces

# How to Use
🔹 Single Customer Prediction

Enter:

Purchase Frequency

Loyalty Score

Annual Income

Click Predict Purchase Amount

View the predicted spending value

##  Batch Prediction (CSV Upload)

You can upload a CSV file to generate predictions for multiple customers at once.

### Input CSV format

| purchase_frequency | loyalty_score | annual_income |
|-------------------|--------------|--------------|
| 5 | 70 | 500000 |
| 2 | 35 | 120000 |
| 10 | 90 | 650000 |

 After uploading, click **Run Batch Prediction**  
 The app will display predicted purchase amounts  
 Click **Download Predictions** to save results as CSV

