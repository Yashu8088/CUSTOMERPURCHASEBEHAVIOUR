import gradio as gr
import joblib
import pandas as pd

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("linear_regression_model.pkl")

FEATURES = ['purchase_frequency', 'loyalty_score', 'annual_income']

# -------------------------------
# Single prediction
# -------------------------------
def predict_single(purchase_frequency, loyalty_score, annual_income):
    try:
        X = pd.DataFrame([{
            "purchase_frequency": float(purchase_frequency),
            "loyalty_score": float(loyalty_score),
            "annual_income": float(annual_income)
        }])

        prediction = model.predict(X)[0]

        return {
            "Predicted_Purchase_Amount": round(float(prediction), 2)
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Batch prediction
# -------------------------------
def predict_batch(file):
    if file is None:
        return pd.DataFrame({"error": ["Please upload a CSV file"]})

    df = pd.read_csv(file.name)

    if not all(col in df.columns for col in FEATURES):
        return pd.DataFrame({
            "error": ["CSV must contain: purchase_frequency, loyalty_score, annual_income"]
        })

    X = df[FEATURES].astype(float)
    df["Predicted_Purchase_Amount"] = model.predict(X).round(2)

    return df

def predict_batch_and_save(file):
    df = predict_batch(file)
    path = "batch_predictions.csv"
    df.to_csv(path, index=False)
    return path

# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 💳 Customer Purchase Behavior Prediction")
    gr.Markdown(
        "Predict customer purchase amount using a Linear Regression model "
        "trained on purchase frequency, loyalty score, and annual income."
    )

    with gr.Tab("Single Customer Prediction"):
        purchase_frequency = gr.Number(label="Purchase Frequency", value=5)
        loyalty_score = gr.Number(label="Loyalty Score", value=70)
        annual_income = gr.Number(label="Annual Income", value=500000)

        btn = gr.Button("Predict Purchase Amount")
        output = gr.JSON()

        btn.click(
            fn=predict_single,
            inputs=[purchase_frequency, loyalty_score, annual_income],
            outputs=output
        )

    with gr.Tab("Batch Prediction (CSV Upload)"):
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        btn_batch = gr.Button("Run Batch Prediction")

        out_df = gr.Dataframe(interactive=False)
        download_btn = gr.DownloadButton(label="Download Predictions")

        btn_batch.click(
            fn=predict_batch,
            inputs=file_input,
            outputs=out_df
        )

        download_btn.click(
            fn=predict_batch_and_save,
            inputs=file_input,
            outputs=download_btn
        )

if __name__ == "__main__":
    demo.launch()
