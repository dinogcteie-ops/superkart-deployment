import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

model = joblib.load("superkart_model.joblib")

FEATURE_COLS = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category",
]


@app.get("/")
def home():
    return jsonify({"status": "SuperKart sales prediction API is running"})


@app.post("/v1/predict")
def predict():
    """Online inference: predict sales for a single product/store combination."""
    payload = request.get_json()
    row = pd.DataFrame([payload])[FEATURE_COLS]
    prediction = model.predict(row)[0]
    return jsonify({"predicted_sales": round(float(prediction), 2)})


@app.post("/v1/predictbatch")
def predict_batch():
    """Batch inference: predict sales for every row in an uploaded CSV file."""
    file = request.files["file"]
    batch_df = pd.read_csv(file)
    predictions = model.predict(batch_df[FEATURE_COLS])
    result = {str(i): round(float(p), 2) for i, p in enumerate(predictions)}
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
