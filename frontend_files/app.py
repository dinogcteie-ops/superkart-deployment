import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Predictor", layout="centered")

backend_url = st.sidebar.text_input(
    "Backend URL", value=os.environ.get("BACKEND_URL", "http://localhost:7860")
)

st.title("SuperKart Sales Forecasting")
st.write("Predict `Product_Store_Sales_Total` using the deployed SuperKart model.")

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

with tab_single:
    st.subheader("Online Inference")
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
        product_sugar_content = st.selectbox(
            "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
        )
        product_allocated_area = st.number_input(
            "Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f"
        )
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
        product_id_char = st.selectbox("Product Id Char", ["FD", "DR", "NC"])
    with col2:
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
        store_location_city_type = st.selectbox(
            "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
        )
        store_type = st.selectbox(
            "Store Type",
            ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
        )
        store_age_years = st.number_input("Store Age (Years)", min_value=0, value=16)
        product_type_category = st.selectbox(
            "Product Type Category", ["Perishables", "Non Perishables"]
        )

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age_years,
            "Product_Type_Category": product_type_category,
        }
        try:
            response = requests.post(f"{backend_url}/v1/predict", json=payload, timeout=30)
            response.raise_for_status()
            st.success(f"Predicted Sales: {response.json()['predicted_sales']}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")

with tab_batch:
    st.subheader("Batch Inference")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        st.write("Preview:")
        preview_df = pd.read_csv(uploaded_file)
        st.dataframe(preview_df.head())
        if st.button("Run Batch Prediction"):
            uploaded_file.seek(0)
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{backend_url}/v1/predictbatch", files=files, timeout=60)
                response.raise_for_status()
                predictions = response.json()
                result_df = preview_df.copy()
                result_df["Predicted_Sales"] = [
                    predictions[str(i)] for i in range(len(result_df))
                ]
                st.dataframe(result_df)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
