import os
from datetime import datetime

import pandas as pd
import streamlit as st
from autogluon.common import TabularDataset
from autogluon.tabular import TabularPredictor
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split

AZURE_CONN_STR = os.getenv("AZURE_CONN_STR")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)


def log_query_to_blob(input_df: pd.DataFrame):
    input_df["query_time"] = datetime.utcnow().isoformat()

    csv_data = input_df.to_csv(index=False, header=False)

    blob_name = f"queries_{datetime.utcnow().strftime('%Y-%m-%d')}.csv"

    blob_client = container_client.get_blob_client(blob_name)

    try:
        blob_client.create_append_blob()
    except Exception:
        pass
    blob_client.append_block(csv_data)


train = False

if train:
    df = pd.read_csv("vehicles.csv", nrows=10000)
    df.drop(
        [
            "id",
            "url",
            "region",
            "region_url",
            "image_url",
            "VIN",
            "description",
            "county",
            "lat",
            "long",
            "posting_date",
        ],
        axis=1,
        inplace=True,
    )
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_data = TabularDataset(train_df)

    predictor = TabularPredictor(label="price", path="models").fit(
        train_data,
        presets="medium_quality",
        time_limit=600,
        excluded_model_types=["RF", "XT"],
    )


# for require_version_match=False
predictor = TabularPredictor.load("models")

st.set_page_config(page_title="AI Wycena Samochodu", page_icon="🚗")
st.title("🧠 AI Wycena Samochodu")
st.write("Wprowadź dane pojazdu, aby uzyskać szacunkową cenę rynkową.")

with st.form("car_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Rok produkcji", min_value=1980, max_value=2025, value=2015
        )
        manufacturer = st.selectbox(
            "Producent",
            ["ford", "toyota", "bmw", "audi", "chevrolet", "honda", "hyundai"],
        )
        model = st.text_input("Model", "corolla")
        condition = st.selectbox(
            "Stan", ["new", "like new", "excellent", "good", "fair", "salvage"]
        )
        fuel = st.selectbox(
            "Rodzaj paliwa", ["gas", "diesel", "electric", "hybrid", "other"]
        )

    with col2:
        odometer = st.number_input(
            "Przebieg (km)", min_value=0, max_value=1000000, value=100000
        )
        cylinders = st.selectbox(
            "Liczba cylindrów", ["4 cylinders", "6 cylinders", "8 cylinders", "other"]
        )
        transmission = st.selectbox("Skrzynia biegów", ["manual", "automatic", "other"])
        drive = st.selectbox("Napęd", ["fwd", "rwd", "4wd"])
        paint_color = st.selectbox(
            "Kolor lakieru",
            ["black", "white", "red", "blue", "silver", "gray", "other"],
        )

    title_status = st.selectbox(
        "Status tytułu", ["clean", "salvage", "rebuilt", "lien", "missing"]
    )
    car_type = st.selectbox(
        "Typ pojazdu",
        ["sedan", "SUV", "truck", "wagon", "hatchback", "convertible", "van"],
    )
    size = st.selectbox("Rozmiar", ["compact", "mid-size", "full-size", "other"])
    state = st.selectbox(
        "Stan USA", ["ca", "ny", "tx", "fl", "wa", "il", "pa", "oh", "az", "ga"]
    )

    submitted = st.form_submit_button("🔍 Oszacuj wartość")


if submitted:
    input_data = pd.DataFrame(
        [
            {
                "year": year,
                "manufacturer": manufacturer,
                "model": model,
                "condition": condition,
                "cylinders": cylinders,
                "fuel": fuel,
                "odometer": odometer,
                "title_status": title_status,
                "drive": drive,
                "transmission": transmission,
                "size": size,
                "type": car_type,
                "paint_color": paint_color,
                "state": state,
            }
        ]
    )
    log_query_to_blob(input_data)
    try:
        required_columns = predictor.feature_metadata.get_features()
        missing_cols = [
            col for col in required_columns if col not in input_data.columns
        ]

        if missing_cols:
            st.error(f"❌ Brakuje wymaganych kolumn: {missing_cols}")
        else:
            with st.spinner("Obliczanie szacunkowej wartości..."):
                prediction = predictor.predict(input_data)
                predicted_value = int(round(prediction.values[0]))
                st.success(f"💰 Szacunkowa wartość pojazdu: **{predicted_value} USD**")
    except Exception as e:
        st.error(f"Wystąpił błąd podczas predykcji: {e}")

st.sidebar.header("ℹ️ O aplikacji")
st.sidebar.write(
    """
Aplikacja wykorzystuje model AI (AutoGluon), który analizuje dane techniczne pojazdu
i przewiduje jego wartość rynkową.
"""
)
