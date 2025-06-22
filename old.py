from datetime import datetime

import pandas as pd
import streamlit as st
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

# wczytaj dataset
df = pd.read_csv("vehicles.csv", nrows=10000)
df.drop(["id", "url", "region", "region_url", "image_url", "VIN", "description", "county", "lat", "long", "posting_date"], axis=1, inplace=True)
# train test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
import pandas as pd
from autogluon.tabular import TabularPredictor

# mozliwe wartosci kolumn - do selectbox'a
train_data = TabularDataset(train_df)
print(train_data.head())
for col in df.columns:
    print(col)
print(df['condition'].unique())
print(df['cylinders'].unique())
print(df['fuel'].unique())
print(df['title_status'].unique())
print(df['transmission'].unique())
print(df['drive'].unique())
print(df['size'].unique())
print(df['type'].unique())
print(df['paint_color'].unique())
print(df['state'].unique())
# Wczytaj model
predictor = TabularPredictor.load("models")
# Zmień ścieżkę, jeśli model masz gdzie indziej

# odkomentuj aby trenować model - aktualne trenowanie na 10000 rekordów dla szybkości, dla lepszych wyników można: - dać więcej danych - zwiększyć preset na good/high -zwiększyć limit czasu
predictor = TabularPredictor(label="price", path="models").fit(train_data, presets="medium_quality", time_limit=600, excluded_model_types=['RF', 'XT'])
# wczytanie danych, zakomentuj jeśli trenujesz model
#predictor = TabularPredictor.load(path="models")
# Konfiguracja strony
st.set_page_config(page_title="AI Wycena Samochodu", page_icon="🚗")
st.title("🧠 AI Wycena Samochodu")
st.write("Wprowadź dane pojazdu, aby uzyskać szacunkową cenę rynkową.")

test_data = TabularDataset(test_df)
# 📋 Formularz danych
with st.form("car_form"):
    col1, col2 = st.columns(2)

predictions = predictor.predict(test_data)
    with col1:
        year = st.number_input("Rok produkcji", min_value=1980, max_value=2025, value=2015)
        manufacturer = st.selectbox("Producent", ["ford", "toyota", "bmw", "audi", "chevrolet", "honda", "hyundai"])
        model = st.text_input("Model", "corolla")
        condition = st.selectbox("Stan", ["new", "like new", "excellent", "good", "fair", "salvage"])
        fuel = st.selectbox("Rodzaj paliwa", ["gas", "diesel", "electric", "hybrid", "other"])

leaderboard = predictor.leaderboard()
    with col2:
        odometer = st.number_input("Przebieg (km)", min_value=0, max_value=1000000, value=100000)
        cylinders = st.selectbox("Liczba cylindrów", ["4 cylinders", "6 cylinders", "8 cylinders", "other"])
        transmission = st.selectbox("Skrzynia biegów", ["manual", "automatic", "other"])
        drive = st.selectbox("Napęd", ["fwd", "rwd", "4wd"])
        paint_color = st.selectbox("Kolor lakieru", ["black", "white", "red", "blue", "silver", "gray", "other"])

print(predictions.head())
print(leaderboard)
print(predictor.evaluate(train_data))
    title_status = st.selectbox("Status tytułu", ["clean", "salvage", "rebuilt", "lien", "missing"])
    car_type = st.selectbox("Typ pojazdu", ["sedan", "SUV", "truck", "wagon", "hatchback", "convertible", "van"])
    size = st.selectbox("Rozmiar", ["compact", "mid-size", "full-size", "other"])
    state = st.selectbox("Stan USA", ["ca", "ny", "tx", "fl", "wa", "il", "pa", "oh", "az", "ga"])

# dane przykładowego samochodu, trzeba przekazać jako DataFrame
    submitted = st.form_submit_button("🔍 Oszacuj wartość")

example_data = {
    'year': 2004,
    'manufacturer': "ford",
    'model': "f-150",
    'condition': "good",
    'cylinders': "6 cylinders",
    'fuel': "gas",
    'odometer': 100000,
    'title_status': "clean",
    'transmission': "automatic",
    'drive': "4wd",
    'size': "full-size",
    'type': "truck",
    'paint_color': "red",
    'state': "tx"
}
# 🔮 Predykcja
if submitted:
    input_data = pd.DataFrame([{
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
        "state": state
    }])

example_df = pd.DataFrame([example_data])
price = predictor.predict(example_df)
print("Car price prediction is: " + str(price.values[0]))
    # ✅ Sprawdzenie, czy kolumny się zgadzają
    try:
        required_columns = predictor.feature_metadata.get_features()
        missing_cols = [col for col in required_columns if col not in input_data.columns]

#odkomentuj aby zapisać wytrenowany model
#predictor.save()
        if missing_cols:
            st.error(f"❌ Brakuje wymaganych kolumn: {missing_cols}")
        else:
            with st.spinner("Obliczanie szacunkowej wartości..."):
                prediction = predictor.predict(input_data)
                # Zaokrąglanie do liczby całkowitej bez przecinków
                predicted_value = int(round(prediction.values[0]))  # Usuwanie części dziesiętnej
                st.success(f"💰 Szacunkowa wartość pojazdu: **{predicted_value} USD**")
    except Exception as e:
        st.error(f"Wystąpił błąd podczas predykcji: {e}")

def start_prediction():
    entered_data = {
        'year': st.session_state.year,
        'manufacturer': st.session_state.manufacturer,
        'model': st.session_state.model,
        'condition': st.session_state.condition,
        'cylinders': st.session_state.cylinders,
        'fuel': st.session_state.fuel,
        'odometer': st.session_state.odometer,
        'title_status': st.session_state.title_status,
        'transmission': st.session_state.transmission,
        'drive': st.session_state.drive,
        'size': st.session_state.size,
        'type': st.session_state.type,
        'paint_color': st.session_state.paint_color,
        'state': st.session_state.state
    }
    entered_df = pd.DataFrame([entered_data])
    with st.spinner("Calculating car price..."):
        result = predictor.predict(entered_df)
    st.info("Predicted car price is " + str(round(result.values[0], 2)) + " USD")

st.title("Car Price Predictor")
with st.form("car"):
    st.subheader("Enter car data:")
    st.number_input("Year of manufacture:", min_value=1950, max_value=datetime.now().year, key="year")
    st.text_input("Manufacturer:", key="manufacturer")
    st.text_input("Model:", key="model")
    st.selectbox("Condition:", options=df['condition'].unique()[1:], key="condition")
    st.selectbox("Cylinders:", options=df['cylinders'].unique()[1:], key="cylinders")
    st.selectbox("Fuel:", options=df['fuel'].unique()[1:], key="fuel")
    st.number_input("Odometer value:", min_value=0, max_value=1000000, key="odometer")
    st.selectbox("Title status:", options=df['title_status'].unique()[1:], key="title_status")
    st.selectbox("Transmission:", options=df['transmission'].unique()[1:], key="transmission")
    st.selectbox("Drive:", options=df['drive'].unique()[1:], key="drive")
    st.selectbox("Size:", options=df['size'].unique()[1:], key="size")
    st.selectbox("Type:", options=df['type'].unique()[1:], key="type")
    st.selectbox("Paint color:", options=df['paint_color'].unique()[1:], key="paint_color")
    st.selectbox("US State:", options=df['state'].unique()[1:], key="state")
    st.form_submit_button("Predict my price", on_click=start_prediction)
# ℹ️ Sidebar info
st.sidebar.header("ℹ️ O aplikacji")
st.sidebar.write("""
Aplikacja wykorzystuje model AI (AutoGluon), który analizuje dane techniczne pojazdu
i przewiduje jego wartość rynkową.
""")
