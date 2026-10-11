from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("dengue_rf_model.pkl")
df = pd.read_csv("Dengue_preprocessed.csv")
df_year = df.groupby("Year")["DengueCases"].sum().reset_index()
df_city = df.groupby("City")["DengueCases"].sum().reset_index()
temperatures = df["Temperature_C"].tolist()
humidities = df["Humidity"].tolist()
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        year = float(request.form["year"])
        week = float(request.form["week"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        precipitation = float(request.form["precipitation"])

        prediction = model.predict([[
            year,
            week,
            temperature,
            humidity,
            precipitation
        ]])[0]

        prediction = round(prediction)

    return render_template(
    "index.html",
    prediction=prediction,
    data=df,
    years=df_year["Year"].tolist(),
    cases=df_year["DengueCases"].tolist(),
    cities=df_city["City"].tolist(),
city_cases=df_city["DengueCases"].tolist(),
temperatures=temperatures,
humidities=humidities,
    )
if __name__ == "__main__":
    app.run(debug=True)