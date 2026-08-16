from flask import Flask, request, render_template

from src.house_price_prediction.pipelines.prediction_pipeline import (
    CustomData,
    PredictPipeline,
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict_price():
    # Step 1: pull each value out of the submitted form
    data = CustomData(
        area=int(request.form.get("area")),
        bedrooms=int(request.form.get("bedrooms")),
        bathrooms=int(request.form.get("bathrooms")),
        stories=int(request.form.get("stories")),
        parking=int(request.form.get("parking")),
        mainroad=request.form.get("mainroad"),
        guestroom=request.form.get("guestroom"),
        basement=request.form.get("basement"),
        hotwaterheating=request.form.get("hotwaterheating"),
        airconditioning=request.form.get("airconditioning"),
        prefarea=request.form.get("prefarea"),
        furnishingstatus=request.form.get("furnishingstatus"),
    )

    # Step 2: turn it into a DataFrame the model can understand
    pred_df = data.get_data_as_dataframe()

    # Step 3: load model + preprocessor and predict
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(pred_df)

    # Step 4: format the number nicely (e.g. 10,559,436) and show the result page
    predicted_price = round(prediction[0])
    formatted_price = f"{predicted_price:,}"

    return render_template("index.html", results=formatted_price)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
