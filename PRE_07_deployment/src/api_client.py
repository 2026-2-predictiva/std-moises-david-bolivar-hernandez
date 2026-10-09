from flask import Flask, request
import pandas as pd
import pickle


app = Flask(__name__)


FEATURES = [
    "bathrooms",
    "bedrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


@app.route("/", methods=["POST"])
def index():

    args = request.json

    filt_args = {
        key: [int(args[key])]
        for key in FEATURES
    }

    df = pd.DataFrame.from_dict(filt_args)

    with open(
        "PRE_07_deployment/submission/house_predictor.pkl",
        "rb"
    ) as file:
        loaded_model = pickle.load(file)

    prediction = loaded_model.predict(df)

    return str(prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
    