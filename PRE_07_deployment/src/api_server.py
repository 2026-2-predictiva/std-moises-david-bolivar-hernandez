from flask import Flask, request
import pandas as pd
import pickle


app = Flask(__name__)


FEATURES = [
    # Aquí van los nombres exactos de las variables
    # utilizadas por el modelo.
    # Ejemplo:
    # "area",
    # "bedrooms",
    # "bathrooms",
    # "stories",
    # "parking"
]


@app.route("/", methods=["GET"])
def index():

    # Recibir los inputs enviados a la API
    args = request.args

    # Convertir los argumentos recibidos a enteros
    # y organizarlos según las variables del modelo
    filt_args = {
        key: [int(args[key])]
        for key in FEATURES
    }

    # Crear DataFrame con los inputs
    df = pd.DataFrame.from_dict(filt_args)

    # Cargar el modelo entrenado
    with open(
        "PRE_07_deployment/submission/house_predictor.pkl",
        "rb"
    ) as file:

        loaded_model = pickle.load(file)

    # Realizar la predicción
    prediction = loaded_model.predict(df)

    # Devolver la predicción como texto
    return str(prediction[0][0])


if __name__ == "__main__":
    app.run(debug=True)