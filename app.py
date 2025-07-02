from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model and preprocessing objects
model = tf.keras.models.load_model("model/best_habitability_model_neural_network_absolute_best_v2.h5")

with open("model/habitability_preprocessor_absolute_best_v2.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("model/habitability_features_final.pkl", "rb") as f:
    feature_names = pickle.load(f)  # Should be a list of strings

@app.route("/")
def home():
    return render_template("index.html", features=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(request.form.get(f)) for f in feature_names]
        processed_input = preprocessor.transform([input_data])
        prediction = model.predict(processed_input)
        result = "Habitable" if prediction[0][0] > 0.5 else "Not Habitable"
        return render_template("result.html", prediction=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
