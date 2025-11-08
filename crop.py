from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open("crop_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            N = float(request.form["nitrogen"])
            P = float(request.form["phosphorus"])
            K = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
            yield_hd = float(request.form["yield_hd"])  

            features = np.array([[N, P, K, temperature, humidity, ph, rainfall, yield_hd]])
            prediction = model.predict(features)[0]
        except:
            prediction = "Invalid input. Please enter numeric values."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)



