from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('air_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([
        data['PM2.5'],
        data['PM10'],
        data['CO'],
        data['NO2'],
        data['SO2'],
        data['O3'],
        data['Temperature'],
        data['Humidity'],
        data['WindSpeed']
    ])
    features = scaler.transform([features])
    prediction = model.predict(features)
    result = prediction[0]
    
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
