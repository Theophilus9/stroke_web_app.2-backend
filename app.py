from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React

# Load the pre-trained model and columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "stroke_model.pkl")
columns_path = os.path.join(BASE_DIR, "columns.pkl")

model = joblib.load(model_path)
original_cols = joblib.load(columns_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from React
        data = request.get_json()
        print("Received from frontend:", data)

        input_dict = {
        'gender': [data['gender']],
        'age': [float(data['age'])],
        'hypertension': [int(data['hypertension'])],
        'heart_disease': [int(data['heart_disease'])],
        'ever_married': [data['ever_married']],
        'work_type': [data['work_type']],
        'Residence_type': [data['Residence_type']],
        'avg_glucose_level': [float(data['avg_glucose_level'])],
        'bmi': [float(data['bmi'])],
        'smoking_status': [data['smoking_status']]
        }
        # Convert to DataFrame
        input_df = pd.DataFrame(input_dict)

        # One-hot encode same as during training
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Align columns to what the model expects
        input_encoded = input_encoded.reindex(columns=original_cols, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]
        result = "Stroke Risk Detected" if prediction == 1 else "No Stroke Risk Dtetected"


        # Just return a test message to confirm connection
        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
