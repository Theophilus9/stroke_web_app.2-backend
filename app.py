from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("new_model.pkl")

# Label encoding mappings (MUST match training exactly)
label_encoders = {
    "gender": {"Female": 0, "Male": 1, "Other": 2},
    "ever_married": {"No": 0, "Yes": 1},
    "work_type": {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4},
    "Residence_type": {"Rural": 0, "Urban": 1},
    "smoking_status": {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received from frontend:", data)

        # Encode categorical variables using mappings
        for col, mapping in label_encoders.items():
            data[col] = mapping[data[col]]

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([data], columns=[
            "gender", "age", "hypertension", "heart_disease",
            "ever_married", "work_type", "Residence_type",
            "avg_glucose_level", "bmi", "smoking_status"
        ])

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Stroke Risk Detected" if prediction == 1 else "No Stroke Risk Detected"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=7860)
