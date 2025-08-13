import pandas as pd
import joblib

# Load model and original training columns
model_path = r"C:\Users\USER\Desktop\react\tut1\src\backend\stroke_model.pkl"
columns_path = r"C:\Users\USER\Desktop\react\tut1\src\backend\columns.pkl"
model = joblib.load(model_path)
original_cols = joblib.load(columns_path)

# Load dataset
csv_path = r"D:\final\healthcare-dataset-stroke-data.csv"
df = pd.read_csv(csv_path)

# Drop ID and stroke label (not used for prediction)
df = df.drop(columns=['id', 'stroke'], errors='ignore')

# Iterate through each row and predict
for index, row in df.iterrows():
    try:
        input_dict = {
            'gender': [row['gender']],
            'age': [float(row['age'])],
            'hypertension': [int(row['hypertension'])],
            'heart_disease': [int(row['heart_disease'])],
            'ever_married': [row['ever_married']],
            'work_type': [row['work_type']],
            'Residence_type': [row['Residence_type']],
            'avg_glucose_level': [float(row['avg_glucose_level'])],
            'bmi': [float(row['bmi']) if str(row['bmi']).replace('.', '', 1).isdigit() else 25.0],  # default bmi
            'smoking_status': [row['smoking_status']]
        }

        # Convert to DataFrame and apply same preprocessing
        input_df = pd.DataFrame(input_dict)
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=original_cols, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]
        if prediction == 1:
            print(f"✅ Stroke Risk Detected at row {index + 1}")
            print(input_dict)
            print("-" * 50)

    except Exception as e:
        print(f"❌ Error on row {index + 1}: {e}")
