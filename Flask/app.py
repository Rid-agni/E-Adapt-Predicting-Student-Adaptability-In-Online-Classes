from flask import Flask, request, render_template
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import json


with open('encoder_mapping.json', 'r') as f:
    encoder_mapping = json.load(f)


app = Flask(__name__)


# Features and categories from training
FEATURES = [
    'Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student',
    'Location', 'Load-shedding', 'Financial Condition', 'Internet Type',
    'Network Type', 'Class Duration', 'Self Lms', 'Device'
]

CATEGORIES = [
    ['Boy', 'Girl'],
    ['11-15', '16-20', '21-25'],
    ['School', 'College', 'University'],
    ['Government', 'Non Government'],
    ['Yes', 'No'],  # IT Student
    ['Yes', 'No'],  # Location
    ['High', 'Low'],
    ['Poor', 'Mid', 'Rich'],
    ['Mobile Data', 'Wifi'],
    ['2G', '3G', '4G', 'No Internet Service'],
    ['0', '1-3', '3-6'],
    ['Yes', 'No'],
    ['Mobile', 'Laptop', 'Tab']
]

def manual_transform(user_input_dict):
    encoded = []
    for feature in FEATURES:  # Use fixed order
        value = user_input_dict.get(feature)
        if value is None:
            raise KeyError(f"Feature '{feature}' missing in input")
        if feature in encoder_mapping:
            try:
                encoded_val = encoder_mapping[feature].index(value)
            except ValueError:
                raise ValueError(f"Invalid value '{value}' for feature '{feature}'")
            encoded.append(encoded_val)
        else:
            raise KeyError(f"Feature '{feature}' not found in mapping")
    return [np.array(encoded, dtype=int)]



model = CatBoostClassifier()
model.load_model("catboost_retrained_model.cbm")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {
        "Gender": request.form['gender'],
        "Age": request.form['age'],
        "Education Level": request.form['education_level'],
        "Institution Type": request.form['institute_type'],
        "IT Student": request.form['it_student'],
        "Location": request.form['location'],
        "Load-shedding": request.form['load_shedding'],
        "Financial Condition": request.form['financial_condition'],
        "Internet Type": request.form['internet_type'],
        "Network Type": request.form['network_type'],
        "Class Duration": request.form['class_duration'],
        "Self Lms": request.form['self_lms'],
        "Device": request.form['device']
    }

    # ✅ Directly convert to DataFrame with raw string features
    input_df = pd.DataFrame([user_input])

    # 🧠 Predict
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)

    print("Prediction:", prediction)
    print("Probabilities:", probabilities)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
