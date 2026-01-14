import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify, render_template
import csv
import logging

# --- 1. Setup ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- 2. Robust Data Loading and Model Training ---
def clean_text(text: str) -> str:
    if isinstance(text, str):
        return text.strip().replace(' ', '_').lower()
    return text

def load_csv_safely(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            num_columns = len(header)
            for i, row in enumerate(reader):
                row = (row + [None] * num_columns)[:num_columns]
                data.append(dict(zip(header, row)))
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Cannot find the file '{file_path}'. Make sure it's in the same folder as app.py.")
        return None
    except Exception as e:
        print(f"❌ FATAL ERROR reading {file_path}: {e}")
        return None

def train_model():
    print("Attempting to load data and train model...")
    df_train = load_csv_safely("dataset.csv")
    df_info = pd.read_csv("disease_data.csv", dtype=str).fillna("N/A") # Read all as text to avoid warnings
    
    if df_train is None or df_info is None:
        return None, None, None

    df_train['Disease'] = df_train['Disease'].apply(clean_text)
    df_info['Disease'] = df_info['Disease'].apply(clean_text)
    for col in df_train.columns:
        if col != 'Disease':
            df_train[col] = df_train[col].apply(clean_text)
    
    df_melted = df_train.melt(id_vars=['Disease'], value_name='Symptom').dropna(subset=['Symptom'])
    df_melted = df_melted[df_melted['Symptom'].astype(str).str.strip() != '']
    pivot_df = df_melted.pivot_table(index='Disease', columns='Symptom', aggfunc=lambda x: 1, fill_value=0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(pivot_df.values, pivot_df.index)
    
    print("✅ Model trained successfully!")
    return model, pivot_df.columns, df_info

model, all_symptoms_list, df_info = train_model()
if all_symptoms_list is not None:
    all_symptoms_list = sorted([str(s) for s in all_symptoms_list])

# --- 3. Flask Routes ---
@app.route('/')
def home():
    if model is None:
        return "<h1>Error: Model could not be trained. Please check the terminal for error messages.</h1>"
    return render_template('index.html', all_symptoms=all_symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.get_json().get('symptoms', [])
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    input_vector = [1 if s in symptoms else 0 for s in all_symptoms_list]
    predicted_disease = model.predict([input_vector])[0]
    
    try:
        disease_info = df_info[df_info['Disease'] == predicted_disease].iloc[0]
        description = disease_info['Overview']
        precautions_str = disease_info['Preventions']
        precautions = [p.strip() for p in precautions_str.split('.') if p.strip() and p.strip().lower() != 'n/a']
    except (IndexError, KeyError):
        description = "Detailed description not available."
        precautions = ["Consult a healthcare professional."]

    return jsonify({
        'disease': predicted_disease.replace('_', ' ').title(),
        'description': description,
        'precautions': precautions,
    })

# --- 4. Run the App ---
if __name__ == '__main__':
    if model is not None:
        print("🚀 Starting Flask server...")
        # --- THIS IS THE FIX ---
        # Running on 0.0.0.0 makes the server accessible from your browser without network issues.
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        print("❌ Server did not start because model training failed.")
