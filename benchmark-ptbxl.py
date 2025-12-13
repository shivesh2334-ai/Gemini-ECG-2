import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import google.generativeai as genai
from PIL import Image
import time
import os
from tqdm import tqdm

# --- USER CONFIGURATION ---
# 1. Add your API Key
API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# 2. Point to your PTB-XL data (Change these paths to your real data locations)
IMAGE_FOLDER = "./ptb-xl/images/" 
LABELS_CSV = "./ptb-xl/labels.csv" # Must contain 'filename' and 'true_label' columns

# --- AI SETUP ---
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

def get_ai_diagnosis(image_path):
    """
    Sends image to Gemini for Zero-Shot Classification.
    """
    try:
        if not os.path.exists(image_path):
            return "Error: File Missing"
            
        img = Image.open(image_path)
        
        prompt = """
        Act as a diagnostic machine. 
        Classify this ECG into exactly one of the following labels:
        1. NORM (Normal)
        2. MI (Myocardial Infarction)
        3. STTC (ST/T Change)
        4. CD (Conduction Disturbance)
        5. HYP (Hypertrophy)
        
        Return ONLY the label code (e.g., "MI"). Do not add explanation.
        """
        
        response = model.generate_content([prompt, img])
        return response.text.strip()
        
    except Exception as e:
        print(f"Failed on {image_path}: {e}")
        return "Error"

def run_benchmark():
    # Load Data
    try:
        df = pd.read_csv(LABELS_CSV)
        # For testing, limit to first 20 rows. Remove .head(20) for full run.
        df = df.head(20) 
        print(f"Loaded {len(df)} records for benchmarking.")
    except FileNotFoundError:
        print("CSV file not found. Please check paths in USER CONFIGURATION.")
        return

    predictions = []
    true_labels = []

    print("🚀 Starting Batch Processing...")
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row['filename']
        actual = row['true_label']
        
        full_path = os.path.join(IMAGE_FOLDER, filename)
        
        # Get AI Prediction
        predicted = get_ai_diagnosis(full_path)
        
        predictions.append(predicted)
        true_labels.append(actual)
        
        # Rate limit pause (Gemini Free tier is 15 req/min)
        time.sleep(4) 

    # --- CALCULATE METRICS ---
    print("\n" + "="*40)
    print("📊 PERFORMANCE METRICS")
    print("="*40)
    
    # Filter out errors if any
    clean_true = [t for t, p in zip(true_labels, predictions) if p != "Error"]
    clean_pred = [p for t, p in zip(true_labels, predictions) if p != "Error"]
    
    accuracy = accuracy_score(clean_true, clean_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(clean_true, clean_pred, average='weighted', zero_division=0)

    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"Weighted Precision:{precision:.4f}")
    print(f"Weighted Recall:   {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    
    print("\n--- Detailed Class Breakdown ---")
    print(classification_report(clean_true, clean_pred, zero_division=0))

if __name__ == "__main__":
    run_benchmark()
