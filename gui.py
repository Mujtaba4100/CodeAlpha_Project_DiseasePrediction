
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import matplotlib.pyplot as plt
from datetime import datetime


model = joblib.load("disease_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptom_list = joblib.load("symptom_list.pkl")
training_data = pd.read_csv("Training.csv").drop(columns=[col for col in ['Unnamed: 133'] if col in pd.read_csv("Training.csv").columns])

my_list = []

root = tk.Tk()
root.title("Disease Prediction System")
root.geometry("880x720")

search_var = tk.StringVar()

def search_symptoms():
    query = search_var.get().lower()
    listbox.delete(0, tk.END)
    for symptom in symptom_list:
        if query in symptom.lower():
            listbox.insert(tk.END, symptom)

def reset_list():
    listbox.delete(0, tk.END)
    for symptom in symptom_list:
        listbox.insert(tk.END, symptom)

def update_my_listbox():
    my_listbox.delete(0, tk.END)
    for item in my_list:
        my_listbox.insert(tk.END, item)

def toggle_symptom(event):
    selection = listbox.get(listbox.curselection())
    if selection in my_list:
        my_list.remove(selection)
    else:
        my_list.append(selection)
    update_my_listbox()

def remove_symptom():
    selection = my_listbox.get(tk.ACTIVE)
    if selection in my_list:
        my_list.remove(selection)
        update_my_listbox()

def suggest_symptoms(top_disease):
    subset = training_data[training_data['prognosis'] == top_disease]
    means = subset[symptom_list].mean().sort_values(ascending=False)
    suggested = [symptom for symptom in means.index if symptom not in my_list and means[symptom] > 0.4]
    if suggested:
        messagebox.showinfo("Suggested Symptoms", f"Based on '{top_disease}', consider adding:\n\n" + ", ".join(suggested[:10]))

    else:
        messagebox.showinfo("Suggested Symptoms", "No new suggestions based on current prediction.")


def matched_symptoms(disease):
    subset = training_data[training_data['prognosis'] == disease]
    common = subset[symptom_list].mean().sort_values(ascending=False)
    matched = [s for s in my_list if s in common and common[s] > 0.3]
    missing = [s for s in common.index if s not in my_list and common[s] > 0.4]
    return matched, missing[:10]

def log_prediction(symptoms, diseases, probs):
    log_df = pd.DataFrame({
        "symptoms": [", ".join(symptoms)],
        "top1": [f"{diseases[0]} ({probs[0]:.2f})"],
        "top2": [f"{diseases[1]} ({probs[1]:.2f})"],
        "top3": [f"{diseases[2]} ({probs[2]:.2f})"]
    })
    try:
        log_df.to_csv("prediction_history.csv", mode="a", header=not pd.read_csv("prediction_history.csv").shape[0], index=False)
    except:
        log_df.to_csv("prediction_history.csv", index=False)

def predict():
    if not my_list:
        messagebox.showwarning("Input Error", "Please select or enter symptoms.")
        return

    input_vector = np.zeros(len(symptom_list))
    for symptom in my_list:
        if symptom in symptom_list:
            input_vector[symptom_list.index(symptom)] = 1

    input_scaled = scaler.transform([input_vector])
    probs = model.predict_proba(input_scaled)[0]

    top3 = np.argsort(probs)[-3:][::-1]
    top3_diseases = label_encoder.inverse_transform(top3)
    top3_probs = probs[top3]

    result_lines = []
    for disease, prob in zip(top3_diseases, top3_probs):
        result_lines.append(f"{disease}: {prob:.2f}")

    matched, missing = matched_symptoms(top3_diseases[0])

    result_text = "\n".join(result_lines)
    result_text += f"\n\nMatched: {', '.join(matched)}"
   

    if top3_probs[0] < 0.5:
        result_text += "\n\n Low confidence. Please select more symptoms for better prediction."

    messagebox.showinfo("Prediction", f"Top Predicted Diseases:\n\n{result_text}")
   
    suggest_symptoms(top3_diseases[0])
    log_prediction(my_list, top3_diseases, top3_probs)

# Layout
tk.Label(root, text="Search Symptom:", font=("Arial", 12)).pack(pady=(10, 0))
search_frame = tk.Frame(root)
search_frame.pack()
tk.Entry(search_frame, textvariable=search_var, width=40).pack(side=tk.LEFT, padx=5)
tk.Button(search_frame, text="Search", command=search_symptoms).pack(side=tk.LEFT)
tk.Button(search_frame, text="Reset", command=reset_list).pack(side=tk.LEFT)

tk.Label(root, text="Symptom List (Click to Toggle Selection)", font=("Arial", 12)).pack()
listbox = tk.Listbox(root, selectmode=tk.SINGLE, width=50, height=15)
listbox.pack()
listbox.bind("<<ListboxSelect>>", toggle_symptom)
reset_list()

tk.Label(root, text="Selected Symptoms (my_list):", font=("Arial", 12)).pack(pady=(10, 0))
my_listbox = tk.Listbox(root, width=50, height=10)
my_listbox.pack()

tk.Button(root, text="← Remove Selected", command=remove_symptom).pack(pady=5)
tk.Button(root, text="Predict Disease", command=predict, font=("Arial", 14)).pack(pady=15)

root.mainloop()
