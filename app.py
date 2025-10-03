import pandas as pd
import pickle
from flask import Flask, render_template

# Initialize Flask app
app = Flask(__name__)

# Load ML model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset (your CSV with inputs)
df = pd.read_csv("data.csv")

# Example preprocessing function (adjust to your real columns & pipeline)
def get_predictions(data):
    # Get probabilities from model
    probs = model.predict_proba(data.drop(columns=["Planet_Name", "ID"], errors="ignore"))
    preds = model.predict(data.drop(columns=["Planet_Name", "ID"], errors="ignore"))

    # Attach results
    data["Prediction"] = preds
    data["Confidence"] = probs.max(axis=1) * 100
    return data

# Route for dashboard
@app.route("/")
def dashboard():
    # Run predictions
    results = get_predictions(df.copy())

    # Count stats
    total = len(results)
    confirmed = (results["Prediction"] == "confirmed").sum()
    candidates = (results["Prediction"] == "candidates").sum()
    false_positives = (results["Prediction"] == "false").sum()

    # Prepare cards (like in your HTML)
    exoplanets = []
    for _, row in results.iterrows():
        exoplanets.append({
            "name": row.get("Planet_Name", "Unknown"),
            "id": row.get("ID", "N/A"),
            "method": row.get("Method", "Unknown"),
            "magnitude": row.get("Magnitude", "N/A"),
            "period": row.get("Orbital_Period", None),
            "confidence": round(row["Confidence"], 1),
            "status": row["Prediction"],  # confirmed/candidates/false
            "dataset": row.get("Dataset", "N/A"),
            "date": row.get("Analysed", "Unknown")
        })

    return render_template("index.html",
                           total=total,
                           confirmed=confirmed,
                           candidates=candidates,
                           false=false_positives,
                           exoplanets=exoplanets)

if __name__ == "__main__":
    app.run(debug=True)
