# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load model & data ONCE
# -----------------------------
model = pickle.load(open("model.sav", "rb"))
df_1 = pd.read_csv("first_telc.csv")

# -----------------------------
# Home page (GET)
# -----------------------------
@app.route("/", methods=["GET"])
def loadPage():
    return render_template("home.html")

# -----------------------------
# Prediction (POST)
# -----------------------------
@app.route("/", methods=["POST"])
def predict():
    try:
        # -------- SAFE INPUTS --------
        input_data = {
            "SeniorCitizen": int(request.form["query1"]),
            "MonthlyCharges": float(request.form["query2"]),
            "TotalCharges": float(request.form["query3"]),
            "gender": request.form["query4"].strip(),
            "Partner": request.form["query5"].strip(),
            "Dependents": request.form["query6"].strip(),
            "PhoneService": request.form["query7"].strip(),
            "MultipleLines": request.form["query8"].strip(),
            "InternetService": request.form["query9"].strip(),
            "OnlineSecurity": request.form["query10"].strip(),
            "OnlineBackup": request.form["query11"].strip(),
            "DeviceProtection": request.form["query12"].strip(),
            "TechSupport": request.form["query13"].strip(),
            "StreamingTV": request.form["query14"].strip(),
            "StreamingMovies": request.form["query15"].strip(),
            "Contract": request.form["query16"].strip(),
            "PaperlessBilling": request.form["query17"].strip(),
            "PaymentMethod": request.form["query18"].strip(),
            "tenure": int(request.form["query19"])
        }

        # -------- SAFE TENURE RANGE --------
        if input_data["tenure"] < 1:
            input_data["tenure"] = 1
        if input_data["tenure"] > 72:
            input_data["tenure"] = 72

        new_df = pd.DataFrame([input_data])

        # -------- TENURE GROUP --------
        labels = [f"{i}-{i+11}" for i in range(1, 72, 12)]
        new_df["tenure_group"] = pd.cut(
            new_df["tenure"],
            bins=range(1, 80, 12),
            labels=labels,
            right=False,
            include_lowest=True
        )

        new_df.drop(columns=["tenure"], inplace=True)

        # -------- DUMMIES --------
        new_df_dummies = pd.get_dummies(new_df)

        # -------- ALIGN WITH MODEL (SAFE) --------
        if hasattr(model, "feature_names_in_"):
            model_columns = model.feature_names_in_
            new_df_dummies = new_df_dummies.reindex(
                columns=model_columns, fill_value=0
            )

        # -------- PREDICT --------
        pred = model.predict(new_df_dummies)[0]
        prob = model.predict_proba(new_df_dummies)[0][1]

        if pred == 1:
            o1 = "This customer is likely to be churned ❌"
        else:
            o1 = "This customer is likely to continue ✅"

        o2 = f"Confidence: {prob*100:.2f}%"

        return render_template(
            "home.html",
            output1=o1,
            output2=o2,
            **request.form
        )

    except Exception as e:
        # SHOW REAL ERROR ON UI
        return render_template(
            "home.html",
            output1="Internal Error ❌",
            output2=str(e)
        )

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
