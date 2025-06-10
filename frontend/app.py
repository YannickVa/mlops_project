import os
import requests
from flask import Flask, render_template, request, abort

app = Flask(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8000")

FEATURE_SCHEMA = {
    "income_am": float,
    "profit_last_am": float,
    "profit_am": float,
    "damage_am": float,
    "damage_inc": float,
    "crd_lim_rec": float,
    "credit_use_ic": float,
    "lactose_ic": float,
    "insurance_ic": float,
    "spa_ic": float,
    "empl_ic": float,
    "cab_requests": float,
    "married_cd": lambda x: x.lower() in ["true", "1", "yes"],
    "bar_no": float,
    "sport_ic": float,
    "neighbor_income": float,
    "age": float,
    "marketing_permit": float,
    "urban_ic": float,
    "dining_ic": float,
    "presidential": float,
    "client_segment": float,
    "sect_empl": float,
    "prev_stay": float,
    "prev_all_in_stay": float,
    "fam_adult_size": float,
    "children_no": float,
    "tenure_mts": float,
    "company_ic": float,
    "claims_no": float,
    "claims_am": float,
    "nights_booked": float,
    "gender": int,
    "shop_am": float,
    "shop_use": float,
    "retired": float,
    "gold_status": float,
}


@app.route("/")
def index():
    features = [
        {"name": name, "type": "number" if dtype != bool else "text", "default": 0}
        for name, dtype in FEATURE_SCHEMA.items()
    ]
    features[12]["default"] = "false"
    return render_template("index.html", features=features)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = {
            key: FEATURE_SCHEMA[key](value) for key, value in request.form.items()
        }
    except (ValueError, KeyError) as e:
        abort(400, f"Invalid input data: {e}")

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        return render_template("result.html", **result)
    except requests.exceptions.RequestException as e:
        return render_template("result.html", error=f"API request failed: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
