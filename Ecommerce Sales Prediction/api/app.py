from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException


PROJECT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MODEL_PATH = PROJECT_DIR / "best_revenue_prediction_model.pkl"
PREPROCESSING_PATH = PROJECT_DIR / "data_preprocessing.pkl"

PRODUCT_CATEGORIES = {"Beauty", "Clothing", "Electronics", "Home"}
REGIONS = {"East", "North", "South", "West"}
PAYMENT_METHODS = {"Card", "COD", "Wallet"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")
model = joblib.load(MODEL_PATH)
preprocessing = joblib.load(PREPROCESSING_PATH)
feature_columns = preprocessing["feature_columns"]
model_feature_columns = model.get_booster().feature_names
if model_feature_columns != feature_columns:
    raise RuntimeError(
        "The saved model's feature names do not match the preprocessing artifact."
    )


@app.errorhandler(HTTPException)
def handle_http_error(error: HTTPException):
    return jsonify(error=error.description), error.code


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    app.logger.exception("Prediction request failed", exc_info=error)
    return jsonify(
        error="The prediction service encountered an internal error. "
        "Check the Flask terminal for details."
    ), 500


def required_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field.replace('_', ' ').capitalize()} must be a number.")
    if not np.isfinite(value):
        raise ValueError(f"{field.replace('_', ' ').capitalize()} must be finite.")
    return float(value)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/predict")
def predict():
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify(error="Send the prediction details as a JSON object."), 400

    product_category = payload.get("product_category")
    region = payload.get("region")
    payment_method = payload.get("payment_method")

    if not isinstance(product_category, str) or product_category not in PRODUCT_CATEGORIES:
        return jsonify(error="Choose a valid product category."), 400
    if not isinstance(region, str) or region not in REGIONS:
        return jsonify(error="Choose a valid region."), 400
    if not isinstance(payment_method, str) or payment_method not in PAYMENT_METHODS:
        return jsonify(error="Choose a valid payment method."), 400

    try:
        quantity = required_number(payload, "quantity")
        unit_price = required_number(payload, "unit_price")
        discount = required_number(payload, "discount")
        order_date = date.fromisoformat(payload.get("order_date", ""))
    except (TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400

    today = date.today()
    if order_date < today:
        return jsonify(error="Order date cannot be before the current system date."), 400

    if quantity <= 0 or not quantity.is_integer():
        return jsonify(error="Quantity must be a positive whole number."), 400
    if unit_price < 0:
        return jsonify(error="Unit price cannot be negative."), 400
    if not 0 <= discount <= 1:
        return jsonify(error="Discount must be between 0 and 1."), 400

    features = pd.DataFrame(
        [
            {
                "product_category": product_category,
                "region": region,
                "quantity": int(quantity),
                "unit_price": unit_price,
                "discount": discount,
                "payment_method": payment_method,
                "order_year": order_date.year,
                "order_month": order_date.month,
                "order_day": order_date.day,
                "order_dayofweek": order_date.weekday(),
            }
        ]
    )

    numeric_columns = preprocessing["num_cols"]
    categorical_columns = preprocessing["cat_cols"]
    features[numeric_columns] = preprocessing["scaler"].transform(
        features[numeric_columns]
    )
    features = pd.get_dummies(features, columns=categorical_columns)
    features = features.reindex(
        columns=preprocessing["feature_columns"], fill_value=0
    )

    # Keep the saved training column order; XGBoost's pandas adapter can lose
    # feature names with newer pandas releases.
    feature_values = features.to_numpy(dtype=np.float32)
    prediction = float(
        np.asarray(model.predict(feature_values, validate_features=False))
        .reshape(-1)[0]
    )
    if not np.isfinite(prediction):
        raise RuntimeError("The model returned a non-finite prediction.")

    return jsonify(prediction=round(prediction, 2))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
