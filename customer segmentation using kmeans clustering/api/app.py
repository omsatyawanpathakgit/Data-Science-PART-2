import math
from pathlib import Path

import joblib
from flask import Flask, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "complete_customer_segmentation_pipeline.pkl"

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "frontend"))
model = joblib.load(MODEL_PATH)


@app.get("/")
def index():
    return render_template("index.html", prediction=None)


@app.post("/predict")
def predict():
    income_value = request.form.get("income", "").strip()
    spending_value = request.form.get("spending", "").strip()

    try:
        income = float(income_value)
        spending = float(spending_value)
    except ValueError:
        return (
            render_template(
                "index.html",
                error="Enter a valid number for both income and spending score.",
                income=income_value,
                spending=spending_value,
                prediction=None,
            ),
            400,
        )

    if not math.isfinite(income) or income < 0:
        return (
            render_template(
                "index.html",
                error="Annual income must be a finite number of 0 or more.",
                income=income_value,
                spending=spending_value,
                prediction=None,
            ),
            400,
        )

    if not math.isfinite(spending) or not 1 <= spending <= 100:
        return (
            render_template(
                "index.html",
                error="Spending score must be a finite number from 1 to 100.",
                income=income_value,
                spending=spending_value,
                prediction=None,
            ),
            400,
        )

    cluster = int(model.predict([[income, spending]])[0])

    status_message = ""
    if cluster==0:
        status_message = "Moderate-Income, Moderate-Spending Customers"
    elif cluster==1:
        status_message = "High-Income, High-Spending Customers"
    elif cluster==2:
        status_message = "Low-Income, High-Spending Customers"
    elif cluster==3:
        status_message = "High-Income, Low-Spending Customers"
    else:
        status_message = "Low-Income, Low-Spending Customers"

    return render_template(
        "index.html",
        prediction=status_message,
        income=income_value,
        spending=spending_value,
    )


if __name__ == "__main__":
    app.run()
