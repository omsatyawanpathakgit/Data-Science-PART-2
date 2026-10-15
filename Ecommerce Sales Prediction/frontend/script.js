const form = document.querySelector("#prediction-form");
const submitButton = form.querySelector("button[type='submit']");
const resultCard = document.querySelector(".result-card");
const resultValue = document.querySelector("#result-value");
const resultMessage = document.querySelector("#result-message");
const resultStatus = document.querySelector("#result-status");
const orderDateInput = form.elements.order_date;
const orderDateError = document.querySelector("#order-date-error");

function isPastOrderDate(value) {
  if (!value) return false;

  const today = new Date();
  const todayLocal = [
    today.getFullYear(),
    String(today.getMonth() + 1).padStart(2, "0"),
    String(today.getDate()).padStart(2, "0"),
  ].join("-");

  return value < todayLocal;
}

function updateOrderDateError() {
  orderDateError.hidden = !isPastOrderDate(orderDateInput.value);
}

orderDateInput.addEventListener("input", updateOrderDateError);
orderDateInput.addEventListener("change", updateOrderDateError);

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  updateOrderDateError();
  if (!orderDateError.hidden) {
    orderDateInput.focus();
    window.requestAnimationFrame(() => {
      window.alert("Customers cannot add an order date before the system's date.");
    });
    return;
  }

  if (!form.reportValidity()) return;

  const values = new FormData(form);
  const body = {
    product_category: values.get("product_category"),
    region: values.get("region"),
    payment_method: values.get("payment_method"),
    order_date: values.get("order_date"),
    quantity: Number(values.get("quantity")),
    unit_price: Number(values.get("unit_price")),
    discount: Number(values.get("discount")),
  };

  submitButton.disabled = true;
  resultCard.classList.remove("is-error");
  resultCard.classList.add("is-loading");
  resultValue.textContent = "Calculating";
  resultMessage.textContent = "The model is processing the order details.";
  resultStatus.textContent = "Prediction in progress";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const responseText = await response.text();
    let result;
    try {
      result = JSON.parse(responseText);
    } catch {
      throw new Error("The prediction service returned an invalid response.");
    }

    if (!response.ok) {
      throw new Error(result.error || "The prediction could not be calculated.");
    }
    if (!Number.isFinite(result.prediction)) {
      throw new Error("The prediction service returned an invalid estimate.");
    }

    resultValue.textContent = new Intl.NumberFormat(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(result.prediction);
    resultMessage.textContent = "Estimated revenue for the order you described.";
    resultStatus.textContent = "Forecast ready";
  } catch (error) {
    resultValue.textContent = "Not available";
    resultMessage.textContent = error.message || "Unable to reach the prediction service.";
    resultStatus.textContent = "Prediction failed";
    resultCard.classList.add("is-error");
  } finally {
    resultCard.classList.remove("is-loading");
    submitButton.disabled = false;
  }
});
