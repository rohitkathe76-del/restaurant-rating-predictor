# 🍽️ Restaurant Rating Predictor

An interactive web app that predicts a restaurant's Zomato rating based on its location, cuisine, cost, and service features — built on the Zomato Bangalore Restaurants dataset using a tuned LightGBM regression model.

🔗 **Live App:** [Add your Streamlit Cloud URL here once deployed]

---

## ✨ Features

- **Instant rating prediction** from 10 restaurant attributes (location, cuisine, cost, votes, service options, etc.)
- **Visual gauge display** showing the predicted rating on a color-coded scale
- **Comparison insights** — predicted rating vs. city-wide average
- **Plain-language insights** explaining what factors are shaping the prediction (pricing tier, popularity, service convenience)
- **Actionable tips** for restaurants with lower predicted ratings
- Clean, restaurant-themed, interactive UI built with Streamlit

---

## 🧠 How It Works

1. **Data Cleaning** — Raw Zomato data cleaned (junk rating values removed, cost/votes converted to numeric, missing values handled).
2. **Feature Engineering** — Low-cardinality features (online order, table booking, restaurant type) one-hot encoded; high-cardinality features (location, cuisine, restaurant name) target-encoded using leakage-safe, train-only category means.
3. **Model Training** — Multiple regression models compared (Linear, Ridge, Lasso, Gradient Boosting, KNN, XGBoost, LightGBM); best model selected via two rounds of hyperparameter tuning with explicit overfitting checks (train-test R² gap).
4. **Final Model** — LightGBM Regressor, achieving ~0.93 R² on held-out test data with a small train-test gap, confirming strong generalization.
5. **Deployment** — Trained model and encoding maps serialized with `joblib`; Streamlit app loads them for real-time predictions.

---

## 📁 Project Structure

```
├── app.py                      # Streamlit app (UI + inference)
├── train_and_save_model.py     # Trains the model and saves all artifacts
├── requirements.txt            # Python dependencies
├── model.pkl                   # Trained LightGBM model
├── target_encoding_maps.pkl    # Category → average rating mappings
├── global_mean.pkl             # Fallback value for unseen categories
├── final_columns.pkl           # Exact feature column order used in training
├── dropdown_values.pkl         # Valid category options for the UI dropdowns
└── README.md
```

> Note: `zomato.csv` (the raw dataset) is not included in this repo due to GitHub's file size limits. Download it from [Kaggle](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) if you want to retrain the model yourself.

---

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model — requires zomato.csv in the project folder
python train_and_save_model.py

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Model** — LightGBM
- **Frontend** — Streamlit
- **Visualization** — Plotly
- **Deployment** — Streamlit Community Cloud

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Test R² | ~0.93 |
| Test RMSE | ~0.10 |
| Train-Test Gap | ~0.01–0.02 (no significant overfitting) |

---

## 📌 Dataset

[Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) — ~51,700 restaurant listings scraped from Zomato, covering location, cuisine, cost, service features, and customer ratings across Bengaluru.

---

## 📄 License

This project is for educational purposes. Dataset credit: Himanshu Poddar (Kaggle).
