# Student Performance Prediction System

A classroom Data Mining and Systems Analysis and Design prototype. It uses the UCI Student Performance dataset (Mathematics records) and a Random Forest classifier to estimate whether a record is **AT RISK** or **NOT AT RISK**.

## Prediction rule
- `G3 < 10` → `AT RISK` (class 1)
- `G3 >= 10` → `NOT AT RISK` (class 0)

**Important:** `G3` is the target and is never used as an input feature. This prevents target leakage.

## Technologies
- Python
- pandas
- scikit-learn
- ucimlrepo
- joblib
- Streamlit
- Git and GitHub

## Dataset
UCI Machine Learning Repository — Student Performance, dataset ID 320:
https://archive.ics.uci.edu/dataset/320/student+performance

The training script retrieves the data online and saves `student_data.csv`.

## Project files
- `train_model.py` — downloads data, preprocesses features, trains and evaluates the model.
- `app.py` — Streamlit dashboard, CSV upload, predictions, and CSV export.
- `requirements.txt` — Python dependencies.
- `README.md` — project documentation.
- Generated after training: `student_data.csv`, `student_model.pkl`, `model_metrics.json`, `sample_prediction_input.csv`.

## Installation and run (Windows)
1. Install Python 3.10 or later.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. (Recommended) Create and activate a virtual environment:

   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Download the dataset and train the model:

   ```bash
   python train_model.py
   ```

7. Start the app:

   ```bash
   streamlit run app.py
   ```

8. Open the local address printed in the terminal (normally http://localhost:8501).
9. In the app, download the sample input CSV, upload it, review predictions, and download the results.

## GitHub
Create a repository named `student-performance-prediction`, then run these commands in the project folder (replace `YOUR-USERNAME`):

```bash
git init
git add train_model.py app.py requirements.txt README.md
git commit -m "Build student performance prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Do not commit passwords, access tokens, or other secrets. Follow your instructor's instructions about whether generated dataset/model files may be uploaded.

## Limitations
- The model uses one public Portuguese secondary-school dataset and may not generalize to another school, country, or student population.
- Historical associations do not prove causes of academic performance.
- Predictions can be wrong; review recall, precision, F1-score, and the confusion matrix, not accuracy alone.
- This is a classroom prototype, not a validated student assessment or automated decision system.
- A faculty member should use a prediction only as a prompt for supportive review, not as the sole basis for a decision.
