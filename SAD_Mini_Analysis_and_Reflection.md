# Student Performance Prediction System
## SAD Mini-Analysis

### 1. Problem statement
Academic difficulties may only become visible after a student receives a poor result. This prototype explores whether historical student attributes and academic indicators can help identify records that may need additional academic support.

### 2. Users
The primary users are faculty members and academic analysts. They can review historical information and examine prediction results.

### 3. Inputs
The system accepts student attributes and academic indicators from the UCI dataset, including study time, absences, previous failures, and first- and second-period grades (G1 and G2). The uploaded CSV must contain the predictor columns used to train the model. G3 is not an input.

### 4. Process
1. Retrieve and inspect the dataset.
2. Remove duplicate records and create the binary target from G3.
3. Separate predictors from the target.
4. Impute missing values, scale numeric columns, and encode categorical columns.
5. Train a Random Forest classifier and evaluate it on a held-out test set.
6. Load the saved model in Streamlit.
7. Validate uploaded CSV columns, predict risk, display results, and allow CSV export.

### 5. Outputs
The system displays `AT RISK` or `NOT AT RISK` for each uploaded record. It also shows dataset information and model evaluation metrics, and provides downloadable prediction results.

### 6. Key limitation
The model is trained using one Portuguese secondary-school dataset. Student characteristics, grading practices, and school environments may differ at another institution, so the model may not generalize.

## Mini Use Case
**Actor:** Faculty member or academic analyst.

**Main flow:**
1. User opens the Streamlit application.
2. User downloads the sample input CSV or prepares a CSV with the required predictor columns.
3. User uploads the CSV.
4. System checks whether all required columns are present.
5. If columns are missing, the system displays an error.
6. If valid, the system applies the trained model.
7. System displays the prediction for each record and summary counts.
8. User downloads the results CSV.

## Reflection Answers

### 1. What is the prediction target, and why is G3 excluded from the input features?
The target is whether a student is classified as at risk of failing Mathematics: G3 below 10 means AT RISK, while G3 of 10 or higher means NOT AT RISK. G3 is excluded because it is the final outcome the model is supposed to predict. Including it as an input would reveal the answer to the model and cause target leakage.

### 2. What does the model's recall for the 'At risk' class tell you?
Recall for the At risk class tells us what proportion of the students who were actually at risk in the test data were correctly identified by the model. A higher recall means fewer actual at-risk records were missed. It does not by itself tell us how many predicted at-risk records were correct; precision helps answer that.

### 3. Why might a model trained on this dataset perform differently on students from your institution?
The dataset comes from a different educational setting in Portugal. Our institution may have different student backgrounds, teaching methods, grading systems, attendance patterns, and learning resources. These differences can affect the relationships learned by the model.

### 4. What action should a faculty member take after seeing an 'AT RISK' prediction?
The faculty member should treat the result as an early signal for supportive follow-up, not as a final judgment. They should review the student's situation, verify the information, talk with the student respectfully, and consider suitable academic support such as consultation, tutoring, or learning resources.

## Testing Checklist
- [ ] Run `python train_model.py`; confirm that the dataset, model, metrics, and sample CSV are created.
- [ ] Run `streamlit run app.py`; confirm that the dashboard opens.
- [ ] Download the sample input CSV and upload it; confirm that predictions appear.
- [ ] Upload a CSV with a required column removed; confirm that a helpful error appears.
- [ ] Download prediction results; confirm that a CSV file is produced.
- [ ] Open the GitHub repository and confirm that the source files and README are visible.

## Evaluation Results
Run `python train_model.py` on your computer to generate the real accuracy, precision, recall, F1-score, and confusion matrix. Enter the values from `model_metrics.json` or the terminal output in your submission. Do not invent evaluation scores before the model has actually run.
