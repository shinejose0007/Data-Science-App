# Data Scientist Demo (Streamlit)

This is a fully interactive Streamlit application designed to demonstrate a data science workflow, including data exploration, machine learning, SAP data ingestion, and Retrieval-Augmented Generation (RAG). The app is structured to mimic a real-world business scenario using synthetic customer data and SAP-like data sources.

## 1. Data Management

### a) Synthetic Customer Dataset
The app includes a synthetic customer dataset (`synthetic_customers.csv`) for demonstration purposes.

You can view the dataset directly in the app via the sidebar button:

- **Show synthetic dataset** → Displays the first 10 rows.

This dataset is used for training a machine learning model and generating exploratory data analysis (EDA) plots.

### b) SAP Mock Ingestion
To simulate SAP BTP data ingestion, the app provides a mock ingestion function (`sap_btp_mock.py`).

Sidebar buttons:

- **Run SAP mock ingestion** → Simulates pulling data from SAP and saves it to `sap_ingested_mock.csv`.
- **Show SAP ingested data** → Displays the first 10 rows of the ingested SAP-like dataset.

This mimics business-critical ETL pipelines in SAP without needing a real SAP connection.

## 2. Machine Learning Model

### a) Model Training
The app trains a RandomForest classifier using the synthetic dataset.

Sidebar button:

- **Train model** → Trains the model and saves:
  - `model.joblib` → the trained RandomForest model
  - `scaler.joblib` → a StandardScaler for feature scaling

The model predicts whether a customer is high-value (`target_highvalue` column) based on features like:

- Age
- Years of experience
- Education level
- Salary

### b) Model Loading
If the model has been trained or exists in `/models/`, it is loaded automatically.

Streamlit displays a success message if the model is loaded, or a warning if no model is found.

## 3. Prediction Interface
Users can enter customer features via input controls:

- Age
- Years of experience
- Education level (HS / BSc / MSc+)
- Salary in EUR

After clicking **Predict**, the app:

- Scales the features using the saved StandardScaler.
- Uses the trained RandomForest model to make a prediction.
- Displays:
  - Prediction (0 = low-value, 1 = high-value)
  - Probability of class=1 (if `predict_proba` is available)

This allows real-time interactive prediction for different hypothetical customers.

## 4. RAG — Ask the Docs
RAG (Retrieval-Augmented Generation) allows the app to answer questions based on a document corpus (`/docs`) using embeddings + FAISS.

Features:

- **Ask** → Enter a natural language question about SAP, ML, or business processes.
- The app converts documents into embeddings, uses FAISS to retrieve the most relevant sentences, and generates a textual answer.
- **Validation mode** → Predefined sample questions can be run to check the RAG system’s accuracy, showing question, answer, matched keywords, and score.

## 5. EDA & Explainability
Users can explore the dataset and model insights interactively:

- **Summary statistics** → Mean, standard deviation, min, max, etc.
- **Value counts of target variable** → Distribution of high-value vs low-value customers.
- **Plots**:
  - Age distribution (histogram)
  - Feature importance (bar chart from RandomForest)
- Supports **matplotlib** or **Plotly**. Automatically falls back if a plotting library is unavailable.

## 6. File Structure
DataScience_streamlit_app/
├─ app.py # Main Streamlit app
├─ src/
│ ├─ train_model.py # Training logic for RandomForest
│ ├─ sap_btp_mock.py # Mock SAP ingestion
│ └─ rag.py # RAG implementation
├─ data/
│ ├─ synthetic_customers.csv
│ └─ sap_ingested_mock.csv
├─ models/
│ ├─ model.joblib # Saved RandomForest model
│ └─ scaler.joblib # Saved StandardScaler
└─ docs/ # Document corpus for RAG
