import streamlit as st
import pandas as pd, numpy as np, joblib, os
from pathlib import Path
st.set_page_config(page_title="sovanta DataSci Demo", layout="wide")

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC = str(PROJECT_ROOT / "src")
if SRC not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

import src.rag as ragmod

st.title("sovanta — (Junior) Data Scientist Demo (Streamlit)")
st.sidebar.title("Controls")

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data
def load_synthetic():
    p = PROJECT_ROOT / "data" / "synthetic_customers.csv"
    return pd.read_csv(p)

@st.cache_data
def load_sap_ingested():
    p = PROJECT_ROOT / "data" / "sap_ingested_mock.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

def run_sap_mock():
    try:
        import src.sap_btp_mock as sap
        sap.main()
        return True, "SAP mock ingestion completed."
    except Exception as e:
        return False, str(e)

def train_model():
    try:
        import src.train_model as tm
        tm.main()
        return True, "Training finished."
    except Exception as e:
        return False, str(e)

def load_model():
    model_path = PROJECT_ROOT / "models" / "model.joblib"
    scaler_path = PROJECT_ROOT / "models" / "scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler

def validate_rag():
    TESTS = [
        {"q": "How does SAP help manage customer data for analytics?", "expected": ["customer data","analytics","centralized"]},
        {"q": "What is the role of SAP BTP in integrating machine learning models?", "expected": ["SAP BTP","machine learning","integration"]},
        {"q": "What are the benefits of using SAP for supply chain optimization?", "expected": ["supply chain","optimization","efficiency"]},
        {"q": "What is the main advantage of using RAG instead of a standard chatbot?", "expected": ["retrieval","documents","context"]},
        {"q": "How does FAISS improve document search in a RAG system?", "expected": ["FAISS","similarity","search"]},
    ]
    docs_path = str(PROJECT_ROOT / "docs")
    sents, embeddings, model, index = ragmod.build_from_docs(docs_path)
    results = []
    for test in TESTS:
        q, exp = test["q"], test["expected"]
        ans = ragmod.answer(q, sents, embeddings, model, index, topk=5)
        matched = sum(1 for kw in exp if kw.lower() in ans.lower())
        score = matched/len(exp)
        results.append({"question": q,"answer": ans,"score": score,"matched_keywords": matched,"expected_keywords": exp})
    return results

# -----------------------------
# Sidebar: Data
# -----------------------------
st.sidebar.header("Data")
if st.sidebar.button("Show synthetic dataset"):
    df = load_synthetic()
    st.subheader("Synthetic dataset (first 10 rows)")
    st.dataframe(df.head(10))

if st.sidebar.button("Run SAP mock ingestion"):
    ok, msg = run_sap_mock()
    if ok:
        st.success(msg)
    else:
        st.error(f"Error: {msg}")

if st.sidebar.button("Show SAP ingested data"):
    df_sap = load_sap_ingested()
    if df_sap is None:
        st.warning("No SAP ingested mock found. Run 'Run SAP mock ingestion' first.")
    else:
        st.subheader("SAP ingested mock (first 10 rows)")
        st.dataframe(df_sap.head(10))

# -----------------------------
# Sidebar: Model
# -----------------------------
st.sidebar.header("Model")
if st.sidebar.button("Train model"):
    with st.spinner("Training model..."):
        ok, msg = train_model()
    if ok:
        st.success(msg)
    else:
        st.error(f"Training failed: {msg}")

clf, scaler = load_model()
if clf is None:
    st.warning("No trained model found. Click 'Train model' to train, or use included model in /models.")
else:
    st.success("Model loaded.")

# -----------------------------
# Prediction
# -----------------------------
st.header("Prediction")
col1, col2, col3, col4 = st.columns(4)
age = int(col1.number_input("Age", min_value=18, max_value=80, value=30))
years_experience = int(col2.number_input("Years of experience", min_value=0, max_value=60, value=5))
education = int(col3.selectbox("Education level", options=[1,2,3], index=1, format_func=lambda x: {1:'HS',2:'BSc',3:'MSc+'}[x]))
salary = float(col4.number_input("Salary (EUR)", min_value=1000, max_value=500000, value=52000))

if st.button("Predict"):
    if clf is None:
        st.error("No model available. Train model first.")
    else:
        input_df = pd.DataFrame([{"age": age,"years_experience": years_experience,"education": education,"salary": salary}])
        Xs = scaler.transform(input_df)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xs)
            if proba.shape[1] > 1:
                result = float(proba[0,1])
                st.write("Prediction (1 = high-value):", int(clf.predict(Xs)[0]))
                st.write(f"Predicted probability (class=1): {result:.3f}")
            else:
                result = float(proba[0,0])
                st.write("Prediction probability:", result)
        else:
            result = float(clf.predict(Xs)[0])
            st.write("Prediction value:", result)

# -----------------------------
# RAG ask-docs + validation
# -----------------------------
st.header("RAG — Ask the docs (embeddings + FAISS with fallback)")
query = st.text_input("Enter a question about SAP / RAG / ML in business")

colq1, colq2 = st.columns(2)
if colq1.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            sents, embeddings, model, index = ragmod.build_from_docs(str(PROJECT_ROOT / "docs"))
            answer = ragmod.answer(query, sents, embeddings, model, index, topk=5)
            st.subheader("Answer")
            st.text(answer)
        except Exception as e:
            st.error(f"RAG error: {e}")

if colq2.button("Run RAG Validation"):
    with st.spinner("Running validation tests..."):
        try:
            results = validate_rag()
            st.subheader("RAG Validation Results")
            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.write(f"**Answer:** {r['answer']}")
                st.write(f"✅ Matched {r['matched_keywords']} / {len(r['expected_keywords'])} expected keywords")
                st.progress(r['score'])
                st.markdown("---")
            st.button("Export validation results (disabled in this version)")
        except Exception as e:
            st.error(f"Validation error: {e}")

# -----------------------------
# EDA & Explainability
# -----------------------------
st.header("EDA & Explainability")
with st.expander("Show dataset summary"):
    df = load_synthetic()
    st.write(df.describe())
    st.write("Value counts for target:")
    st.write(df['target_highvalue'].value_counts())

with st.expander("Show plots"):
    df = load_synthetic()

    use_matplotlib = False
    use_plotly = False
    mpl_err = None
    plotly_err = None
    try:
        import matplotlib.pyplot as plt
        use_matplotlib = True
    except Exception as e:
        mpl_err = str(e)

    if not use_matplotlib:
        try:
            import plotly.express as px
            use_plotly = True
        except Exception as e:
            plotly_err = str(e)

    if use_matplotlib:
        fig1, ax1 = plt.subplots()
        df['age'].hist(ax=ax1)
        ax1.set_title("Age distribution")
        st.pyplot(fig1)

        if clf is not None and hasattr(clf, "feature_importances_"):
            fig2, ax2 = plt.subplots()
            feat_names = ['age','years_experience','education','salary']
            ax2.bar(feat_names, clf.feature_importances_)
            ax2.set_title("Feature importances (from RandomForest)")
            st.pyplot(fig2)
        else:
            st.info("Train the model to see feature importances.")
    elif use_plotly:
        fig1 = px.histogram(df, x='age', nbins=20, title="Age distribution")
        st.plotly_chart(fig1, use_container_width=True)
        if clf is not None and hasattr(clf, "feature_importances_"):
            feat_names = ['age','years_experience','education','salary']
            fig2 = px.bar(x=feat_names, y=clf.feature_importances_, labels={'x':'feature','y':'importance'}, title="Feature importances")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Train the model to see feature importances.")
    else:
        st.error("Neither matplotlib nor plotly are available in this environment.\n\n"
                 "matplotlib error: " + (mpl_err or "None") + "\n\n"
                 "plotly error: " + (plotly_err or "None") + "\n\n"
                 "To fix: install one of these packages:\n\n"
                 "pip install matplotlib\nor\npip install plotly")






















import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
from pathlib import Path

st.set_page_config(page_title="sovanta DataSci Demo", layout="wide")

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC = str(PROJECT_ROOT / "src")
if SRC not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

import src.rag as ragmod

st.title("sovanta — (Junior) Data Scientist Demo (Streamlit)")
st.sidebar.title("Controls")

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data
def load_synthetic():
    p = PROJECT_ROOT / "data" / "synthetic_customers.csv"
    return pd.read_csv(p)

@st.cache_data
def load_sap_ingested():
    p = PROJECT_ROOT / "data" / "sap_ingested_mock.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

def run_sap_mock():
    try:
        import src.sap_btp_mock as sap
        sap.main()
        return True, "SAP mock ingestion completed."
    except Exception as e:
        return False, str(e)

def train_model():
    try:
        import src.train_model as tm
        tm.main()
        return True, "Training finished."
    except Exception as e:
        return False, str(e)

def load_model():
    model_path = PROJECT_ROOT / "models" / "model.joblib"
    scaler_path = PROJECT_ROOT / "models" / "scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler

def validate_rag():
    TESTS = [
        {"q": "How does SAP help manage customer data for analytics?", "expected": ["customer data","analytics","centralized"]},
        {"q": "What is the role of SAP BTP in integrating machine learning models?", "expected": ["SAP BTP","machine learning","integration"]},
        {"q": "What are the benefits of using SAP for supply chain optimization?", "expected": ["supply chain","optimization","efficiency"]},
        {"q": "What is the main advantage of using RAG instead of a standard chatbot?", "expected": ["retrieval","documents","context"]},
        {"q": "How does FAISS improve document search in a RAG system?", "expected": ["FAISS","similarity","search"]},
    ]
    docs_path = str(PROJECT_ROOT / "docs")
    sents, embeddings, model, index = ragmod.build_from_docs(docs_path)
    results = []
    for test in TESTS:
        q, exp = test["q"], test["expected"]
        ans = ragmod.answer(q, sents, embeddings, model, index, topk=5)
        matched = sum(1 for kw in exp if kw.lower() in ans.lower())
        score = matched/len(exp)
        results.append({"question": q,"answer": ans,"score": score,"matched_keywords": matched,"expected_keywords": exp})
    return results

# -----------------------------
# Sidebar: Data
# -----------------------------
st.sidebar.header("Data")
if st.sidebar.button("Show synthetic dataset"):
    df = load_synthetic()
    st.subheader("Synthetic dataset (first 10 rows)")
    st.dataframe(df.head(10))

if st.sidebar.button("Run SAP mock ingestion"):
    ok, msg = run_sap_mock()
    if ok:
        st.success(msg)
    else:
        st.error(f"Error: {msg}")

if st.sidebar.button("Show SAP ingested data"):
    df_sap = load_sap_ingested()
    if df_sap is None:
        st.warning("No SAP ingested mock found. Run 'Run SAP mock ingestion' first.")
    else:
        st.subheader("SAP ingested mock (first 10 rows)")
        st.dataframe(df_sap.head(10))

# -----------------------------
# Sidebar: Model
# -----------------------------
st.sidebar.header("Model")
if st.sidebar.button("Train model"):
    with st.spinner("Training model..."):
        ok, msg = train_model()
    if ok:
        st.success(msg)
    else:
        st.error(f"Training failed: {msg}")

clf, scaler = load_model()
if clf is None:
    st.warning("No trained model found. Click 'Train model' to train, or use included model in /models.")
else:
    st.success("Model loaded.")

# -----------------------------
# Prediction
# -----------------------------
st.header("Prediction")
col1, col2, col3, col4 = st.columns(4)
age = int(col1.number_input("Age", min_value=18, max_value=80, value=30))
years_experience = int(col2.number_input("Years of experience", min_value=0, max_value=60, value=5))
education = int(col3.selectbox("Education level", options=[1,2,3], index=1, format_func=lambda x: {1:'HS',2:'BSc',3:'MSc+'}[x]))
salary = float(col4.number_input("Salary (EUR)", min_value=1000, max_value=500000, value=52000))

# Extra features in UI (won't break prediction)
avg_purchase_frequency = float(st.number_input("Avg purchase frequency", min_value=0.0, max_value=100.0, value=5.0))
churn_risk = float(st.number_input("Churn risk (%)", min_value=0.0, max_value=100.0, value=10.0))
last_purchase_days = int(st.number_input("Days since last purchase", min_value=0, max_value=365, value=30))
loyalty_member = st.checkbox("Loyalty member", value=True)
total_purchase_amount = float(st.number_input("Total purchase amount (€)", min_value=0.0, max_value=1000000.0, value=5000.0))

if st.button("Predict"):
    if clf is None:
        st.error("No model available. Train model first.")
    else:
        input_df = pd.DataFrame([{
            "age": age,
            "years_experience": years_experience,
            "education": education,
            "salary": salary,
            "avg_purchase_frequency": avg_purchase_frequency,
            "churn_risk": churn_risk,
            "last_purchase_days": last_purchase_days,
            "loyalty_member": int(loyalty_member),
            "total_purchase_amount": total_purchase_amount
        }])

        # Only use the features model was trained on
        trained_features = ['age','years_experience','education','salary']
        Xs = scaler.transform(input_df[trained_features])

        # Prediction
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xs)
            if proba.shape[1] > 1:
                result = float(proba[0,1])
                st.write("Prediction (1 = high-value):", int(clf.predict(Xs)[0]))
                st.write(f"Predicted probability (class=1): {result:.3f}")
            else:
                result = float(proba[0,0])
                st.write("Prediction probability:", result)
        else:
            result = float(clf.predict(Xs)[0])
            st.write("Prediction value:", result)

# -----------------------------
# RAG ask-docs + validation
# -----------------------------
st.header("RAG — Ask the docs (embeddings + FAISS with fallback)")
query = st.text_input("Enter a question about SAP / RAG / ML in business")

colq1, colq2 = st.columns(2)
if colq1.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            sents, embeddings, model, index = ragmod.build_from_docs(str(PROJECT_ROOT / "docs"))
            answer = ragmod.answer(query, sents, embeddings, model, index, topk=5)
            st.subheader("Answer")
            st.text(answer)
        except Exception as e:
            st.error(f"RAG error: {e}")

if colq2.button("Run RAG Validation"):
    with st.spinner("Running validation tests..."):
        try:
            results = validate_rag()
            st.subheader("RAG Validation Results")
            rag_results_df = pd.DataFrame(results)
            st.dataframe(rag_results_df)

            # Download button
            st.download_button(
                label="Export RAG Validation Results",
                data=rag_results_df.to_csv(index=False).encode('utf-8'),
                file_name="rag_validation_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Validation error: {e}")

# -----------------------------
# EDA & Explainability
# -----------------------------
st.header("EDA & Explainability")
df = load_synthetic()
with st.expander("Show dataset summary"):
    st.write(df.describe())
    st.write("Value counts for target:")
    st.write(df['target_highvalue'].value_counts())

with st.expander("Show histogram"):
    for col in ['age','years_experience','salary']:
        counts, bins = np.histogram(df[col], bins=20)
        st.write(f"Histogram of {col}")
        for b, c in zip(bins[:-1], counts):
            bar = "█"*int(c/max(counts)*50)
            st.write(f"{b:.1f} - {b+ (bins[1]-bins[0]):.1f}: {bar} ({c})")

with st.expander("Feature importances / coefficients"):
    trained_features = ['age','years_experience','education','salary']
    if clf is not None:
        importances = None
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = clf.coef_[0]

        if importances is not None:
            # Replace NaN with 0
            importances = np.nan_to_num(importances)
            max_val = max(np.abs(importances)) if max(np.abs(importances)) > 0 else 1
            for name, val in zip(trained_features, importances):
                bar_len = int(abs(val)/max_val*50)
                bar = "█"*bar_len
                st.write(f"{name}: {bar} ({val:.3f})")
        else:
            st.info("Model type does not provide feature importances or coefficients.")
    else:
        st.info("Train the model to see feature importances.")



















import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
from pathlib import Path

st.set_page_config(page_title="sovanta DataSci Demo", layout="wide")

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC = str(PROJECT_ROOT / "src")
if SRC not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

import src.rag as ragmod

st.title("sovanta — (Junior) Data Scientist Demo (Streamlit)")
st.sidebar.title("Controls")

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data
def load_synthetic():
    p = PROJECT_ROOT / "data" / "synthetic_customers.csv"
    return pd.read_csv(p)

@st.cache_data
def load_sap_ingested():
    p = PROJECT_ROOT / "data" / "sap_ingested_mock.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

def run_sap_mock():
    try:
        import src.sap_btp_mock as sap
        sap.main()
        return True, "SAP mock ingestion completed."
    except Exception as e:
        return False, str(e)

def train_model():
    try:
        import src.train_model as tm
        tm.main()
        return True, "Training finished."
    except Exception as e:
        return False, str(e)

def load_model():
    model_path = PROJECT_ROOT / "models" / "model.joblib"
    scaler_path = PROJECT_ROOT / "models" / "scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler

def validate_rag():
    TESTS = [
        {"q": "How does SAP help manage customer data for analytics?", "expected": ["customer data","analytics","centralized"]},
        {"q": "What is the role of SAP BTP in integrating machine learning models?", "expected": ["SAP BTP","machine learning","integration"]},
        {"q": "What are the benefits of using SAP for supply chain optimization?", "expected": ["supply chain","optimization","efficiency"]},
        {"q": "What is the main advantage of using RAG instead of a standard chatbot?", "expected": ["retrieval","documents","context"]},
        {"q": "How does FAISS improve document search in a RAG system?", "expected": ["FAISS","similarity","search"]},
    ]
    docs_path = str(PROJECT_ROOT / "docs")
    sents, embeddings, model, index = ragmod.build_from_docs(docs_path)
    results = []
    for test in TESTS:
        q, exp = test["q"], test["expected"]
        ans = ragmod.answer(q, sents, embeddings, model, index, topk=5)
        matched = sum(1 for kw in exp if kw.lower() in ans.lower())
        score = matched/len(exp)
        results.append({"question": q,"answer": ans,"score": score,"matched_keywords": matched,"expected_keywords": exp})
    return results

# -----------------------------
# Sidebar: Data
# -----------------------------
st.sidebar.header("Data")
if st.sidebar.button("Show synthetic dataset"):
    df = load_synthetic()
    st.subheader("Synthetic dataset (first 10 rows)")
    st.dataframe(df.head(10))

if st.sidebar.button("Run SAP mock ingestion"):
    ok, msg = run_sap_mock()
    if ok:
        st.success(msg)
    else:
        st.error(f"Error: {msg}")

if st.sidebar.button("Show SAP ingested data"):
    df_sap = load_sap_ingested()
    if df_sap is None:
        st.warning("No SAP ingested mock found. Run 'Run SAP mock ingestion' first.")
    else:
        st.subheader("SAP ingested mock (first 10 rows)")
        st.dataframe(df_sap.head(10))

# -----------------------------
# Sidebar: Model
# -----------------------------
st.sidebar.header("Model")
if st.sidebar.button("Train model"):
    with st.spinner("Training model..."):
        ok, msg = train_model()
    if ok:
        st.success(msg)
    else:
        st.error(f"Training failed: {msg}")

clf, scaler = load_model()
if clf is None:
    st.warning("No trained model found. Click 'Train model' to train, or use included model in /models.")
else:
    st.success("Model loaded.")

# -----------------------------
# Prediction
# -----------------------------
st.header("Prediction")
col1, col2, col3, col4 = st.columns(4)
age = int(col1.number_input("Age", min_value=18, max_value=80, value=30))
years_experience = int(col2.number_input("Years of experience", min_value=0, max_value=60, value=5))
education = int(col3.selectbox("Education level", options=[1,2,3], index=1, format_func=lambda x: {1:'HS',2:'BSc',3:'MSc+'}[x]))
salary = float(col4.number_input("Salary (EUR)", min_value=1000, max_value=500000, value=52000))

if st.button("Predict"):
    if clf is None:
        st.error("No model available. Train model first.")
    else:
        input_df = pd.DataFrame([{"age": age,"years_experience": years_experience,"education": education,"salary": salary}])
        Xs = scaler.transform(input_df)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xs)
            if proba.shape[1] > 1:
                result = float(proba[0,1])
                st.write("Prediction (1 = high-value):", int(clf.predict(Xs)[0]))
                st.write(f"Predicted probability (class=1): {result:.3f}")
            else:
                result = float(proba[0,0])
                st.write("Prediction probability:", result)
        else:
            result = float(clf.predict(Xs)[0])
            st.write("Prediction value:", result)

# -----------------------------
# RAG ask-docs + validation
# -----------------------------
st.header("RAG — Ask the docs (embeddings + FAISS with fallback)")
query = st.text_input("Enter a question about SAP / RAG / ML in business")

colq1, colq2 = st.columns(2)
if colq1.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            sents, embeddings, model, index = ragmod.build_from_docs(str(PROJECT_ROOT / "docs"))
            answer = ragmod.answer(query, sents, embeddings, model, index, topk=5)
            st.subheader("Answer")
            st.text(answer)
        except Exception as e:
            st.error(f"RAG error: {e}")

if colq2.button("Run RAG Validation"):
    with st.spinner("Running validation tests..."):
        try:
            results = validate_rag()
            st.subheader("RAG Validation Results")
            rag_results_df = pd.DataFrame(results)
            st.dataframe(rag_results_df)
            
            # Download button
            st.download_button(
                label="Download Validation Results",
                data=rag_results_df.to_csv(index=False).encode('utf-8'),
                file_name="rag_validation_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Validation error: {e}")

# -----------------------------
# EDA & Explainability
# -----------------------------
st.header("EDA & Explainability")
df = load_synthetic()
with st.expander("Show dataset summary"):
    st.write(df.describe())
    st.write("Value counts for target:")
    st.write(df['target_highvalue'].value_counts())

with st.expander("Show plots"):
    # Histogram using NumPy + Streamlit
    st.subheader("Age distribution")
    counts, bins = np.histogram(df['age'], bins=20)
    for i in range(len(bins)-1):
        color = "#" + ("%02x%02x%02x" % (255-int(counts[i]*2), 100, 150))  # RGB color based on count
        st.markdown(f"<div style='background-color:{color}; width:{int(counts[i]*5)}px; height:20px;'>{bins[i]:.0f}-{bins[i+1]:.0f}: {counts[i]}</div>", unsafe_allow_html=True)

    # Feature importances / coefficients
    st.subheader("Feature importances / coefficients")
    feat_names = ['age','years_experience','education','salary']
    importances = None
    if clf is not None:
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.ravel(clf.coef_)
    if importances is not None:
        max_val = max(np.abs(importances))
        for name, val in zip(feat_names, importances):
            bar = "█"*int(abs(val)/max_val*50) if max_val > 0 else ""
            st.write(f"{name}: {bar} ({val:.3f})")
    else:
        st.info("Model type does not provide feature importances or coefficients.")

