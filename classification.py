import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "url", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load and clean data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', names=["label", "message"])
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    df['clean_message'] = df['message'].apply(clean_text)
    return df

# Load dataset
df = load_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["clean_message"], df["label_num"], test_size=0.2, random_state=42)

# Define TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_vec))
    results[name] = {"model": model, "accuracy": acc}

# UI
st.title("📧 Spam Classifier with Multiple Models & Techniques")
user_input = st.text_area("Enter your message")

if st.button("Classify"):
    if user_input:
        clean_input = clean_text(user_input)
        input_vec = vectorizer.transform([clean_input])

        st.subheader("🔍 Predictions by Model")
        for name, info in results.items():
            pred = info["model"].predict(input_vec)[0]
            label = "Spam" if pred else "Not Spam"
            st.write(f"**{name}** ➜ `{label}` | Accuracy: `{info['accuracy'] * 100:.2f}%`")

        st.subheader("📊 Advanced Classification Types")

        # Use Naive Bayes for these examples
        pred_nb = results["Naive Bayes"]["model"].predict(input_vec)[0]

        # Binary
        st.markdown("**Binary Classification:**")
        st.write("Spam" if pred_nb else "Not Spam")

        # Multiclass
        st.markdown("**Multiclass (Simulated):**")
        multiclass_labels = ["Promotional", "Urgent", "Informational"]
        st.write("Class:", multiclass_labels[pred_nb % 3])

        # Multilabel
        st.markdown("**Multilabel (Simulated):**")
        multilabels = ["Promo", "Urgent", "Offer"]
        multilabel_flags = [1, 0, 1]
        multilabel_preds = [label for label, flag in zip(multilabels, multilabel_flags) if flag]
        st.write("Tags:", ", ".join(multilabel_preds))

        # Ordinal
        st.markdown("**Ordinal Classification (Simulated):**")
        levels = ["Low", "Medium", "High"]
        st.write("Spam Risk Level:", levels[pred_nb])

        # Imbalanced Handling
        st.markdown("**Imbalanced Classification (Resampled Data):**")
        spam = df[df.label_num == 1]
        ham = df[df.label_num == 0]
        oversampled_spam = resample(spam, replace=True, n_samples=len(ham), random_state=42)
        balanced_df = pd.concat([ham, oversampled_spam])
        X_bal = balanced_df["clean_message"]
        y_bal = balanced_df["label_num"]
        X_bal_vec = vectorizer.transform(X_bal)
        rf_balanced = RandomForestClassifier().fit(X_bal_vec, y_bal)
        balanced_pred = rf_balanced.predict(input_vec)[0]
        st.write("Resampled Prediction:", "Spam" if balanced_pred else "Not Spam")

        # Hierarchical
        st.markdown("**Hierarchical Classification (Simulated):**")
        hierarchy = {"Promo": ["Discount", "Offer"], "Info": ["Help", "Details"]}
        parent = list(hierarchy.keys())[pred_nb % 2]
        child = hierarchy[parent][pred_nb % 2]
        st.write(f"Category: **{parent}** → **{child}**")
