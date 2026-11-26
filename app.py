"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# =============================================================================
# STUDENT ANSWERS BLOCK (fill in; keep this in the SAME .py file)
# =============================================================================
# Student ID: ______
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: OpenAI Codex (ChatGPT).
#   - How you used them: Interpreted the requirements, completed the marked
#     starter sections, and ran functional checks and the fixed-sample test.
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   A security analyst reviewing inbound SMS for spam or scam-like messages.
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   In a local run of this app, accuracy on the fixed 100-message sample was
#   99.00% (99/100 correct).
#   Accuracy alone is insufficient for business reliance because this small,
#   fixed sample does not establish performance on new real-world messages.
#   False positives can block genuine bank alerts or one-time codes, while
#   false negatives can expose users to scams and financial loss, so both
#   error types need separate evaluation and human review.
#
# =============================================================================

import streamlit as st
from datasets import load_dataset
from transformers import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = ""  # Fill in your real student ID before submission.
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    """Return (display_label, confidence) for one SMS string."""
    result = classifier(text)[0]
    display_label = map_label(result["label"])
    score = float(result["score"])
    return display_label, score


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    if len(pred_labels) != len(true_labels) or not true_labels:
        raise ValueError("Predictions and labels must have the same nonzero length.")
    matches = sum(pred == true for pred, true in zip(pred_labels, true_labels))
    accuracy = matches / len(true_labels)
    return accuracy


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    st.write(f"Student ID: {STUDENT_ID or '______'}")
    st.write(f"Course secret: {COURSE_SECRET}")

    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    hf_token = None
    try:
        if "HF_TOKEN" in st.secrets:
            hf_token = st.secrets["HF_TOKEN"]
    except FileNotFoundError:
        pass  # Local runs may have no secrets file; Cloud must set HF_TOKEN.

    with st.spinner("Loading the SMS classification model..."):
        classifier = load_model(hf_token)

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        if alert_text.strip():
            label, confidence = classify_text(classifier, alert_text.strip())
            st.write(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2%}")
        else:
            st.warning("Please enter an SMS message before classifying.")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        with st.spinner("Classifying the fixed 100-message sample..."):
            texts, true_labels = load_test_samples()
            pred_labels = []
            rows = []
            for number, (text, truth) in enumerate(zip(texts, true_labels), start=1):
                prediction, confidence = classify_text(classifier, text)
                pred_labels.append(prediction)
                rows.append({
                    "#": number,
                    "Text (short)": text[:120] + ("..." if len(text) > 120 else ""),
                    "Ground truth": truth,
                    "Prediction": prediction,
                    "Confidence": confidence,
                    "Match": prediction == truth,
                })
            accuracy = compute_accuracy(pred_labels, true_labels)
        st.metric("Accuracy on 100 samples", f"{accuracy:.2%}")
        st.write(f"Correct predictions: {sum(row['Match'] for row in rows)}/{len(rows)}")
        st.caption("Fixed homework sample only; this is not a real-world performance claim.")
        st.dataframe(
            rows,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.NumberColumn(format="percent"),
            },
        )


if __name__ == "__main__":
    main()
