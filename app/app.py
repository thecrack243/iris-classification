import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import plotly.figure_factory as ff
import io
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from utils import predict_flower, predict_confidence


# CONFIG
st.set_page_config(page_title="Iris ML App", layout="wide")


# LOAD DATA (CACHED)
@st.cache_data
def load_data():
    df = pd.read_csv("../data/iris_dataset.csv")
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


df = load_data()

X = df.drop(columns="Species")
y = df["Species"]

y_numeric = y.map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
})

mapping = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}


X_train, X_test, y_train, y_test = train_test_split(
    X, y_numeric, test_size=0.3, random_state=70
)


# MODEL LOAD
@st.cache_resource
def load_model():
    return joblib.load("iris_model.pkl")


model = load_model()


# MODEL INFO
model_infos = {}
model_infos["Random Forest"] = {
    "model": model,
    "train_time": "Pre-trained",
    "accuracy": accuracy_score(y_test, model.predict(X_test)),
    "regularization": "N/A"
}


# PDF GENERATOR (UNCHANGED LOGIC)
def generate_pdf(report_df, accuracy, precision, recall, f1):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    # Custom style for subtitles
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Heading2"],
        textColor=colors.darkblue,
        spaceAfter=10
    )

    normal_style = styles["Normal"]

    elements = []


    # TITLE

    elements.append(Paragraph("IRIS CLASSIFICATION MODEL REPORT", styles["Title"]))
    elements.append(Spacer(1, 6))

    # Date / Time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {now}", normal_style))

    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 12))


    # METRICS SECTION

    elements.append(Paragraph("MODEL PERFORMANCE METRICS", subtitle_style))
    elements.append(Spacer(1, 8))

    metrics_data = [
        ["Metrics", "Scores"],
        ["Accuracy", f"{accuracy*100:.2f}%"],
        ["Precision", f"{precision*100:.2f}%"],
        ["Recall", f"{recall*100:.2f}%"],
        ["F1 Score", f"{f1*100:.2f}%"]
    ]

    table = Table(metrics_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 15))

    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 12))


    # CLASSIFICATION REPORT

    elements.append(Paragraph("CLASSIFICATION REPORT", subtitle_style))
    elements.append(Spacer(1, 8))

    # clean + format
    report_df = report_df.reset_index()
    report_df = report_df.rename(columns={"index": "class"})
    report_df = report_df.round(3)

    # remove duplicate/empty columns if any
    report_df = report_df.loc[:, ~report_df.columns.duplicated()]

    report_table = [report_df.columns.tolist()] + report_df.values.tolist()

    col_widths = [1.5 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]

    table2 = Table(report_table, colWidths=col_widths)

    table2.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))

    elements.append(table2)
    elements.append(Spacer(1, 15))

    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 12))


    # INTERPRETATION

    elements.append(Paragraph("MODEL INTERPRETATION", subtitle_style))
    elements.append(Spacer(1, 8))

    interpretation_text = """
    • The model performs very well on the Iris dataset.<br/>
    • Petal features are the most important for classification.<br/>
    • Sepal features contribute less to separation.<br/>
    • High accuracy indicates strong class separability in this dataset.
    """

    elements.append(Paragraph(interpretation_text, normal_style))


    # BUILD PDF

    doc.build(elements)

    buffer.seek(0)
    return buffer


# TABS
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Home",
    "Quick Prediction",
    "Manual Test",
    "Data Explorer",
    "Model Insights",
    "About",
    "Feedback"
])


# HOME
with tab0:
    st.markdown("""
    # Iris ML Classification App

    A simple interactive machine learning system trained on the Iris dataset.

    ---
    """)


    # QUICK VALUE SNAPSHOT

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "Iris")
        st.write("150 samples")

    with col2:
        st.metric("Model", "Random Forest")
        st.write("Multiclass classifier")

    with col3:
        st.metric("Classes", "3")
        st.write("Setosa, Versicolor, Virginica")

    st.divider()


    # MAIN ACTION (IMPORTANT PART)

    st.markdown("## Start interacting")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Make a prediction")
        st.write("Go to Quick Prediction to test the model instantly with custom values.")

        st.markdown("### Manual testing")
        st.write("Enter precise measurements for controlled predictions.")

    with col2:
        st.markdown("### Explore dataset")
        st.write("Visualize patterns in 2D and 3D plots.")

        st.markdown("### Model analysis")
        st.write("Check performance metrics and feature importance.")


# QUICK PREDICTION
with tab1:
    st.subheader("Quick Prediction")

    col1, col2 = st.columns(2)

    with col1:
        sl = st.slider("Sepal Length", float(X.iloc[:,0].min()), float(X.iloc[:,0].max()), 5.0)
        sw = st.slider("Sepal Width", float(X.iloc[:,1].min()), float(X.iloc[:,1].max()), 3.0)

    with col2:
        pl = st.slider("Petal Length", float(X.iloc[:,2].min()), float(X.iloc[:,2].max()), 1.5)
        pw = st.slider("Petal Width", float(X.iloc[:,3].min()), float(X.iloc[:,3].max()), 0.2)

    if st.button("Predict"):
        pred = predict_flower(sl, sw, pl, pw)
        conf = predict_confidence(sl, sw, pl, pw)

        st.success(f"Prediction: {pred}")

        if conf > 90:
            st.success(f"Confidence: {conf:.2f}%")
        elif conf > 70:
            st.warning(f"Confidence: {conf:.2f}%")
        else:
            st.error(f"Confidence: {conf:.2f}%")


# MANUAL TEST
with tab2:
    st.subheader("Manual Input Test")

    with st.form("manual_form"):
        sl2 = st.number_input("Sepal Length", 0.0, 10.0, 5.0)
        sw2 = st.number_input("Sepal Width", 0.0, 10.0, 3.0)
        pl2 = st.number_input("Petal Length", 0.0, 10.0, 1.5)
        pw2 = st.number_input("Petal Width", 0.0, 10.0, 0.2)

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        pred = predict_flower(sl2, sw2, pl2, pw2)
        conf = predict_confidence(sl2, sw2, pl2, pw2)

        st.success(f"Predicted Class: {pred}")
        st.info(f"Confidence: {conf:.2f}%")


# DATA EXPLORER
with tab3:
    st.subheader("Dataset Explorer")

    st.dataframe(df)


    # SEPAL VISUALIZATION (2D)

    st.markdown("### Sepal Visualization (2D)")

    fig_sepal = px.scatter(
        df,
        x="SepalLengthCm",
        y="SepalWidthCm",
        color="Species",
        title="Sepal Length vs Width"
    )
    st.plotly_chart(fig_sepal, use_container_width=True)


    # PETAL VISUALIZATION (2D)

    st.markdown("### Petal Visualization (2D)")

    fig_petal = px.scatter(
        df,
        x="PetalLengthCm",
        y="PetalWidthCm",
        color="Species",
        title="Petal Length vs Width"
    )
    st.plotly_chart(fig_petal, use_container_width=True)


    # DATASET VISUALIZATION (3D)

    st.markdown("### Dataset Visualization")

    fig_3d = px.scatter_3d(
        df,
        x="SepalLengthCm",
        y="SepalWidthCm",
        z="PetalLengthCm",
        color="Species",
        title="Iris Dataset 3D Projection"
    )

    fig_3d.update_layout(
        height=750,  # clean vertical space
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="Sepal Length",
            yaxis_title="Sepal Width",
            zaxis_title="Petal Length",
            aspectmode="cube"  # balanced proportions
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)


# MODEL INSIGHTS
with tab4:

    # METRICS DASHBOARD

    st.markdown("### Performance Metrics")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Precision", f"{precision*100:.2f}%")
    col3.metric("Recall", f"{recall*100:.2f}%")
    col4.metric("F1 Score", f"{f1*100:.2f}%")

    st.divider()


    # CONFUSION MATRIX SECTION

    st.markdown("### Confusion Matrix Analysis")

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    # arrondi pour affichage propre
    cm_norm_display = np.round(cm_norm, 2)

    fig = ff.create_annotated_heatmap(
        z=cm_norm_display,
        x=list(mapping.values()),
        y=list(mapping.values()),
        colorscale="Blues",
        showscale=True,
        annotation_text=cm_norm_display  # important
    )

    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        yaxis_autorange="reversed"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()


    # CLASSIFICATION REPORT

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=list(mapping.values()),
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df, use_container_width=True)

    st.divider()


    # FEATURE IMPORTANCE

    if hasattr(model, "feature_importances_"):
        st.markdown("### Feature Importance")

        fig2 = px.bar(
            x=X.columns,
            y=model.feature_importances_
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()


    # INTERPRETATION

    st.markdown("### Model Interpretation")

    st.write("""
    - The model performs very well on the Iris dataset.
    - Petal features are the most important for classification.
    - Sepal features contribute less to separation.
    - High accuracy indicates strong class separability in this dataset.
    """)

    if acc == 1.0:
        st.success("Perfect classification achieved.")

    st.divider()


    # EXPORT

    st.markdown("### Export Report")

    pdf_buffer = generate_pdf(report_df, acc, precision, recall, f1)

    st.download_button(
        "Download PDF Report",
        data=pdf_buffer,
        file_name="iris_model_report.pdf",
        mime="application/pdf"
    )


# ABOUT
with tab5:
    st.markdown("## About This App")

    st.markdown(
        """
        The Iris Flower Classification App is an interactive machine learning project built to demonstrate a complete workflow from data exploration to model evaluation and prediction.

        It is based on the classic Iris dataset and uses a trained Random Forest model for classification.
        
        """
    )

    st.markdown("\n")

    st.markdown("### Technologies")
    st.write("Streamlit, Scikit-learn, Plotly, Pandas, Seaborn")

    st.markdown("\n")

    st.markdown("### Author")
    st.write("Emmanuel Ilunga")

    st.markdown("\n")

    st.markdown("### Project purpose")
    st.write(
        "This project was built as a learning and portfolio application to demonstrate machine learning deployment using an interactive web interface."
    )


# FEEDBACK
with tab6:
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        feedback = st.text_area("Your Feedback")
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.success("Thanks for your feedback!")


# FOOTER
st.divider()
st.markdown("Machine Learning Project — 2026 By Emmanuel Ilugna")