from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "dry_bean_clean.csv"
REPORTS_DIR = ROOT / "outputs" / "reports"
METRICS_PATH = REPORTS_DIR / "metricas.json"
PREDICTIONS_PATH = REPORTS_DIR / "predicciones_drybean.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "importancia_variables.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "modelo_final_drybean.joblib"

st.set_page_config(
    page_title="Dry Bean ML | Exposición",
    page_icon="🫘",
    layout="wide",
)

required_paths = (
    DATA_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    FEATURE_IMPORTANCE_PATH,
    MODEL_PATH,
)
missing_paths = [path.relative_to(ROOT).as_posix() for path in required_paths if not path.exists()]
if missing_paths:
    st.error("Faltan artefactos necesarios para mostrar el dashboard:")
    st.write(missing_paths)
    st.info("Desde la raíz del proyecto, ejecuta `python .\\src\\laboratorio_drybean.py` y vuelve a iniciar la app.")
    st.stop()


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path: str) -> dict[str, dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_model(path: str) -> Any:
    return joblib.load(path)


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Variedad predicha",
        ylabel="Variedad real",
        title="Matriz de confusión — Random Forest",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#1F2937",
            )
    figure.colorbar(image, ax=axis, label="Registros")
    figure.tight_layout()
    return figure


def plot_model_comparison(metrics: dict[str, dict[str, float]]) -> plt.Figure:
    model_names = ["Regresión logística", "Random Forest"]
    keys = ["baseline", "random_forest"]
    accuracy = [metrics[key]["accuracy"] for key in keys]
    f1_macro = [metrics[key]["f1_macro"] for key in keys]
    positions = np.arange(len(model_names))
    width = 0.34

    figure, axis = plt.subplots(figsize=(8, 4.5))
    accuracy_bars = axis.bar(positions - width / 2, accuracy, width, label="Accuracy", color="#4472C4")
    f1_bars = axis.bar(positions + width / 2, f1_macro, width, label="F1 macro", color="#70AD47")
    axis.bar_label(accuracy_bars, fmt="%.4f", padding=3, fontsize=9)
    axis.bar_label(f1_bars, fmt="%.4f", padding=3, fontsize=9)
    axis.set_xticks(positions, model_names)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Puntaje")
    axis.set_title("Comparación en el conjunto de prueba")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


data = load_csv(str(DATA_PATH))
metrics = load_metrics(str(METRICS_PATH))
predictions = load_csv(str(PREDICTIONS_PATH))
feature_importance = load_csv(str(FEATURE_IMPORTANCE_PATH))

feature_columns = [column for column in data.columns if column != "Class"]
required_prediction_columns = {"real_class", "predicted_class"}
if "Class" not in data.columns or not feature_columns:
    st.error("El conjunto preparado debe contener predictores y la columna objetivo `Class`.")
    st.stop()
if not required_prediction_columns.issubset(predictions.columns):
    st.error("El archivo de predicciones no contiene `real_class` y `predicted_class`.")
    st.stop()
if not {"baseline", "random_forest"}.issubset(metrics):
    st.error("El archivo de métricas no contiene los resultados esperados para ambos modelos.")
    st.stop()
if not {"feature", "importance"}.issubset(feature_importance.columns):
    st.error("El archivo de importancia debe incluir las columnas `feature` e `importance`.")
    st.stop()

class_labels = sorted(data["Class"].astype(str).unique())
numeric_features = data[feature_columns].select_dtypes(include="number").columns.tolist()
class_counts = (
    data["Class"].value_counts().rename_axis("Variedad").reset_index(name="Registros")
)

st.title("🫘 Clasificación de variedades de frijol seco")
st.caption(
    "Dashboard interactivo para la exposición: calidad de datos, modelos, errores y demostración local."
)

with st.sidebar:
    st.header("Equipo de trabajo")
    st.markdown(
        """
        - **Product Owner:** Sergio Luis Pantoja Lasso
        - **Scrum Master:** Luis Gerardo Restrepo Arredondo
        - **Data Engineer / Analyst:** Dilmer Gutierrez
        - **ML Engineer:** Andres David Bolañoz Paz
        - **ML Engineer:** Eliana Ordoñez Candimance
        """
    )
    st.divider()
    st.caption("Resultados de referencia con partición estratificada 80/20 y semilla 42.")

tab_summary, tab_data, tab_scrum, tab_models, tab_errors, tab_demo = st.tabs(
    [
        "Resumen · Product Owner",
        "Datos y EDA · Dilmer",
        "Organización · Scrum Master",
        "Modelos · ML Engineers",
        "Errores · Evaluación",
        "Demo de predicción",
    ]
)

with tab_summary:
    baseline = metrics["baseline"]
    forest = metrics["random_forest"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Registros preparados", f"{len(data):,}")
    metric_columns[1].metric("Características", len(feature_columns))
    metric_columns[2].metric("Variedades", len(class_labels))
    metric_columns[3].metric("Valores faltantes", f"{int(data.isna().sum().sum()):,}")

    st.subheader("¿Qué decisión apoya el proyecto?")
    st.write(
        "El objetivo es apoyar la clasificación de variedades a partir de medidas geométricas. "
        "La selección del modelo es provisional: el Product Owner debe validar qué tipos de error "
        "son más importantes antes de considerar una decisión final."
    )

    model_columns = st.columns(2)
    model_columns[0].metric("Accuracy · Regresión logística", f"{baseline['accuracy']:.2%}")
    model_columns[1].metric("Accuracy · Random Forest", f"{forest['accuracy']:.2%}")
    model_columns = st.columns(2)
    model_columns[0].metric("F1 macro · Regresión logística", f"{baseline['f1_macro']:.4f}")
    model_columns[1].metric(
        "F1 macro · Random Forest",
        f"{forest['f1_macro']:.4f}",
        delta=f"{forest['f1_macro'] - baseline['f1_macro']:+.4f} vs. baseline",
    )

    figure = plot_model_comparison(metrics)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.info(
        "La diferencia de F1 macro es pequeña. El gráfico resume una sola partición de prueba "
        "y no demuestra que Random Forest sea superior de forma concluyente."
    )

with tab_data:
    st.subheader("Distribución de las variedades")
    left, right = st.columns(2)
    with left:
        st.bar_chart(class_counts.set_index("Variedad"), y="Registros", height=340)
    with right:
        selected_feature = st.selectbox("Explora la distribución de una medida", numeric_features)
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(data[selected_feature], bins=35, color="#4472C4", edgecolor="white")
        axis.set(title=f"Distribución de {selected_feature}", xlabel=selected_feature, ylabel="Registros")
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)

    st.subheader("Relación entre medidas")
    x_feature, y_feature = st.columns(2)
    x_name = x_feature.selectbox("Eje X", numeric_features, index=numeric_features.index("Area"))
    default_y = numeric_features.index("Perimeter") if "Perimeter" in numeric_features else 1
    y_name = y_feature.selectbox("Eje Y", numeric_features, index=default_y)
    sample_size = min(3_000, len(data))
    scatter_data = data.sample(n=sample_size, random_state=42)
    figure, axis = plt.subplots(figsize=(10, 5))
    for label in class_labels:
        class_sample = scatter_data[scatter_data["Class"] == label]
        axis.scatter(
            class_sample[x_name],
            class_sample[y_name],
            s=13,
            alpha=0.55,
            label=label,
        )
    axis.set(xlabel=x_name, ylabel=y_name, title=f"{x_name} frente a {y_name} (muestra de {sample_size:,})")
    axis.legend(title="Variedad", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axis.grid(alpha=0.15)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    with st.expander("Ver estadísticas descriptivas"):
        st.dataframe(data[numeric_features].describe().T.round(2), width="stretch")

with tab_scrum:
    st.subheader("Aporte del Scrum Master")
    st.write(
        "Luis facilita la coordinación y la visibilidad del trabajo. Esta secuencia es una propuesta "
        "para la exposición; no representa fechas ni reuniones ya registradas."
    )
    sprint_plan = pd.DataFrame(
        [
            {
                "Sprint propuesto": "1 · Datos",
                "Objetivo": "Fuente, calidad, limpieza y EDA",
                "Responsable técnico": "Dilmer",
                "Revisión": "Conjunto preparado y hallazgos reproducibles",
            },
            {
                "Sprint propuesto": "2 · Modelado",
                "Objetivo": "Baseline, alternativa y evaluación",
                "Responsable técnico": "Andres y Eliana",
                "Revisión": "Métricas, errores por clase y decisión por validar",
            },
            {
                "Sprint propuesto": "3 · Entrega",
                "Objetivo": "Modelo serializado, documentación y exposición",
                "Responsable técnico": "Equipo",
                "Revisión": "Definition of Done y pendientes explícitos",
            },
        ]
    )
    st.dataframe(sprint_plan, hide_index=True, width="stretch")
    st.info(
        "Completen fechas, Daily Scrum y retrospectiva con los acuerdos y experiencias reales del equipo."
    )

with tab_models:
    st.subheader("Comparación de modelos")
    metrics_table = pd.DataFrame(
        [
            {"Modelo": "Regresión logística (baseline)", **metrics["baseline"]},
            {"Modelo": "Random Forest", **metrics["random_forest"]},
        ]
    ).set_index("Modelo")
    st.dataframe(
        metrics_table.style.format({"accuracy": "{:.4f}", "f1_macro": "{:.4f}"}),
        width="stretch",
    )

    st.subheader("Variables más importantes en Random Forest")
    top_features = feature_importance.nlargest(10, "importance").sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(top_features["feature"], top_features["importance"], color="#70AD47")
    axis.set(xlabel="Importancia estimada", ylabel="Característica")
    axis.set_title("Diez características principales")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.caption("La importancia describe el uso de variables por el modelo; no prueba causalidad.")

with tab_errors:
    st.subheader("¿En qué variedades se equivoca el modelo?")
    actual = predictions["real_class"].astype(str)
    predicted = predictions["predicted_class"].astype(str)
    evaluation_labels = sorted(set(actual) | set(predicted))
    matrix = confusion_matrix(actual, predicted, labels=evaluation_labels)
    figure = plot_confusion_matrix(matrix, evaluation_labels)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    report = classification_report(
        actual,
        predicted,
        labels=evaluation_labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = (
        pd.DataFrame(report)
        .T.reindex(evaluation_labels)[["precision", "recall", "f1-score", "support"]]
        .rename(
            columns={
                "precision": "Precisión",
                "recall": "Recall",
                "f1-score": "F1",
                "support": "Registros",
            }
        )
    )
    st.subheader("Métricas por variedad")
    st.dataframe(
        per_class.style.format("{:.3f}", subset=["Precisión", "Recall", "F1"]),
        width="stretch",
    )
    st.info(
        "DERMASON y SIRA concentran las confusiones mutuas más altas en esta partición. "
        "La matriz muestra el patrón de errores, pero no explica por sí sola su causa."
    )

with tab_demo:
    st.subheader("Prueba una predicción local")
    st.write(
        "Selecciona una variedad y una observación del conjunto preparado para mostrar cómo se reutiliza "
        "el modelo serializado. Esta es una demostración; no es una nueva evaluación independiente."
    )
    selected_class = st.selectbox("Variedad real de la observación", class_labels)
    class_rows = data[data["Class"].astype(str) == selected_class]
    sample_number = st.slider(
        "Observación de esa variedad",
        min_value=1,
        max_value=len(class_rows),
        value=1,
    )
    sample = class_rows.iloc[[sample_number - 1]]
    sample_features = sample[feature_columns]
    model = load_model(str(MODEL_PATH))
    prediction = str(model.predict(sample_features)[0])

    result_columns = st.columns(3)
    result_columns[0].metric("Variedad real", selected_class)
    result_columns[1].metric("Predicción del modelo", prediction)
    result_columns[2].metric("Resultado", "Coincide" if prediction == selected_class else "No coincide")

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probabilities = model.predict_proba(sample_features)[0]
        probability_table = pd.DataFrame(
            {"Probabilidad": probabilities},
            index=model.classes_,
        ).sort_values("Probabilidad", ascending=False)
        st.bar_chart(probability_table, y="Probabilidad", height=320)

    with st.expander("Ver las medidas usadas en esta predicción"):
        st.dataframe(
            sample_features.T.rename(columns={sample_features.index[0]: "Valor"}),
            width="stretch",
        )
