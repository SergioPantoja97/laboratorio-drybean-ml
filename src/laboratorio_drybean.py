"""Reproducible Dry Bean classification lab."""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "Dry_Bean_Dataset.arff"
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"


def ensure_directories() -> None:
    for relative_path in (
        "data/raw",
        "data/processed",
        "notebooks",
        "outputs/models",
        "outputs/reports",
    ):
        (ROOT / relative_path).mkdir(parents=True, exist_ok=True)


def download_dataset() -> None:
    request = urllib.request.Request(
        UCI_ARCHIVE_URL,
        headers={"User-Agent": "DryBeanMLLab/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = "DryBeanDataset/Dry_Bean_Dataset.arff"
        if member not in archive.namelist():
            raise FileNotFoundError(f"No se encontro {member} en el archivo UCI.")
        RAW_DATA.write_bytes(archive.read(member))


def load_dataset() -> pd.DataFrame:
    if not RAW_DATA.exists():
        print(f"Descargando el dataset desde UCI: {UCI_ARCHIVE_URL}")
        download_dataset()

    with RAW_DATA.open("r", encoding="utf-8") as source:
        records, _ = arff.loadarff(source)

    if "Class" not in records.dtype.names:
        raise ValueError("El dataset no contiene la columna objetivo 'Class'.")
    frame = pd.DataFrame.from_records(records)
    frame["Class"] = frame["Class"].map(
        lambda value: value.decode("utf-8")
        if isinstance(value, bytes)
        else str(value)
    )
    return frame


def save_class_distribution(class_counts: pd.Series) -> None:
    class_counts.rename_axis("Class").reset_index(name="count").to_csv(
        ROOT / "outputs" / "reports" / "distribucion_clases.csv",
        index=False,
    )
    axis = class_counts.plot(kind="bar", figsize=(9, 5), color="#4472C4")
    axis.set_title("Distribucion de clases Dry Bean")
    axis.set_xlabel("Clase de frijol")
    axis.set_ylabel("Cantidad de registros")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ROOT / "outputs" / "reports" / "distribucion_clases.png", dpi=150)
    plt.close()


def save_data_profile(frame: pd.DataFrame) -> None:
    quality = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_values": frame.isna().sum(),
            "unique_values": frame.nunique(),
        }
    )
    quality.to_csv(
        ROOT / "outputs" / "reports" / "calidad_datos.csv",
        index_label="column",
    )
    frame.describe().T.to_csv(
        ROOT / "outputs" / "reports" / "resumen_descriptivo.csv",
        index_label="feature",
    )


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> tuple[dict[str, float], Any]:
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }
    report = classification_report(y_test, predictions, zero_division=0)
    (ROOT / "outputs" / "reports" / f"reporte_clasificacion_{model_name}.txt").write_text(
        report,
        encoding="utf-8",
    )
    return metrics, predictions


def write_results_report(
    *,
    original_rows: int,
    duplicate_rows: int,
    missing_count: int,
    frame: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    winner_name: str,
    confusion: pd.DataFrame,
    top_features: pd.DataFrame,
) -> None:
    class_counts = frame["Class"].value_counts()
    class_ratio = class_counts.max() / class_counts.min()
    confusion_values = confusion.to_numpy().copy()
    for index in range(len(confusion_values)):
        confusion_values[index, index] = 0
    pair_counts = confusion_values + confusion_values.T
    left, right = divmod(int(pair_counts.argmax()), len(confusion))
    confused_classes = confusion.index[left], confusion.columns[right]
    top_feature_names = ", ".join(top_features["feature"].head(5).tolist())

    content = f"""# Reporte de resultados - Dry Bean

## 1. Comprension del negocio

Se busca automatizar la clasificacion de granos secos para apoyar el control de calidad de una empresa agricola. El modelo predice la variedad (`Class`) usando 16 caracteristicas geometricas. Es un problema de clasificacion multiclase.

## 2. Comprension de los datos

- Fuente: UCI Machine Learning Repository, Dry Bean Dataset.
- Registros recibidos: {original_rows:,}; columnas: {frame.shape[1]} (16 predictoras y `Class`).
- Registros despues de retirar duplicados: {len(frame):,}; duplicados encontrados: {duplicate_rows:,}.
- Valores faltantes: {missing_count}.
- Clases: {len(class_counts)}; distribucion por clase guardada en `distribucion_clases.csv` y `.png`.
- Relacion entre la clase mas frecuente y la menos frecuente: {class_ratio:.2f}. Por tanto, las clases no tienen exactamente el mismo numero de ejemplos; F1 macro ayuda a dar el mismo peso a cada clase.
- Tipos, valores unicos, faltantes por columna y estadisticas descriptivas estan en `calidad_datos.csv` y `resumen_descriptivo.csv`.

## 3. Preparacion

Se eliminaron filas duplicadas, se separo `Class` de las 16 variables predictoras y se dividieron los datos en entrenamiento/prueba (80/20), con `random_state=42` y estratificacion por clase. La regresion logistica incluye estandarizacion en un pipeline; Random Forest se entrena sin escalamiento.

## 4. Modelado y evaluacion

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Regresion logistica (baseline) | {metrics['baseline']['accuracy']:.4f} | {metrics['baseline']['f1_macro']:.4f} |
| Random Forest | {metrics['random_forest']['accuracy']:.4f} | {metrics['random_forest']['f1_macro']:.4f} |

El modelo seleccionado por F1 macro es **{winner_name}**. Los reportes por clase estan en `reporte_clasificacion_baseline.txt` y `reporte_clasificacion_random_forest.txt`. La matriz de confusion de Random Forest esta en `matriz_confusion_random_forest.png`.

La pareja de clases con mas confusiones mutuas en la prueba fue **{confused_classes[0]}** y **{confused_classes[1]}**. Una explicacion posible es la similitud o el solapamiento de sus medidas geometricas; esta hipotesis debe contrastarse con los perfiles por clase y no implica causalidad.

Las cinco variables con mayor importancia en Random Forest fueron: {top_feature_names}. La importancia describe el uso de variables por este modelo, no demuestra que causen la clase.

## 5. Despliegue local

El modelo final se guardo en `outputs/models/modelo_final_drybean.joblib`, se cargo nuevamente y se utilizo para predecir una muestra reservada. Las predicciones de prueba se guardaron en `predicciones_drybean.csv`.

## 6. Respuestas a las actividades

- **¿Cuantas columnas tiene el dataset y cual es el objetivo?** {frame.shape[1]} columnas en total; el objetivo es `Class`, con 16 caracteristicas predictoras.
- **¿Clasificacion o regresion?** Clasificacion multiclase, porque se predicen siete variedades discretas.
- **¿Las clases estan balanceadas?** No perfectamente: la razon entre la clase mas y menos frecuente es {class_ratio:.2f}. Accuracy puede ocultar un bajo desempeno en una clase menos frecuente; F1 macro promedia el F1 de todas con igual peso.
- **¿Por que usar un baseline?** Para tener una referencia sencilla y reproducible contra la cual juzgar si la complejidad adicional mejora el resultado.
- **¿Que clases se confunden mas?** {confused_classes[0]} y {confused_classes[1]}, considerando errores en ambas direcciones. El solapamiento en forma/tamano es una explicacion a investigar, no una conclusion causal.
- **¿Que metrica elegir?** F1 macro como criterio principal si el costo de fallar en cualquiera de las variedades es similar y hay desequilibrio; tambien se revisa recall por clase y accuracy global.

## 7. Reflexion final

1. **Fase CRISP-DM clave:** comprension y evaluacion de datos, porque validan que las medidas representan el problema y que el resultado sirve antes de proponer despliegue.
2. **Ventaja de TDSP:** estructura y trazabilidad para reproducir los datos, decisiones, metricas y modelo, en vez de depender de archivos dispersos.
3. **Scrum ML ante incertidumbre:** entrega resultados en incrementos y permite revisar hipotesis y priorizar el siguiente experimento con evidencia.
4. **Baseline:** establece una referencia cuantitativa para valorar modelos mas complejos.
5. **Errores concentrados en una clase:** revisar ejemplos y calidad de etiquetas, analizar metricas por clase y perfiles de variables, probar ajustes de pesos o datos y validar de nuevo con particiones apropiadas.
6. **Antes de produccion:** acordar costos de error y umbrales, validar representatividad y calidad en operacion, definir integracion, latencia, seguridad, monitoreo de deriva y desempeno, y un proceso de mantenimiento/reentrenamiento.

## 8. Scrum ML

El backlog inicial, las tareas por sprint y las plantillas de seguimiento estan en `scrum_ml.md`. La revision de sprint, el daily Scrum y la retrospectiva se deben completar con las decisiones y experiencias reales del equipo.
"""
    (ROOT / "outputs" / "reports" / "reporte_resultados.md").write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    ensure_directories()
    frame = load_dataset()
    original_rows = len(frame)
    missing_count = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    print(f"Dimensiones originales: {frame.shape}")
    print(f"Valores faltantes: {missing_count}")
    print(f"Filas duplicadas: {duplicate_rows}")
    print(frame["Class"].value_counts().to_string())
    save_data_profile(frame)

    frame = frame.drop_duplicates().copy()
    frame.to_csv(ROOT / "data" / "processed" / "dry_bean_clean.csv", index=False)
    X = frame.drop(columns="Class")
    y = frame["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    class_counts = y.value_counts()
    save_class_distribution(class_counts)

    baseline_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    baseline_model.fit(X_train, y_train)
    baseline_metrics, _ = evaluate_model(
        baseline_model, X_test, y_test, "baseline"
    )

    forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    forest_model.fit(X_train, y_train)
    forest_metrics, forest_predictions = evaluate_model(
        forest_model, X_test, y_test, "random_forest"
    )
    metrics = {
        "baseline": baseline_metrics,
        "random_forest": forest_metrics,
    }
    metrics_path = ROOT / "outputs" / "reports" / "metricas.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, forest_predictions, labels=labels)
    confusion = pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="Clase real"),
        columns=pd.Index(labels, name="Clase predicha"),
    )
    confusion.to_csv(ROOT / "outputs" / "reports" / "matriz_confusion.csv")
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Clase predicha",
        ylabel="Clase real",
        title="Matriz de confusion - Random Forest",
    )
    axis.tick_params(axis="x", labelrotation=45)
    figure.colorbar(image, ax=axis)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(
        ROOT / "outputs" / "reports" / "matriz_confusion_random_forest.png",
        dpi=150,
    )
    plt.close(figure)

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": forest_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(
        ROOT / "outputs" / "reports" / "importancia_variables.csv",
        index=False,
    )
    feature_axis = feature_importance.head(10).plot(
        x="feature",
        y="importance",
        kind="bar",
        legend=False,
        figsize=(10, 5),
        color="#70AD47",
    )
    feature_axis.set_title("Top 10 variables mas importantes")
    feature_axis.set_xlabel("Variable")
    feature_axis.set_ylabel("Importancia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        ROOT / "outputs" / "reports" / "importancia_variables.png",
        dpi=150,
    )
    plt.close()

    winner_name = (
        "Random Forest"
        if forest_metrics["f1_macro"] >= baseline_metrics["f1_macro"]
        else "Regresion logistica (baseline)"
    )
    final_model = forest_model if winner_name == "Random Forest" else baseline_model
    model_path = ROOT / "outputs" / "models" / "modelo_final_drybean.joblib"
    joblib.dump(final_model, model_path)
    loaded_model = joblib.load(model_path)
    prediction = loaded_model.predict(X_test.iloc[[0]])[0]
    print(f"Prediccion de prueba: {prediction}; valor real: {y_test.iloc[0]}")

    predictions_frame = X_test.copy()
    predictions_frame["real_class"] = y_test
    predictions_frame["predicted_class"] = forest_predictions
    predictions_frame.to_csv(
        ROOT / "outputs" / "reports" / "predicciones_drybean.csv",
        index=False,
    )
    write_results_report(
        original_rows=original_rows,
        duplicate_rows=duplicate_rows,
        missing_count=missing_count,
        frame=frame,
        metrics=metrics,
        winner_name=winner_name,
        confusion=confusion,
        top_features=feature_importance,
    )
    print(f"Reporte guardado en: {ROOT / 'outputs' / 'reports' / 'reporte_resultados.md'}")


if __name__ == "__main__":
    main()
