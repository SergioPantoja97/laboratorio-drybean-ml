# Clasificacion de frijoles secos

Proyecto del laboratorio de Machine Learning con el conjunto Dry Bean de UCI. Integra CRISP-DM para organizar el analisis, TDSP para mantener un proyecto reproducible y Scrum ML para planear el trabajo del equipo.

## Equipo

| Rol | Integrante | Responsabilidad principal |
|---|---|---|
| Product Owner | Sergio Luis Pantoja Lasso | Definir el objetivo, priorizar el backlog y validar las conclusiones |
| Scrum Master | Luis Gerardo Restrepo Arredondo | Coordinar los sprints, mantener el tablero y facilitar la retrospectiva |
| Data Engineer / Analyst | Dilmer Gutierrez | Obtener, revisar, limpiar y describir los datos |
| ML Engineer | Andres David Bolañoz Paz | Implementar el baseline y apoyar la reproducibilidad |
| ML Engineer | Eliana Ordoñez Candimance | Entrenar el modelo alternativo, comparar metricas y evaluar errores |

## Estructura

```text
laboratorio_drybean_ml/
├── data/
│   ├── raw/                 # ARFF original de UCI
│   └── processed/           # CSV limpio usado por el analisis
├── notebooks/               # Reservado para notebooks opcionales
├── outputs/
│   ├── models/              # Modelo serializado con joblib
│   └── reports/             # Metricas, graficas y evidencia Scrum ML
├── src/
│   └── laboratorio_drybean.py
├── requirements.txt
└── README.md
```

## Ejecucion en Windows PowerShell

Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\laboratorio_drybean.py
```

La primera ejecucion descarga el archivo ARFF desde el repositorio oficial UCI. Si ya existe `data/raw/Dry_Bean_Dataset.arff`, se reutiliza para poder repetir el analisis sin descargarlo de nuevo.

## Resultados generados

El programa registra las dimensiones y la calidad de los datos, genera estadisticas descriptivas, elimina filas duplicadas, hace una particion estratificada 80/20 con semilla 42, compara regresion logistica escalada con Random Forest y selecciona el modelo final por F1 macro. Guarda el modelo, metricas, reportes por clase, matriz de confusion, importancia de variables, distribucion de clases y predicciones.

Los resultados y las respuestas del laboratorio se escriben en `outputs/reports/reporte_resultados.md`. El plan del equipo, backlog, tablero y plantillas Scrum estan en `outputs/reports/scrum_ml.md`. La retrospectiva y la revision de sprint deben completarse con las observaciones reales del equipo; no se presuponen.

## Metodologias

- **CRISP-DM:** comprension del negocio y de los datos, preparacion, modelado, evaluacion y una prueba local de despliegue.
- **TDSP:** estructura estable, datos y artefactos separados, semilla fija, metricas guardadas y modelo serializado.
- **Scrum ML:** backlog priorizado y tres sprints con entregables incrementales. La evidencia y el estado inicial del tablero estan en el reporte Scrum.

## Fuente de datos

UCI Machine Learning Repository, [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). El objetivo es `Class`; el dataset contiene 13.611 registros, 16 variables predictoras y siete clases.
