# Clasificación de variedades de frijol seco

Proyecto académico de Machine Learning para clasificar granos de frijol a partir de sus características geométricas. El trabajo integra **CRISP-DM**, **TDSP** y **Scrum ML** para documentar el proceso completo: desde la comprensión del problema hasta la evaluación y prueba local del modelo.

> **Objetivo:** apoyar el control de calidad agrícola mediante la predicción reproducible de la variedad de un grano de frijol seco.

| Información | Detalle |
|---|---|
| Conjunto de datos | [Dry Bean Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| Tipo de problema | Clasificación multiclase |
| Variable objetivo | `Class` |
| Clases | 7 variedades |
| Características predictoras | 16 medidas geométricas |
| Lenguaje y herramientas | Python, pandas, scikit-learn y joblib |
| Gestión del proyecto | CRISP-DM, TDSP y Scrum ML |

## Equipo y responsabilidades

El equipo combina responsabilidades de producto, facilitación, preparación de datos y modelado. Las asignaciones describen el alcance de cada rol; las tareas y revisiones se registran en la [evidencia Scrum ML](outputs/reports/scrum_ml.md).

| Rol | Integrante | Responsabilidades |
|---|---|---|
| **Product Owner** | Sergio Luis Pantoja Lasso | Define el objetivo de negocio, prioriza el Product Backlog, revisa si los resultados responden a la necesidad del usuario y valida la entrega final. |
| **Scrum Master** | Luis Gerardo Restrepo Arredondo | Facilita la coordinación del equipo, da seguimiento a los sprints, mantiene visible el tablero e impulsa la resolución de bloqueos y la retrospectiva. |
| **Data Engineer / Analyst** | Dilmer Gutierrez | Obtiene y documenta los datos, revisa su estructura y calidad, analiza las clases, prepara los datos y comparte hallazgos del análisis exploratorio. |
| **ML Engineer** | Andres David Bolañoz Paz | Implementa y evalúa el modelo baseline, mantiene la reproducibilidad de los experimentos y apoya la serialización del modelo. |
| **ML Engineer** | Eliana Ordoñez Candimance | Implementa y evalúa el modelo alternativo, compara métricas, examina errores por clase y apoya la selección del modelo final. |

## Resultados de la ejecución de referencia

Se evaluaron los modelos con una partición estratificada de entrenamiento y prueba de 80/20 y `random_state=42`. Después de revisar y retirar duplicados, el conjunto quedó con **13.543 filas** y **17 columnas**. No se encontraron valores faltantes.

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Regresión logística (baseline) | 0,9192 | 0,9302 |
| Random Forest | 0,9192 | 0,9303 |

Random Forest obtuvo el F1 macro ligeramente más alto y se guardó como modelo final. La diferencia es pequeña; por tanto, el equipo debe considerar esta elección provisional y revisarla según los costos de error y los resultados por clase.

El reporte generado contiene la interpretación de las métricas, las respuestas a las actividades del laboratorio y las reflexiones de CRISP-DM y TDSP:

- [Reporte de resultados](outputs/reports/reporte_resultados.md)
- [Backlog, plan de sprints y evidencia Scrum ML](outputs/reports/scrum_ml.md)

## Inicio rápido

### Requisitos

- Python 3.10 o posterior.
- Acceso a Internet durante la primera ejecución para descargar el conjunto desde UCI.
- Windows PowerShell, Terminal de Windows o una terminal compatible.

### Instalación y ejecución en Windows PowerShell

Abre la terminal en la carpeta raíz del repositorio y ejecuta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python .\src\laboratorio_drybean.py
```

La primera ejecución descarga el archivo ARFF oficial y lo guarda en `data/raw/`. En ejecuciones posteriores se reutiliza el archivo local. El programa limpia los duplicados, entrena ambos modelos y actualiza los reportes y artefactos en `outputs/`.

Si PowerShell impide activar el entorno virtual, también puedes ejecutar el script directamente usando el intérprete de Python instalado, después de instalar las dependencias.

### Ejecución en macOS o Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/laboratorio_drybean.py
```

## ¿Qué genera el laboratorio?

| Ruta | Contenido |
|---|---|
| `data/raw/Dry_Bean_Dataset.arff` | Datos originales obtenidos de UCI. |
| `data/processed/dry_bean_clean.csv` | Datos preparados después de eliminar filas duplicadas. |
| `outputs/models/modelo_final_drybean.joblib` | Modelo seleccionado y serializado para reutilización. |
| `outputs/reports/reporte_resultados.md` | Resultados, interpretación y respuestas de las actividades. |
| `outputs/reports/scrum_ml.md` | Roles, Product Backlog, tablero, sprints y plantillas de seguimiento. |
| `outputs/reports/metricas.json` | Accuracy y F1 macro de cada modelo. |
| `outputs/reports/reporte_clasificacion_*.txt` | Precisión, recall y F1 por clase. |
| `outputs/reports/matriz_confusion.csv` y `.png` | Errores y aciertos por clase. |
| `outputs/reports/importancia_variables.csv` y `.png` | Importancia estimada de características para Random Forest. |
| `outputs/reports/predicciones_drybean.csv` | Etiquetas reales y predicciones para el conjunto de prueba. |
| `outputs/reports/calidad_datos.csv` y `resumen_descriptivo.csv` | Tipos, valores faltantes, valores únicos y estadísticas descriptivas. |

Los archivos de resultados se regeneran al ejecutar el script. Revisa las fechas de modificación antes de comparar experimentos.

## Estructura del proyecto

```text
laboratorio_drybean_ml/
├── data/
│   ├── raw/                  # Fuente original
│   └── processed/            # Datos preparados
├── notebooks/                # Espacio para exploración opcional
├── outputs/
│   ├── models/               # Modelos serializados
│   └── reports/              # Métricas, gráficas y documentación
├── src/
│   └── laboratorio_drybean.py
├── requirements.txt
└── README.md
```

## Metodología

### CRISP-DM

1. **Comprensión del negocio:** formular la clasificación como apoyo al control de calidad agrícola.
2. **Comprensión de los datos:** explorar las 16 características, las siete clases, los tipos y la calidad de los datos.
3. **Preparación:** retirar duplicados, separar predictores y objetivo y crear una partición estratificada.
4. **Modelado:** comparar regresión logística como baseline y Random Forest como alternativa.
5. **Evaluación:** medir Accuracy, F1 macro y métricas por clase; revisar la matriz de confusión.
6. **Despliegue:** guardar y cargar el modelo para demostrar una predicción local reproducible.

### TDSP

La organización separa datos originales y procesados, código fuente, modelos y reportes. La semilla fija y los artefactos guardados permiten repetir y revisar el experimento.

### Scrum ML

El trabajo se organiza en tres incrementos:

1. **Comprensión y preparación:** fuente, calidad de datos, análisis exploratorio y partición.
2. **Modelado y evaluación:** baseline, modelo alternativo y comparación de métricas.
3. **Despliegue y documentación:** serialización, reporte, revisión final y presentación.

El estado y la planificación detallada están en [scrum_ml.md](outputs/reports/scrum_ml.md). Completen las reuniones, revisiones y retrospectivas con lo que efectivamente ocurra; no se deben presentar como realizadas de antemano.

## Decisiones y consideraciones técnicas

- La clase se conserva en `Class`; las otras 16 columnas se usan como predictores.
- La partición de prueba conserva la proporción de las clases (`stratify=y`).
- La regresión logística usa escalamiento mediante un pipeline; Random Forest no requiere escalamiento.
- Se prioriza F1 macro para que cada clase contribuya por igual al promedio, y también se reporta Accuracy y desempeño por clase.
- La importancia de variables describe el comportamiento del modelo; no demuestra causalidad.
- Una evaluación académica en una partición no garantiza el mismo desempeño con datos de operación. Antes de desplegarlo, validar representatividad, costos de error, calidad de etiquetas y monitoreo.

## Reproducibilidad y colaboración

Para obtener una copia y trabajar en una rama propia:

```bash
git clone https://github.com/SergioPantoja97/laboratorio-drybean-ml.git
cd laboratorio-drybean-ml
git checkout -b feature/nombre-de-la-tarea
```

Antes de proponer cambios, ejecuta el laboratorio desde cero y revisa que los reportes estén actualizados. Coordina los cambios de código y resultados con el equipo para evitar sobrescribir artefactos generados por otra persona.

## Fuente y atribución de datos

Los datos provienen del [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), publicado por el **UCI Machine Learning Repository**. El conjunto contiene 13.611 observaciones, 16 características y siete variedades. Consulta la página de UCI para conocer los detalles y condiciones de uso de los datos.
