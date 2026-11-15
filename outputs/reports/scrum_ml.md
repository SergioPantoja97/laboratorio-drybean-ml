# Plan de trabajo Scrum ML — Dry Bean

Este plan adapta el formato de seguimiento Scrum al laboratorio de clasificación de frijoles secos. El backlog, las ramas y los puntos son una propuesta de organización para el equipo; deben confirmarse en la planificación. **“Listo para revisar”** indica que el artefacto técnico ya existe en el repositorio, no que el integrante asignado haya confirmado personalmente la tarea.

## Equipo y responsabilidades

| Rol | Integrante | Responsabilidad en este proyecto |
|---|---|---|
| Product Owner | Sergio Luis Pantoja Lasso | Acordar el valor de negocio y los criterios de aceptación; priorizar el backlog y validar las conclusiones. |
| Scrum Master | Luis Gerardo Restrepo Arredondo | Facilitar la planificación y el seguimiento, mantener actualizado el tablero y apoyar la documentación del equipo. |
| Data Engineer / Analyst | Dilmer Gutierrez | Revisar la fuente, preparar los datos, analizar calidad, distribución de clases y comunicar los hallazgos del EDA. |
| ML Engineer | Andres David Bolañoz Paz | Revisar la partición reproducible y el baseline de regresión logística; apoyar pruebas y serialización. |
| ML Engineer | Eliana Ordoñez Candimance | Revisar Random Forest, métricas por clase, matriz de confusión e interpretación de errores. |

## Resumen del Product Backlog

Los puntos son estimaciones iniciales relativas para conversar en la planificación, no esfuerzo ya registrado. Las ramas son nombres sugeridos; los cambios publicados hasta ahora están en `main`.

| ID | Elemento del backlog | Prioridad | Puntos | Responsable de revisión | Sprint | Rama sugerida | Estado técnico |
|---|---|---:|---:|---|---:|---|---|
| PB-01 | Obtener y cargar el conjunto Dry Bean desde UCI | Alta | 2 | Dilmer Gutierrez | 1 | `feature/data-source` | Listo para revisar |
| PB-02 | Analizar calidad, duplicados, clases y características | Alta | 3 | Dilmer Gutierrez | 1 | `feature/eda` | Listo para revisar |
| PB-03 | Entrenar el baseline de regresión logística | Alta | 3 | Andres David Bolañoz Paz | 2 | `feature/baseline` | Listo para revisar |
| PB-04 | Entrenar Random Forest como alternativa | Alta | 3 | Eliana Ordoñez Candimance | 2 | `feature/random-forest` | Listo para revisar |
| PB-05 | Comparar métricas e interpretar errores por clase | Alta | 3 | Sergio Luis Pantoja Lasso + equipo ML | 2 | `feature/evaluation` | Resultados disponibles; aceptación pendiente |
| PB-06 | Guardar, cargar y probar el modelo final | Alta | 2 | Andres David Bolañoz Paz + Eliana Ordoñez Candimance | 3 | `feature/deployment` | Prueba técnica disponible; revisión pendiente |
| PB-07 | Documentar el proceso y preparar el notebook de exposición | Alta | 3 | Luis Gerardo Restrepo Arredondo + equipo | 3 | `feature/documentation` | README y notebook publicados; revisión pendiente |
| PB-08 | Validar criterios de aceptación y preparar presentación final | Alta | 2 | Sergio Luis Pantoja Lasso + equipo | 3 | `feature/final-review` | Pendiente de validación del equipo |

## Sprint 1 — Comprensión y preparación de datos

**Objetivo:** dejar una fuente trazable, conocer la calidad del conjunto y preparar una partición reproducible.

**Fechas:** por confirmar en la planificación del equipo.

**Historias:** PB-01 y PB-02.

| Tarea | ID | Responsable | Evidencia esperada | Estado |
|---|---|---|---|---|
| Cargar el archivo ARFF oficial de UCI y conservar la fuente original | PB-01 | Dilmer Gutierrez | `data/raw/Dry_Bean_Dataset.arff` | Listo para revisar |
| Revisar tamaño, tipos, estadísticas, faltantes y duplicados | PB-02 | Dilmer Gutierrez | `outputs/reports/calidad_datos.csv` y `resumen_descriptivo.csv` | Listo para revisar |
| Analizar las siete clases y las características geométricas | PB-02 | Dilmer Gutierrez | Notebook y gráficas de EDA en `notebooks/` y `outputs/reports/` | Listo para revisar |
| Retirar duplicados y documentar la preparación | PB-02 | Dilmer Gutierrez | `data/processed/dry_bean_clean.csv` | Listo para revisar |
| Preparar la partición estratificada 80/20 con semilla fija | PB-02 | Dilmer Gutierrez + Andres David Bolañoz Paz | Código reproducible en el notebook y `src/` | Listo para revisar |

### Revisión del Sprint 1

- Registros originales: **13.611**, con **17 columnas** (16 predictoras y `Class`).
- Valores faltantes: **0**; filas duplicadas: **68**.
- Registros después de retirar duplicados: **13.543**.
- La distribución de las variedades no es uniforme; la razón entre la clase más y menos frecuente es aproximadamente **6,79**.
- Evidencia: [notebook de EDA](../../notebooks/01_laboratorio_drybean.ipynb), [reporte de resultados](reporte_resultados.md) y archivos de calidad en `outputs/reports/`.

**Criterio de aceptación pendiente:** Dilmer revisa los hallazgos, confirma las decisiones de preparación y el equipo acuerda que el conjunto está listo para modelar.

## Sprint 2 — Modelado y evaluación

**Objetivo:** comparar un modelo de referencia y uno alternativo con la misma partición de prueba.

**Fechas:** por confirmar en la planificación del equipo.

**Historias:** PB-03, PB-04 y PB-05.

| Tarea | ID | Responsable | Evidencia esperada | Estado |
|---|---|---|---|---|
| Entrenar regresión logística con escalamiento en un pipeline | PB-03 | Andres David Bolañoz Paz | Métricas y reporte por clase del baseline | Listo para revisar |
| Entrenar Random Forest con pesos balanceados | PB-04 | Eliana Ordoñez Candimance | Métricas y reporte por clase de Random Forest | Listo para revisar |
| Comparar Accuracy y F1 macro | PB-05 | Equipo ML | `outputs/reports/metricas.json` | Resultados disponibles |
| Revisar errores y pares de clases confundidas | PB-05 | Eliana Ordoñez Candimance + Dilmer Gutierrez | Matriz de confusión y explicación en el reporte | Resultados disponibles |
| Revisar variables importantes y validar la interpretación de negocio | PB-05 | Sergio Luis Pantoja Lasso + equipo ML | Reporte y notebook de exposición | Aceptación pendiente |

### Resultados técnicos de referencia

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Regresión logística (baseline) | 0,9192 | 0,9302 |
| Random Forest | 0,9192 | 0,9303 |

Random Forest queda como selección provisional por su F1 macro ligeramente superior. La diferencia es pequeña (aproximadamente **0,0001**), por lo que la decisión final debe considerar la revisión por clase y el costo real de los errores. En esta partición, las confusiones mutuas más altas fueron entre **DERMASON** y **SIRA**; la similitud geométrica es una hipótesis para investigar, no una causa demostrada.

**Evidencia:** [métricas](metricas.json), [matriz de confusión](matriz_confusion_random_forest.png), [importancia de variables](importancia_variables.png) y [reporte completo](reporte_resultados.md).

**Criterio de aceptación pendiente:** Andres y Eliana reproducen las métricas; Sergio valida que F1 macro y el análisis de errores respondan al objetivo del equipo.

## Sprint 3 — Despliegue local y entrega

**Objetivo:** dejar el modelo reutilizable y preparar una entrega comprensible y reproducible.

**Fechas:** por confirmar en la planificación del equipo.

**Historias:** PB-06, PB-07 y PB-08.

| Tarea | ID | Responsable | Evidencia esperada | Estado |
|---|---|---|---|---|
| Guardar y volver a cargar el modelo seleccionado | PB-06 | Andres David Bolañoz Paz + Eliana Ordoñez Candimance | `outputs/models/modelo_final_drybean.joblib` | Prueba técnica disponible |
| Probar una predicción sobre una muestra reservada | PB-06 | Equipo ML | Celda de despliegue local en el notebook | Predicción ejecutada; revisión pendiente |
| Documentar instalación, ejecución y resultados | PB-07 | Luis Gerardo Restrepo Arredondo | `README.md` y `reporte_resultados.md` | Publicado; revisión pendiente |
| Preparar guía paso a paso con gráficas para exponer | PB-07 | Equipo; coordinación de Luis Gerardo Restrepo Arredondo | `notebooks/01_laboratorio_drybean.ipynb` | Publicado; revisión pendiente |
| Revisar la Definition of Done y preparar la presentación final | PB-08 | Sergio Luis Pantoja Lasso + equipo | Revisión del equipo y diapositivas | Pendiente |

## Daily Scrum

La tabla siguiente es una plantilla de seguimiento, no un registro de reuniones ya realizadas. Cada integrante debe completar “¿Qué hice?” y “¿Bloqueos?” con información real. La columna “Siguiente acción sugerida” propone un próximo paso según el rol.

| Integrante | ¿Qué hice desde el último Daily? | Siguiente acción sugerida | ¿Bloqueos? |
|---|---|---|---|
| Sergio Luis Pantoja Lasso — Product Owner | Por completar por el integrante | Validar criterios de aceptación, resultados y decisión del modelo | Por confirmar |
| Luis Gerardo Restrepo Arredondo — Scrum Master | Por completar por el integrante | Actualizar tablero, acuerdos, fechas y seguimiento de los sprints | Por confirmar |
| Dilmer Gutierrez — Data Engineer / Analyst | Por completar por el integrante | Revisar EDA, limpieza, balance de clases y trazabilidad de datos | Por confirmar |
| Andres David Bolañoz Paz — ML Engineer | Por completar por el integrante | Reproducir el baseline y verificar la serialización del modelo | Por confirmar |
| Eliana Ordoñez Candimance — ML Engineer | Por completar por el integrante | Revisar Random Forest, métricas por clase y matriz de confusión | Por confirmar |

## Definition of Done

- [x] El flujo del laboratorio se ejecuta sin errores en el entorno utilizado.
- [x] El conjunto oficial se carga y se reportan faltantes y duplicados.
- [x] Se entrenan el baseline y el modelo alternativo.
- [x] Se comparan Accuracy y F1 macro.
- [x] Se genera una matriz de confusión y una explicación de resultados.
- [x] El modelo final se guarda, se carga y se prueba con una predicción local.
- [x] El proyecto contiene instrucciones, reporte y notebook reproducible.
- [ ] El equipo revisa y acepta los resultados y la selección del modelo.
- [ ] Cada integrante completa los registros reales del Daily Scrum.
- [ ] El equipo completa la retrospectiva y valida la presentación final.

## Retrospectiva técnica propuesta

Este borrador se basa en los resultados ejecutados y debe discutirse y ajustarse en la retrospectiva real del equipo.

| Pregunta | Borrador para conversar |
|---|---|
| ¿Qué salió bien? | La descarga y carga del conjunto se reproducen; se documentaron calidad y duplicados; ambos modelos y sus artefactos están disponibles. |
| ¿Qué fue difícil? | Las clases tienen frecuencias distintas y el modelo alternativo mejora muy poco el F1 macro frente al baseline. |
| ¿Qué mejoraríamos? | Validar estabilidad con validación cruzada estratificada, inspeccionar ejemplos DERMASON/SIRA y acordar costos de error con el Product Owner. |
| ¿Qué aprendizaje técnico obtuvimos? | Accuracy y F1 macro responden preguntas distintas; la matriz de confusión y las métricas por clase ayudan a explicar errores que una cifra global oculta. |

## Flujo de colaboración recomendado

El repositorio está publicado en GitHub. Para futuros cambios, el equipo puede crear ramas desde `main`, usar los nombres sugeridos en el backlog y revisar los cambios antes de integrarlos. Las ramas listadas arriba son propuestas y no indican que ya existan.

```bash
git checkout main
git pull origin main
git checkout -b feature/nombre-de-la-tarea
```

Al finalizar, ejecutar el laboratorio, revisar los reportes y abrir un Pull Request para revisión antes de integrar los cambios.
