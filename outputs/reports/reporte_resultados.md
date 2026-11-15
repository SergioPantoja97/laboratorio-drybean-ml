# Reporte de resultados - Dry Bean

## 1. Comprension del negocio

Se busca automatizar la clasificacion de granos secos para apoyar el control de calidad de una empresa agricola. El modelo predice la variedad (`Class`) usando 16 caracteristicas geometricas. Es un problema de clasificacion multiclase.

## 2. Comprension de los datos

- Fuente: UCI Machine Learning Repository, Dry Bean Dataset.
- Registros recibidos: 13,611; columnas: 17 (16 predictoras y `Class`).
- Registros despues de retirar duplicados: 13,543; duplicados encontrados: 68.
- Valores faltantes: 0.
- Clases: 7; distribucion por clase guardada en `distribucion_clases.csv` y `.png`.
- Relacion entre la clase mas frecuente y la menos frecuente: 6.79. Por tanto, las clases no tienen exactamente el mismo numero de ejemplos; F1 macro ayuda a dar el mismo peso a cada clase.
- Tipos, valores unicos, faltantes por columna y estadisticas descriptivas estan en `calidad_datos.csv` y `resumen_descriptivo.csv`.

## 3. Preparacion

Se eliminaron filas duplicadas, se separo `Class` de las 16 variables predictoras y se dividieron los datos en entrenamiento/prueba (80/20), con `random_state=42` y estratificacion por clase. La regresion logistica incluye estandarizacion en un pipeline; Random Forest se entrena sin escalamiento.

## 4. Modelado y evaluacion

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Regresion logistica (baseline) | 0.9192 | 0.9302 |
| Random Forest | 0.9192 | 0.9303 |

El modelo seleccionado por F1 macro es **Random Forest**. Los reportes por clase estan en `reporte_clasificacion_baseline.txt` y `reporte_clasificacion_random_forest.txt`. La matriz de confusion de Random Forest esta en `matriz_confusion_random_forest.png`.

La pareja de clases con mas confusiones mutuas en la prueba fue **DERMASON** y **SIRA**. Una explicacion posible es la similitud o el solapamiento de sus medidas geometricas; esta hipotesis debe contrastarse con los perfiles por clase y no implica causalidad.

Las cinco variables con mayor importancia en Random Forest fueron: Perimeter, ShapeFactor1, Compactness, ShapeFactor3, MinorAxisLength. La importancia describe el uso de variables por este modelo, no demuestra que causen la clase.

## 5. Despliegue local

El modelo final se guardo en `outputs/models/modelo_final_drybean.joblib`, se cargo nuevamente y se utilizo para predecir una muestra reservada. Las predicciones de prueba se guardaron en `predicciones_drybean.csv`.

## 6. Respuestas a las actividades

- **¿Cuantas columnas tiene el dataset y cual es el objetivo?** 17 columnas en total; el objetivo es `Class`, con 16 caracteristicas predictoras.
- **¿Clasificacion o regresion?** Clasificacion multiclase, porque se predicen siete variedades discretas.
- **¿Las clases estan balanceadas?** No perfectamente: la razon entre la clase mas y menos frecuente es 6.79. Accuracy puede ocultar un bajo desempeno en una clase menos frecuente; F1 macro promedia el F1 de todas con igual peso.
- **¿Por que usar un baseline?** Para tener una referencia sencilla y reproducible contra la cual juzgar si la complejidad adicional mejora el resultado.
- **¿Que clases se confunden mas?** DERMASON y SIRA, considerando errores en ambas direcciones. El solapamiento en forma/tamano es una explicacion a investigar, no una conclusion causal.
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
