# Evidencia Scrum ML - tablero de arranque

El proyecto y el analisis tecnico se prepararon en el espacio de trabajo. Los responsables del equipo deben revisar y aceptar sus entregables; no se atribuyen actividades a integrantes sin confirmacion.

## Roles y acuerdos

| Rol | Integrante | Compromiso |
|---|---|---|
| Product Owner | Sergio Luis Pantoja Lasso | Definir el valor esperado, priorizar y aceptar los resultados |
| Scrum Master | Luis Gerardo Restrepo Arredondo | Facilitar seguimiento, tablero, reuniones y retrospectiva |
| Data Engineer / Analyst | Dilmer Gutierrez | Obtener y preparar datos, revisar calidad y documentar el EDA |
| ML Engineer | Andres David Bolañoz Paz | Desarrollar el baseline y registrar experimentos reproducibles |
| ML Engineer | Eliana Ordoñez Candimance | Desarrollar Random Forest, evaluar resultados y analizar errores |

## Definition of Done

Una tarea se considera terminada cuando su criterio de aceptacion se verifica, los cambios y resultados quedan documentados y reproducibles, y los artefactos estan en la estructura TDSP del proyecto. Para cerrar el laboratorio, el codigo debe ejecutarse sin errores, incluir ambos modelos, comparar Accuracy y F1 macro, mostrar matriz de confusion, guardar/cargar el modelo y explicar resultados junto con CRISP-DM, TDSP y Scrum ML.

## Product backlog inicial

| ID | Historia / tarea | Prioridad | Responsable | Criterio de aceptacion | Estado inicial |
|---|---|---|---|---|---|
| PB-01 | Como analista, quiero cargar el Dry Bean real para analizarlo | Alta | Dilmer Gutierrez | Fuente UCI documentada y datos cargados sin errores | Hecho (borrador tecnico) |
| PB-02 | Como equipo, quiero conocer la calidad y distribucion de clases | Alta | Dilmer Gutierrez | Reportar dimensiones, nulos, duplicados y conteos por clase | Hecho (borrador tecnico) |
| PB-03 | Como equipo, quiero un baseline reproducible | Alta | Andres David Bolañoz Paz | Regresion logistica, particion estratificada y Accuracy/F1 macro | Hecho (borrador tecnico) |
| PB-04 | Como equipo, quiero comparar un modelo alternativo | Media | Eliana Ordoñez Candimance | Random Forest evaluado con la misma particion y metricas | Hecho (borrador tecnico) |
| PB-05 | Como Product Owner, quiero entender los aciertos y errores | Alta | Sergio Luis Pantoja Lasso + equipo ML | Comparacion, matriz de confusion e interpretacion revisadas | En progreso (revision pendiente) |
| PB-06 | Como equipo, quiero reutilizar el mejor modelo | Alta | Andres David Bolañoz Paz + Eliana Ordoñez Candimance | Modelo guardado y cargado para una prediccion | Hecho (borrador tecnico) |
| PB-07 | Como equipo tecnico, quiero un proyecto reproducible y documentado | Alta | Luis Gerardo Restrepo Arredondo | Estructura TDSP, instrucciones, reporte y evidencia Scrum presentes | En progreso (GitHub y presentacion pendientes) |
| PB-08 | Como equipo, queremos aceptar y presentar el resultado final | Alta | Sergio Luis Pantoja Lasso | DoD verificada y presentacion preparada | Por hacer |

## Tablero inicial

| Por hacer | En progreso | Hecho |
|---|---|---|
| PB-05: revision del Product Owner; PB-07: GitHub/presentacion; PB-08: aceptacion final | Validacion de PB-01 a PB-04 y PB-06 por el equipo | Borradores tecnicos de PB-01 a PB-04 y PB-06 preparados |

## Plan de sprints

### Sprint 1 - Comprension y preparacion

- **Objetivo:** tener una fuente de datos trazable, conocer su calidad y preparar una particion reproducible.
- **Tareas:** PB-01, PB-02 y preparacion de datos de PB-07.
- **Responsables:** Dilmer obtiene y analiza los datos; Luis facilita el tablero; Sergio valida el objetivo de negocio.
- **Entregables:** dataset limpio, analisis exploratorio, particion train/test y estructura del proyecto.

### Sprint 2 - Modelado y evaluacion

- **Objetivo:** comparar un baseline y un modelo alternativo con el mismo conjunto de prueba.
- **Tareas:** PB-03, PB-04 y PB-05.
- **Responsables:** Andres implementa el baseline; Eliana implementa Random Forest; ambos revisan metricas y errores; Sergio valida que las metricas respondan al objetivo.
- **Entregables:** comparacion Accuracy/F1 macro, reportes por clase, matriz de confusion e interpretacion.

### Sprint 3 - Despliegue y documentacion

- **Objetivo:** dejar el mejor modelo utilizable y entregar evidencia reproducible.
- **Tareas:** PB-06, completar PB-07 y PB-08.
- **Responsables:** Andres y Eliana validan serializacion y prediccion; Luis consolida documentacion y Scrum; Sergio acepta el resultado y la presentacion.
- **Entregables:** modelo joblib, reporte, README profesional y presentacion final.

## Daily Scrum

Completar con las reuniones reales del equipo; no se registran avances ni bloqueos que no hayan sido reportados.

| Integrante | ¿Que hice desde la ultima reunion? | ¿Que hare ahora? | ¿Bloqueos? |
|---|---|---|---|
| Sergio Luis Pantoja Lasso | Pendiente | Pendiente | Pendiente |
| Luis Gerardo Restrepo Arredondo | Pendiente | Pendiente | Pendiente |
| Dilmer Gutierrez | Pendiente | Pendiente | Pendiente |
| Andres David Bolañoz Paz | Pendiente | Pendiente | Pendiente |
| Eliana Ordoñez Candimance | Pendiente | Pendiente | Pendiente |

## Sprint review y retrospectiva

Completar al final de cada sprint, basandose en lo que efectivamente ocurrio.

| Pregunta | Respuesta del equipo |
|---|---|
| ¿Que entregables se completaron y quien los valido? | Analisis y artefactos tecnicos generados; validacion del equipo pendiente |
| ¿Que salio bien? | Pendiente |
| ¿Que fue dificil o no funciono? | Pendiente |
| ¿Que mejorariamos en el siguiente sprint? | Pendiente |
| ¿Que aprendizaje tecnico obtuvimos? | Pendiente |
