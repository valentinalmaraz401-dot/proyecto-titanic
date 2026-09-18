# Análisis de Pasajeros del Titanic — ED.02

## Datos del Dataset
* **Nombre del dataset:** Titanic - Machine Learning from Disaster (`train.csv`)
* **Fuente:** Kaggle Titanic Competition
* **Descripción breve:** Evaluación de variables socio-demográficas, clase social y características de embarque de 891 pasajeros para determinar patrones de supervivencia.

## Objetivo
Analizar la relación entre el género, la clase del pasajero, el rango de edad y la condición de acompañamiento familiar con la tasa de supervivencia general.

## Requisitos
Se requiere **Python 3.8+** y las dependencias incluidas en el archivo `requirements.txt` (`pandas`, `matplotlib`, `seaborn`).

## Estructura del Proyecto
```text
proyecto-titanic/
├── data/
│   └── train.csv
├── outputs/
│   └── resultados/
│       ├── supervivencia_acompanamiento.png
│       ├── supervivencia_genero_clase.png
│       └── supervivencia_grupo_edad.png
├── src/
│   └── analysis.py
├── .gitignore
├── README.md
└── requirements.txt
