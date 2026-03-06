# Deep Learning for Breathing Anomaly Detection

## Overview

Sleep apnea is a sleep disorder where breathing repeatedly stops or becomes shallow during sleep. These interruptions are generally categorized as **apnea** (complete pause in airflow) and **hypopnea** (partial reduction in airflow). Detecting such events accurately is important for sleep disorder diagnosis and clinical assessment.

This project explores a **multi-modal deep learning approach** for detecting apnea and hypopnea events using physiological signals recorded during sleep studies. Instead of relying on a single signal, the model combines information from multiple modalities including **nasal airflow, thoracic movement, and blood oxygen saturation (SpO₂)**.

The repository contains the complete workflow: loading raw physiological recordings, preparing the dataset, training the model, evaluating its performance, and generating visualizations.

---

## Project Structure

```
Multi-Modal-Deep-Learning-for-Apnea-and-Hypopnea-Detection/
│
├── data/
│   ├── AP01/
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
│
├── scripts/
│   ├── create_dataset.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── vis.py
│
├── models/
│   └── cnn_model.py
│
├── outputs/
│   ├── evaluation_report.pdf
│   ├── metrics_summary.csv
│   ├── predictions/
│   └── models/
│
├── visualizations/
│
├── requirements.txt
└── README.md
```

---

## Dataset Description

The dataset contains physiological signals recorded during overnight sleep monitoring sessions. Each subject recording contains multiple signal streams representing breathing activity.

The main signals used are:

* **Nasal Airflow** – measures airflow through the nasal passage
* **Thoracic Movement** – captures chest movement during breathing
* **SpO₂ (Blood Oxygen Saturation)** – indicates oxygen level in the blood
* **Sleep Profile** – contains sleep stage annotations
* **Flow Events** – annotated apnea or hypopnea events

The recordings are organized by subject:

```
data/
 ├── AP01/
 ├── AP02/
 ├── AP03/
 ├── AP04/
 └── AP05/
```

Each folder contains time-series recordings corresponding to the physiological signals listed above.

---

## Approach

The project follows a structured pipeline:

### 1. Data Preparation

Raw signal files are loaded and synchronized. Relevant physiological signals are extracted and aligned with annotated respiratory events.

### 2. Dataset Creation

The breathing signals are segmented into windows. Each window is labeled based on whether it contains normal breathing, apnea, or hypopnea events.

### 3. Model Training

A **convolutional neural network (CNN)** processes the multi-modal breathing signals and learns patterns that correspond to abnormal respiratory activity.

### 4. Evaluation

The trained model is evaluated using cross-validation across multiple subjects. Predictions are compared with the annotated events to compute performance metrics.

---

## Installation

Clone the repository:

```
git clone https://github.com/sriramkv3010/Multi-Modal-Deep-Learning-for-Apnea-and-Hypopnea-Detection.git
cd Multi-Modal-Deep-Learning-for-Apnea-and-Hypopnea-Detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

### Create the dataset

```
python scripts/create_dataset.py
```

### Train the model

```
python scripts/train_model.py
```

### Evaluate the model

```
python scripts/evaluate.py
```

### Generate visualizations

```
python scripts/vis.py
```

---



## Project Pipeline

The overall workflow of the system can be summarized as:

```
Raw Physiological Signals
        │
        │
Signal Preprocessing
        │
        │
Dataset Construction
        │
        │
Deep Learning Model (CNN)
        │
        │
Prediction of Apnea / Hypopnea Events
        │
        │
Evaluation and Visualization
```

---

## Visualizations

The repository also includes visualizations of the physiological signals and detected respiratory events. These plots help illustrate breathing patterns and model predictions.

Example visualizations include:

* Airflow signal patterns during apnea events
* Thoracic movement patterns across breathing cycles
* SpO₂ variation during abnormal respiratory activity

These figures are available in the `visualizations` directory.

---

## Author

Kotipalli Venkata Sriram, B.Tech CSE, IIIT VADODARA

