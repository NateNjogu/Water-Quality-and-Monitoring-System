# 🇰🇪 Predictive Modeling of Water Quality in Kenyan Streams

## Project Overview
This project is an offline-capable, edge-computing predictive modeling system designed to assess water safety in rural Kenyan streams. It utilizes **PySpark** for big data processing and **TensorFlow Lite** for decentralized, offline inference on mobile devices and laptops.

## Authors
* Nathan Karoki
* Emmanuel Ngeti

## Key Features
* **Big Data Processing:** Uses PySpark to ingest, clean, and normalize large-scale regional environmental data.
* **Deep Machine Learning:** A Sequential Neural Network trained to map complex chemical interactions (pH, Turbidity, Conductivity) to binary safety classifications (MAE: 0.0846).
* **Edge Computing:** The model is compressed into a `.tflite` format, allowing field agents to make instantaneous water safety predictions without an internet connection.
* **Streamlit Dashboard:** An interactive UI for filtering regional data and conducting manual offline predictions.

## Repository Contents
* `Water Quality Monitor (1).ipynb`: The PySpark data pipeline and TensorFlow model training code.
* `purity_edge_model.tflite`: The compressed offline Edge AI model.
* `water_quality_app.py`: The Streamlit dashboard application.
* `Project_Summary_Guide_No_IoT.pdf`: Complete system documentation.
* `2_Design_Mockups_Milestone1.pdf`: UI design wireframes.

## How to Run the Dashboard Locally
1. Ensure you have Python installed.
2. Install the required libraries: `pip install streamlit pandas numpy`
3. Run the application: `streamlit run water_quality_app.py`
