AI-Driven Demand Forecasting System using LSTM

Project Overview:

Developed an AI-based demand forecasting system using LSTM
Captures time-series patterns, seasonality, and festive trends
Includes an interactive Streamlit dashboard for future demand predictions

Objectives:
Predict future demand using historical data
Incorporate seasonal and festive effects
Reduce forecasting error using deep learning
Provide an easy-to-use prediction interface

Technologies Used:
Python
TensorFlow / Keras (LSTM)
Pandas, NumPy
Scikit-learn
Streamlit
Matplotlib / Seaborn

Dataset:
File: demand_forecasting.csv
Contains historical sales data
Features -total 16 including date, product id, category,etc

Features:
Data cleaning and preprocessing
Label encoding and scaling
Time-series sequence generation
LSTM model implementation
Seasonal and festive feature integration
Evaluation using RMSE and MAE
Streamlit-based user interface

Working Process:
Data collection
Data preprocessing
Feature engineering 
Sequence creation for LSTM
Model training
Model evaluation
Prediction Visualization using Streamlit

Model Architecture:
Input layer
LSTM layers
Dense output layer
Learns long-term dependencies in time-series data

Evaluation Metrics:
RMSE (Root Mean Square Error)
MAE (Mean Absolute Error)

How to Run:
Install dependencies: pip install -r requirements.txt
Run application: streamlit run app.py

Output:
Interactive dashboard inputs
Predict demand successfully
Demand prediction & demand trend graphs
Actual vs predicted comparison graph
Usefull insights

Applications:
Retail demand forecasting
Inventory management
Supply chain optimization
E-commerce analytics

Conclusion:
LSTM improves forecasting accuracy
Handles seasonality and trend effectively
Useful for real-world business decision-making

Future Enhancements:
Real-time data integration
Advanced models (Transformer, Prophet)
Multi-product forecasting
Automated retraining
Deployment:Streamlit Cloud


-Author
Vishakha Ahirwar
