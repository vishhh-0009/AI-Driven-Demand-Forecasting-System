# AI-Driven Demand Forecasting System using LSTM

**Live demo:** _add link after deployment_

## Project Overview
- Developed an AI-based demand forecasting system using an LSTM network
- Learns patterns from historical sales, pricing, promotion, weather and seasonality features
- Includes an interactive Streamlit dashboard for demand predictions

## Technologies Used
- Python 3.10
- TensorFlow / Keras (LSTM)
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## Dataset
- File: `demand_forecasting.csv`
- 76,000 records across 5 stores, 20 products, 5 categories and 4 regions
- 16 columns: Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Price, Discount, Weather Condition, Promotion, Competitor Pricing, Seasonality, Epidemic and the target Demand
- Demand ranges from 4 to 430 units (average about 104)

## Features
- Data cleaning and preprocessing
- Label encoding and MinMax scaling
- Time-series sequence generation
- LSTM model implementation
- Seasonality and weather features
- Evaluation using RMSE and MAE
- Streamlit-based user interface

## Model Architecture
- Input layer (10 time steps x 15 features)
- LSTM layers
- Dense output layer (single demand value)
- Learns sequential dependencies in the data

## Evaluation Metrics
| Metric | Test set |
|--------|----------|
| RMSE (Root Mean Square Error) | 36.89 |
| MAE (Mean Absolute Error) | 28.41 |

## How to Run
```powershell
git clone https://github.com/vishhh-0009/AI-Driven-Demand-Forecasting-System.git
cd AI-Driven-Demand-Forecasting-System
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Python 3.10 is recommended (the model was built with TensorFlow 2.13).

## Output
- Interactive dashboard with sidebar inputs
- Predicted demand with summary metrics
- Demand prediction and demand trend charts (illustrative)
- Actual vs predicted comparison chart (illustrative)
- Simple stock-level insight based on predicted demand

## Project Structure
- `app.py`: Streamlit dashboard
- `app.ipynb`: data preparation, training and evaluation
- `lstm_model.h5`: trained model
- `scaler.pkl`: fitted MinMaxScaler used by the app
- `demand_forecasting.csv`: dataset
- `requirements.txt`: dependencies

## Limitations
- The dashboard repeats one input row across the 10-step window instead of using real history
- The trend and actual-vs-predicted charts are illustrative and not yet based on test-set results
- Date is represented by a simple Time_Step index, not calendar features
- No comparison against baseline models yet

## Future Enhancements
- Replace illustrative charts with test-set actual vs predicted results
- Add calendar features (month, day of week) and baseline comparison (naive, ARIMA)
- Advanced models (Transformer, Prophet)
- Multi-product forecasting
- Docker, CI/CD and AWS/GCP deployment
- Real-time data integration and automated retraining

## Conclusion
This project demonstrates an end-to-end machine learning workflow: data preparation, LSTM modeling, evaluation, and an interactive Streamlit app for demand prediction. It can be extended into a more robust forecasting tool for inventory and supply chain decisions.

## Author
Vishakha Ahirwar