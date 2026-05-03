# Cholera Outbreak Prediction System

## Overview
This project implements a machine learning-based early warning system for cholera outbreak prediction at the district level. The system uses environmental, temporal, and epidemiological data to forecast cholera risk, enabling proactive public health interventions.

## Features
- **Interactive Dashboard**: Built with Streamlit for intuitive data exploration and model interaction
- **Machine Learning Model**: Stacking ensemble (Random Forest, Gradient Boosting, AdaBoost) with Logistic Regression meta-learner
- **Temporal Features**: Incorporates lagged variables (1, 2, 4 weeks) and rolling statistics (4-week windows)
- **Environmental Drivers**: Temperature, precipitation, vegetation index (LAI), and coastal proximity
- **Model Explainability**: SHAP values for feature importance interpretation
- **Multi-page Interface**: Overview, Data Analysis, Model Training, and Forecasting sections
- **Dark Theme UI**: Optimized for visual clarity and reduced eye strain

<img width="1366" height="657" alt="Capture" src="https://github.com/user-attachments/assets/c3473e6c-f0a7-428d-8525-363cc8124d0f" />

<img width="1365" height="656" alt="Capture1" src="https://github.com/user-attachments/assets/2b8d3b2b-41d3-4a26-8ba3-6a52870e37cf" />

## Dataset
The system uses `cholera_data_v3.csv` containing district-level cholera cases and associated environmental variables including:
- Temporal: year, month, day, week of outbreak
- Epidemiological: cholera cases, fatality rate
- Environmental: temperature, precipitation, leaf area index (LAI)
- Geographic: latitude, longitude, state/UT, district, coastal indicator
- Seasonal: seasonal classifications

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/AC052001/Cholera-Prediction-System.git
   cd cholera-prediction-system
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
<img width="1366" height="656" alt="Capture2" src="https://github.com/user-attachments/assets/5ef4e644-a068-43b6-81f3-ed155b81b602" />

<img width="1366" height="655" alt="Capture3" src="https://github.com/user-attachments/assets/07c8f7b4-71eb-488d-8702-86fffa75894a" />

## Usage
1. Ensure the `cholera_data_v3.csv` file is in the project root directory
2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Navigate through the sidebar menu:
   - **Overview**: Project summary and key metrics
   - **Data Analysis**: Exploratory data analysis with interactive visualizations
   - **Model Training**: Train the stacking ensemble model and view performance metrics
   - **Forecasting**: Generate cholera outbreak predictions for future periods

## Model Architecture
The prediction system employs a stacking ensemble approach:
- **Base Learners**: Random Forest, Gradient Boosting, AdaBoost classifiers
- **Meta-Learner**: Logistic Regression
- **Feature Engineering**:
  - Temporal lags (1, 2, 4 weeks) for precipitation, temperature, and cases
  - Rolling statistics (4-week windows) for precipitation and cases
  - Interaction terms (temperature × precipitation)
  - Aggregated features (week number, coastal indicator)
- **Target Variable**: Binary classification (above/below median cholera cases)

<img width="1366" height="652" alt="Capture8" src="https://github.com/user-attachments/assets/c1fed5a2-2e41-4232-9964-245e188a230f" />

## File Structure
```
cholera-prediction-system/
│
├── app.py                 # Main Streamlit application
├── cholera_data_v3.csv    # Dataset
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
└── .streamlit/            # Streamlit configuration
    └── config.toml        # Streamlit theme and layout settings
```

<img width="1366" height="655" alt="Capture9" src="https://github.com/user-attachments/assets/462c83c7-e46a-4f6f-bb99-b7b9cc9ea8a6" />

<img width="1366" height="651" alt="Capture10" src="https://github.com/user-attachments/assets/a0c565eb-eab3-4cee-90d1-0e97a40b1622" />

## Dependencies
Key Python packages required:
- streamlit==1.32.0
- pandas
- numpy
- scikit-learn
- catboost==1.2.5
- matplotlib
- seaborn
- shap
- joblib
- threadpoolctl>=3.0.0
- plotly

## Model Performance
The model is evaluated using F2-score (emphasizing recall) and Precision-Recall AUC, suitable for imbalanced outbreak prediction scenarios. Feature importance analysis reveals:
- Lagged case variables (Cases_lag_1, Cases_lag_2, Cases_lag_4) as top predictors
- Temperature × precipitation interaction as key environmental driver
- Coastal districts showing elevated risk profiles

<img width="1366" height="654" alt="Capture4" src="https://github.com/user-attachments/assets/035cb7fb-7a5a-406f-b462-f432ed8d7c81" />

## How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<img width="1366" height="653" alt="Capture5" src="https://github.com/user-attachments/assets/22836924-8335-48b5-89fa-c14cb8dede9d" />

<img width="1366" height="654" alt="Capture6" src="https://github.com/user-attachments/assets/54b7c63c-0182-41dc-9db2-437309327d64" />

<img width="1366" height="656" alt="Capture7" src="https://github.com/user-attachments/assets/8355e5d2-a532-4fe1-bdc4-729f46e44308" />

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Open-source libraries: Streamlit, Scikit-learn, CatBoost, SHAP
- Public health professionals working on cholera prevention and control

## Contact
For questions or collaboration inquiries, please open an issue in the GitHub repository.
