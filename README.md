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
   git clone https://github.com/yourusername/cholera-prediction-system.git
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

## How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Open-source libraries: Streamlit, Scikit-learn, CatBoost, SHAP
- Public health professionals working on cholera prevention and control

## Contact
For questions or collaboration inquiries, please open an issue in the GitHub repository.