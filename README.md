# Autism Prediction Model

A comprehensive machine learning project for predicting Autism Spectrum Disorder (ASD) using behavioral screening data and demographic information. This project implements multiple classification algorithms with hyperparameter tuning and addresses class imbalance using SMOTE technique.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Performance Metrics](#performance-metrics)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project develops a machine learning model to predict the likelihood of Autism Spectrum Disorder (ASD) based on behavioral screening questionnaire responses and demographic information. The model is designed to assist healthcare professionals in early screening and identification of individuals who may benefit from further diagnostic evaluation.

**Key Objectives:**
- Build a robust classification model for ASD prediction
- Handle class imbalance in the dataset effectively
- Implement comprehensive data preprocessing pipeline
- Compare multiple machine learning algorithms
- Provide interpretable results for healthcare applications

## Dataset

The dataset contains **800 records** with **22 features** representing behavioral screening data and demographic information.

### Data Composition

**Target Variable:**
- `Class/ASD`: Binary classification (0 = No ASD, 1 = ASD indication)
- **Class Distribution**: 639 (No ASD) vs 161 (ASD indication)
- **Imbalance Ratio**: Approximately 4:1

**Feature Categories:**

1. **Behavioral Screening Questions (A1-A10)**
   - 10 binary features representing responses to ASD screening questionnaire
   - Each score indicates presence (1) or absence (0) of specific behavioral traits

2. **Demographic Information**
   - `age`: Participant age
   - `gender`: Gender (male/female)
   - `ethnicity`: Ethnic background (12 categories)
   - `country_of_res`: Country of residence (54 countries)
   - `relation`: Relationship to participant (Self, Parent, Relative, Others)

3. **Medical History**
   - `jaundice`: History of jaundice (yes/no)
   - `austim`: Family history of autism (yes/no)

4. **Assessment Details**
   - `used_app_before`: Previous use of screening app (yes/no)
   - `result`: Numerical screening result score

## Features

### Data Analysis Features
- **Comprehensive Exploratory Data Analysis (EDA)**
- **Statistical analysis** with correlation matrices
- **Distribution analysis** for numerical and categorical variables
- **Outlier detection** using IQR method
- **Class imbalance analysis**

### Data Preprocessing
- **Missing value handling** for categorical variables
- **Label encoding** for categorical features
- **Outlier treatment** using median replacement
- **Feature engineering** and data cleaning
- **Data standardization** and normalization

### Machine Learning Pipeline
- **Multiple algorithm comparison**
- **Hyperparameter optimization** using RandomizedSearchCV
- **Class imbalance handling** with SMOTE
- **Cross-validation** for robust model evaluation
- **Model persistence** using pickle serialization

## Technical Architecture

### Libraries and Dependencies
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Imbalanced Learning**: imbalanced-learn (SMOTE)
- **Model Persistence**: pickle

### Algorithms Implemented
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

### Model Selection Process
- **Cross-validation** with 5-fold strategy
- **Hyperparameter tuning** using RandomizedSearchCV
- **Performance comparison** across algorithms
- **Best model selection** based on accuracy metrics

## Data Preprocessing

### Data Cleaning Steps
1. **Column standardization**: Renamed inconsistent column names
2. **Data type conversion**: Converted age to integer type
3. **Missing value imputation**: Replaced '?' with 'Others' in categorical fields
4. **Category consolidation**: Merged similar categories in ethnicity and relation fields
5. **Country name standardization**: Fixed inconsistent country naming

### Feature Engineering
1. **Label encoding** applied to all categorical variables
2. **Outlier handling** using IQR method with median replacement
3. **Feature scaling** and normalization
4. **Data type optimization** for memory efficiency

### Class Imbalance Treatment
- **SMOTE (Synthetic Minority Oversampling Technique)** implementation
- **Balanced dataset creation**: 507 samples for each class
- **Stratified train-test split** to maintain class distribution

## Model Development

### Training Strategy
- **Train-Test Split**: 80% training, 20% testing
- **Cross-Validation**: 5-fold stratified cross-validation
- **Hyperparameter Search**: RandomizedSearchCV with 20 iterations
- **Class Balancing**: SMOTE applied to training data only

### Hyperparameter Grids

**Decision Tree Parameters:**
- criterion: ['gini', 'entropy']
- max_depth: [None, 10, 20, 30, 50, 70]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

**Random Forest Parameters:**
- n_estimators: [50, 100, 200, 500]
- max_depth: [None, 10, 20, 30]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- bootstrap: [True, False]

**XGBoost Parameters:**
- n_estimators: [50, 100, 200, 500]
- max_depth: [3, 5, 7, 10]
- learning_rate: [0.01, 0.1, 0.2, 0.3]
- subsample: [0.5, 0.7, 1]

## Performance Metrics

### Cross-Validation Results
- **Decision Tree**: 86.88% accuracy
- **Random Forest**: 92.80% accuracy (Best performing)
- **XGBoost**: 91.12% accuracy

### Test Set Performance
**Best Model: Random Forest Classifier**
- **Overall Accuracy**: 83.13%
- **Confusion Matrix**:
  - True Negatives: 113
  - False Positives: 19
  - False Negatives: 8
  - True Positives: 20

### Classification Report
```
              precision    recall  f1-score   support
Class 0 (No ASD)   0.93      0.86      0.89       132
Class 1 (ASD)      0.51      0.71      0.60        28
Accuracy                             0.83       160
Macro avg         0.72      0.79      0.75       160
Weighted avg      0.86      0.83      0.84       160
```

## File Structure

```
Autism-Prediction-Model/
│
├── Autism-Prediction-Model.ipynb    # Main notebook with complete analysis
├── autism-data.csv                  # Original dataset
├── autism_prediction_model.pkl      # Trained model (serialized)
├── encoders.pkl                     # Label encoders (serialized)
└── README.md                        # Project documentation
```

### File Descriptions

**Autism-Prediction-Model.ipynb**
- Complete data science workflow
- Exploratory data analysis with visualizations
- Data preprocessing and feature engineering
- Model training and hyperparameter tuning
- Performance evaluation and model comparison
- Predictive system implementation

**autism-data.csv**
- Raw dataset with 800 records and 22 features
- Behavioral screening responses and demographic data
- Target variable for ASD classification

**autism_prediction_model.pkl**
- Serialized trained Random Forest model
- Includes feature names and model parameters
- Ready for deployment and inference

**encoders.pkl**
- Serialized label encoders for categorical variables
- Required for preprocessing new prediction data
- Maintains consistency with training data encoding

## Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### Setup Instructions
1. Clone the repository
2. Install required dependencies
3. Launch Jupyter Notebook
4. Open `Autism-Prediction-Model.ipynb`
5. Run all cells to reproduce the analysis

## Usage

### Training the Model
1. **Data Loading**: Load the autism-data.csv file
2. **Preprocessing**: Run data cleaning and preprocessing steps
3. **Model Training**: Execute the model training pipeline
4. **Evaluation**: Review performance metrics and visualizations

### Making Predictions
```python
# Load the trained model and encoders
with open('autism_prediction_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

with open('encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Prepare input data
input_data = {
    "A1_Score": 1, "A2_Score": 1, "A3_Score": 0, "A4_Score": 1,
    "A5_Score": 0, "A6_Score": 1, "A7_Score": 0, "A8_Score": 1,
    "A9_Score": 0, "A10_Score": 1, "age": 40, "gender": "f",
    "ethnicity": "White-European", "jaundice": "no", "austim": "no",
    "country_of_res": "United States", "used_app_before": "no",
    "result": 7, "relation": "Self"
}

# Make prediction
prediction = model.predict(processed_data)
```

## Results

### Key Findings
1. **Random Forest** achieved the best performance with 92.80% cross-validation accuracy
2. **SMOTE technique** effectively addressed class imbalance
3. **Behavioral screening scores** (A1-A10) are highly predictive features
4. **Age and result score** contribute significantly to prediction accuracy
5. **Model generalization** confirmed through cross-validation

### Clinical Relevance
- **High precision** (93%) for negative cases reduces false alarms
- **Good recall** (71%) for positive cases aids in early detection
- **Balanced performance** suitable for screening applications
- **Interpretable results** support clinical decision-making

## Limitations

### Data Limitations
1. **Limited sample size** (800 records) may affect generalization
2. **Class imbalance** in original dataset requires synthetic data generation
3. **Geographic bias** with uneven representation across countries
4. **Demographic representation** may not reflect global population

### Model Limitations
1. **Screening tool focus**: Model is for screening, not diagnostic purposes
2. **Feature dependency**: Relies on subjective questionnaire responses
3. **Temporal validity**: Model may require updates as screening criteria evolve
4. **Cultural considerations**: May need adaptation for different cultural contexts

### Technical Limitations
1. **Static model**: Requires retraining for new data patterns
2. **Preprocessing dependency**: New data must follow exact preprocessing steps
3. **Feature engineering**: Limited to current feature set and transformations

## Future Enhancements

### Model Improvements
1. **Deep learning approaches**: Explore neural networks for pattern recognition
2. **Ensemble methods**: Combine multiple algorithms for improved accuracy
3. **Feature selection**: Advanced techniques for optimal feature subset
4. **Regularization techniques**: Prevent overfitting with larger datasets

### Data Enhancements
1. **Larger datasets**: Collect more diverse and representative data
2. **Longitudinal studies**: Track individuals over time for better insights
3. **Multi-modal data**: Include additional behavioral and physiological measures
4. **Cross-cultural validation**: Test model across different populations

### Technical Enhancements
1. **Web application**: Deploy model as user-friendly web interface
2. **API development**: Create REST API for integration with healthcare systems
3. **Real-time processing**: Enable instant predictions for clinical use
4. **Model monitoring**: Implement drift detection and automatic retraining

### Clinical Integration
1. **Healthcare workflow**: Integration with electronic health records
2. **Clinical validation**: Collaborate with healthcare professionals for validation
3. **Regulatory compliance**: Ensure adherence to medical device regulations
4. **User training**: Develop training materials for healthcare providers

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** and include comprehensive comments
3. **Add unit tests** for new functionality
4. **Update documentation** to reflect changes
5. **Submit pull request** with detailed description of changes

### Areas for Contribution
- Data preprocessing improvements
- New algorithm implementations
- Performance optimization
- Documentation enhancements
- Bug fixes and code quality improvements

## License

This project is available for educational and research purposes. Please ensure compliance with data privacy regulations and obtain appropriate approvals for clinical use.

**Disclaimer**: This model is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.

---

**Contact Information**: For questions, suggestions, or collaborations, please open an issue in the repository or contact the project maintainer.

**Last Updated**: September 2025
