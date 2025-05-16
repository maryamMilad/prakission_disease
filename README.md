# Parkinson's Disease Detection System

## Overview
This project implements a machine learning system to detect Parkinson's Disease using voice measurement features. The system uses a Support Vector Machine (SVM) classifier to analyze vocal characteristics and predict the presence of Parkinson's with 87% accuracy.

## Key Features

### Data Analysis
- 195 voice measurement samples
- 22 predictive features including:
  - Fundamental frequency variations (jitter)
  - Amplitude variations (shimmer)
  - Nonlinear complexity measures (RPDE, DFA, PPE)
- Class imbalance: 147 Parkinson's vs 48 healthy cases

### Model Implementation
- SVM classifier with linear kernel
- StandardScaler for feature normalization
- 80-20 train-test split
- Achieves 87.2% accuracy on both training and test sets

## Technical Implementation
```python
# Data Preprocessing
x = df.drop(columns=['status','name'])
y = df['status']
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# Model Training
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

# Prediction Example
input_data = (223.365,238.987,...)  # Voice measurements
std_data = scaler.transform(input_data.reshape(1,-1))
prediction = model.predict(std_data)
```

## Usage
1. Provide voice measurement data as input array
2. System returns:
   - Binary prediction (0 = healthy, 1 = Parkinson's)
   - Clear diagnostic message

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn

## Performance
- Training Accuracy: 87.2%
- Test Accuracy: 87.2%
- Consistent performance across datasets

## Future Enhancements
- Address class imbalance with SMOTE/oversampling
- Experiment with other classifiers (Random Forest, XGBoost)
- Develop feature importance analysis
- Create web/mobile interface for clinical use
- Incorporate additional biomarkers for improved accuracy

## Clinical Relevance
This system provides a non-invasive, voice-based screening tool that could assist clinicians in early Parkinson's detection, though it should not replace comprehensive medical evaluation.
