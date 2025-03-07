# Machine Learning Class - Project: Car Price Predictor

### Author: Dominik Mikołajczyk

## Project Overview
The aim of this project was to develop several regression models capable of predicting the value of a car based on various attributes such as brand, model, condition, and mileage. The data for this project was sourced from craigslist.org via a dataset available on Kaggle. The project involved preprocessing the data, training multiple models, evaluating their performance, and comparing the results to determine the most effective model.

## Requirements
To run this project, you need to have the following software installed:

1. **Jupyter Notebook**: For running and sharing the code.
   ```bash
   pip install jupyter
   ```

2. **Python Libraries**: The project requires specific versions of several Python libraries.
   ```bash
   pip install numpy==1.21.5
   pip install pandas==1.4.4
   pip install scikit-learn==1.0.2
   pip install tensorflow==2.11.0
   ```

## Dataset
The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) and contains information on cars listed on craigslist.org. The original dataset contained 426,880 records. After cleaning the data by removing records with missing values and outliers, we obtained 102,502 records. The dataset was then split into training and testing sets as follows:
- **Training Set**: 82,001 examples (80% of the data)
- **Testing Set**: 20,501 examples (20% of the data)

### Data Preprocessing
The preprocessing steps included handling missing values, removing outliers, and normalizing the data to ensure that the models could learn effectively from the dataset.

### Original Dataset
![Original Dataset](https://github.com/ShakinBruno/car-price-predictor/assets/71774757/a29bc927-69a2-4f89-ac40-8bf3decf435b)

### [Preprocessed Dataset](https://github.com/ShakinBruno/car-price-predictor/blob/main/Data/vehicles_preprocessed.csv)
![Preprocessed Dataset](https://github.com/ShakinBruno/car-price-predictor/assets/71774757/956093a9-7fa5-41fc-877f-0a4a1088ac65)

## Models Implemented
The project explored the performance of six different regression models:

1. **Linear Regression**: A basic model that assumes a linear relationship between the input variables and the target variable.
2. **Polynomial Regression (3rd Degree)**: A non-linear model without regularization to capture more complex relationships.
3. **Polynomial Regression (5th Degree with L2 Regularization)**: A more complex model with regularization to prevent overfitting.
4. **Random Forest Regressor**: An ensemble method using 100 decision trees to improve prediction accuracy.
5. **K-Nearest Neighbors (KNN) Regressor**: A model that predicts the value based on the closest 3 examples in the dataset.
6. **Neural Network**: A linear neural network with four hidden layers (512, 128, 256, 64 neurons) using ReLU activation and Dropout layers to prevent overfitting.

## Evaluation Metrics
The models were evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R2 Score**

## Results
The evaluation results for each model are summarized in the table below:

| Model                           | MAE      | RMSE     | R2 Score |
|---------------------------------|----------|----------|----------|
| Linear Regression               | 4463.23  | 5876.05  | 0.7474   |
| Polynomial Regression (3rd)     | 3195.77  | 4510.25  | 0.8512   |
| Polynomial Regression (5th)     | 2681.56  | 4038.33  | 0.8807   |
| Random Forest Regressor         | 1495.93  | 2837.46  | 0.9411   |
| K-Nearest Neighbors Regressor   | 2027.60  | 3686.37  | 0.9006   |
| Neural Network                  | 2909.49  | 4260.30  | 0.8672   |

The best performance was achieved by the Random Forest Regressor, which outperformed other models across all metrics. The results indicate that the data is non-linear, as evidenced by the improved performance of polynomial regression models over linear regression. However, increasing the polynomial degree beyond a certain point without regularization led to overfitting.

## Conclusion
The project successfully demonstrated the application of various regression techniques to predict car prices. The Random Forest Regressor emerged as the most effective model, highlighting the importance of ensemble methods in handling complex datasets. Future work could explore further hyperparameter tuning and the use of additional features to enhance model performance.

