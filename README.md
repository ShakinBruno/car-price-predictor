# Machine Learning Class - Project: Car Price Predictor
### author: Dominik Mikołajczyk

## The task the of project
The aim of the project was to create couple of regression models that, based on various categories such as: brand, model, condition or mileage; is able to predict the value of a car based on information from website craiglist.org. Also the task was to run the evaluation for each model and compare them together.

## Requirements:
1. Install jupyter:
    <!-- -->
    
        pip install jupyter
        
2. Install other required python libraries:
    <!-- -->
    
        pip install numpy==1.21.5
        pip install pandas==1.4.4
        pip install scikit-learn==1.0.2
        pip install tensorflow==2.11.0
        
## Dataset
The data comes from the Kaggle platform. The original data weighs over 1GB, so you have to [download](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) it on your own and move it to the "Data" folder. At the beginning, 426880 records were counted in the dataset and after rejecting records with missing values ​​and observing outliers, 102502 records were obtained. After division into training and testing sets, we have:
- 82001 examples for the training set (80% of all records)
- 20501 examples for the testing set (20% of all records)

### Original dataset

![image](https://github.com/ShakinBruno/car-price-predictor/assets/71774757/a29bc927-69a2-4f89-ac40-8bf3decf435b)

### [Preprocessed dataset](https://github.com/ShakinBruno/car-price-predictor/blob/main/Data/vehicles_preprocessed.csv)

![image](https://github.com/ShakinBruno/car-price-predictor/assets/71774757/956093a9-7fa5-41fc-877f-0a4a1088ac65)

## Results

![image](https://github.com/ShakinBruno/car-price-predictor/assets/71774757/53d412af-4127-4990-ac9b-dab4177563c0)

The best results in every aspect were achieved by the random forest algorithm. It can be seen that as the degree of polynomial regression increases, the model result increases (errors decrease), which proves that the data are distributed non-linearly. The metrics R2 Score, MAE (Mean Absolute Error), MSE (Mean Squared Error) and RMSE (Root Mean Squared Error) were used for evaluation.
