from sklearn.model_selection import train_test_split
import pandas as pd
from KNN_classifier import Knn_Classifier as knn_classifier

diabetes_df = pd.read_csv('data/diabetes.csv')

predictors = diabetes_df.drop(labels=['Outcome'], axis=1)
target = diabetes_df[['Outcome']]
# print(f'The target column for our dataset is {target.columns.values} and the predictors are {predictors.columns.values}')

# Split data in training & tests sub sets
X_Train,X_Test,Y_Train,Y_Test = train_test_split(predictors, target, test_size=0.75, random_state=10, stratify=target)
# print(f'The training set has shape:\n\tPredictor\'s: {X_Train.shape}\n\tTarget: {Y_Train.shape} and test data has shape:\n\tPredictor\'s: {X_Test.shape}\n\tTarget: {Y_Test.shape}')

knn = knn_classifier(X_Train,X_Test,Y_Train,Y_Test,3)
print(knn.predict(X_Test[:1].values))