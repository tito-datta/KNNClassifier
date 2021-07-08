from sklearn.model_selection import train_test_split
import pandas as pd
from KNN_classifier import Knn_Classifier as knn_classifier

diabetes_df = pd.read_csv('data/diabetes.csv')

predictors = diabetes_df.drop(labels=['Outcome'], axis=1)
target = diabetes_df[['Outcome']]

X_Train,X_Test,Y_Train,Y_Test = train_test_split(predictors, target, test_size=0.75, random_state=10, stratify=target)

diabetes_train = pd.DataFrame(data=X_Train, columns=predictors.columns).assign(Outcome=Y_Train)
diabetes_test = pd.DataFrame(data=X_Test, columns=predictors.columns).assign(Outcome=Y_Test)

knn = knn_classifier(diabetes_train,3)

print(knn.predict(X_Test[:1].values))
