from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pandas as pd
from KNN_classifier import Knn_Classifier as knn_classifier

diabetes_df = pd.read_csv('data/diabetes.csv')

predictors = diabetes_df.drop(labels=['Outcome'], axis=1)
target = diabetes_df[['Outcome']]

X_Train,X_Test,Y_Train,Y_Test = train_test_split(predictors, target, test_size=0.75, random_state=10, stratify=target)

diabetes_train = pd.DataFrame(data=X_Train, columns=predictors.columns).assign(Outcome=Y_Train)
diabetes_test = pd.DataFrame(data=X_Test, columns=predictors.columns).assign(Outcome=Y_Test)

performance_hist = {}
for i in range(1,30):
    knn = knn_classifier(diabetes_train,i,'Outcome')
    predicted = []
    for row in diabetes_test.itertuples():
        predicted.append(knn.predict(row[0:8]))

    accuracy = metrics.accuracy_score(Y_Test, predicted)
    f1_score = metrics.f1_score(Y_Test, predicted)
    performance_hist[i] = (round(accuracy, 4),round(f1_score, 4))
    # print(f'The accuracy of the model for k = {i} is {round(accuracy, 4)} and the f1 score of the model for k = {i} is {round(f1_score, 4)}.')

sorted_idxs = sorted(performance_hist, key=performance_hist.get, reverse=True)
print('The top three performing values of K are: ')
for idx in sorted_idxs[0:3]:
    print(f'For value {idx} of k the accuracy is {performance_hist[idx][0]} and the f1 score is {performance_hist[idx][1]}')
