import numpy as np
from pandas.core.frame import DataFrame

class Knn_Classifier:  
    training_set: DataFrame
    k_val:int = 0

    def __init__(self, x_train, k:int):
        self.training_set = x_train
        self.k_val = k 
    
    @staticmethod
    def get_distance(row1, row2):
        return np.sqrt(sum((row1 - row2)**2))

    def get_neighbours(self, input):
        row_dist: dict = {}
        for index, row in self.training_set.iterrows():
            distance = self.get_distance(input[0], row[0:8])
            row_dist[index] = distance
        idxs = sorted(row_dist, key=row_dist.get) 
        return row_dist, idxs

    def predict(self, input):                
        distances, sorted_idxs = self.get_neighbours(input)
        print(sorted_idxs[0:self.k_val])
        possible_outcomes = self.training_set.iloc[sorted_idxs[0:self.k_val]]
        winning_class = max(set(possible_outcomes), key=possible_outcomes.count)
        return winning_class