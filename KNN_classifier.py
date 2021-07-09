import numpy as np
from pandas.core.frame import DataFrame

class Knn_Classifier:  
    training_set: DataFrame
    k_val:int = 0    

    def __init__(self, training_set, k:int):
        self.training_set = training_set
        self.k_val = k         
    
    @staticmethod
    def get_distance(row1, row2):
        return np.sqrt(sum((row1 - row2)**2))

    def get_neighbours(self, input):
        row_dist: dict = {}
        for index, row in self.training_set.iterrows():
            distance = self.get_distance(input[0], row[0:8])
            row_dist[index] = distance
        idxs = sorted(row_dist.items(), key= lambda kv:(kv[1])) 
        return idxs

    def predict(self, input):   
        idxs = []             
        sorted_idxs = self.get_neighbours(input)
        # Get k indices
        for idx, _ in sorted_idxs[0:self.k_val]:
            idxs.append(idx)
        print(self.training_set['Outcome'].index)
        possible_outcomes = self.training_set['Outcome'].iloc[idxs]
        winning_class = max(set(possible_outcomes), key=possible_outcomes.count)
        return winning_class