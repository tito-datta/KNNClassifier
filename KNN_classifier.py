import numpy as np
from pandas.core.frame import DataFrame

class Knn_Classifier:  
    training_set: DataFrame
    k_val:int = 0    
    target_column: str = ''
    target_column_idx: int = -1

    def __init__(self, training_set: DataFrame, k:int, target_column: str):
        self.training_set = training_set
        self.k_val = k        
        if target_column == '' or target_column not in training_set.columns:
             print(f'The provided target column {target_column} does not exist in the training set.')
             return np.nan
        self.target_column = target_column
        idx: int = 0
        for col in training_set.columns:
            if col == target_column:
                break
            idx += 1
        self.target_column_idx = idx
    
    @staticmethod
    def get_distance(row1, row2):
        sum: float = 0.0
        for i in range(0, len(row1)):
            sum += (row1[i]- row2[i])**2
        return np.sqrt(sum)

    def get_neighbours(self, input):
        row_dist: dict = {}
        for row in self.training_set.itertuples():
            distance = self.get_distance(input[1:8], row[1:8])
            row_dist[row[0]] = distance
        idxs = sorted(row_dist.items(), key= lambda kv:(kv[1])) 
        return idxs
    
    @staticmethod
    def vote_winning_class(members):
        return max(members, key=members.count)
    
    def get_k_neighbours(self, sorted_list_idxs: list):
        neighbours = []
        for idx, _ in sorted_list_idxs[0:self.k_val]:
            neighbours.append(self.training_set[self.target_column].loc[idx])
        return neighbours   

    def predict(self, input: tuple):             
        sorted_idxs = self.get_neighbours(input)
        neighbours = self.get_k_neighbours(sorted_idxs)
        winning_class = self.vote_winning_class(neighbours)
        return winning_class