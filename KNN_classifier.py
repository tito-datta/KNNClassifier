import numpy as np
import pandas as pd

class Knn_Classifier:  
    training_set  = []
    test_set = []
    k_val = 0

    def __init__(self, x_train, y_train, x_test, y_test, k:int):
        self.training_set = pd.concat([x_train, y_train])
        self.test_set = pd.concat([x_test, y_test])
        self.k_val = k
    
    @staticmethod
    def get_distance(row1: tuple, row2: tuple):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return np.sqrt(distance)

    def get_neighbours(self, input):
        row_dist = {}
        for row in self.training_set.items():
            distance = self.get_distance(input, row[1])
            row_dist[row.index] = distance
        row_dist = sorted(row_dist, key=row_dist.get)
        return row_dist

    def predict(self, input):
        neighbours = self.get_neighbours(input)
        voter_indices = neighbours[0: self.k_val]
        possible_outcomes = self.training_set.__getitem__(voter_indices)
        winning_class = max(set(possible_outcomes), key=possible_outcomes.count)
        return winning_class