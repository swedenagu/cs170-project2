import pandas as pd
import math

data = pd.read_csv("CS170_Small_DataSet__62.txt")

def leave_one_out_cross_validation(data, current_set, feature):
    pass

accuracy = leave_one_out_cross_validation(data, current_set, feature_to_add) # cross-validation (not w/ K-folds), not implemented yet

number_correctly_classified = 0

for i in range(1, len(data)): # probably needs revision to properly parse file (adapted from MATLAB)
    object_to_classify = data[2:] # should start from second column and go to end in row i -- how w/ pandas?
    label_object_to_classify = data[i] # label in column 1 of row i in data

    nearest_neigbor_distance = None # or infinity
    nearest_neighbor_location = None # or infinity
    for k in range(len(data)):
        if k != i:
            distance = math.sqrt(sum(object_to_classify - data[2:])**2)
            if distance < nearest_neigbor_distance:
                nearest_neigbor_distance = distance
                nearest_neighbor_location = k
                nearest_neighbor_label = data[nearest_neighbor_location] # should be item in column 1 of row (nearest_neighbor_location)
    if label_object_to_classify == nearest_neighbor_label:
        number_correctly_classified += 1

accuracy = number_correctly_classified / len(data)
