import pandas as pd
import math
import numpy as np

data = pd.read_csv("CS170_Small_DataSet__62.txt")

def leave_one_out_cross_validation(data, current_set, feature):
    pass

def forward_selection(data, current_set, feature_to_add):
    pass

def backward_elimination(data, current_set, feature_to_add):
    pass

accuracy = leave_one_out_cross_validation(data, current_set=None, feature_to_add=None) # cross-validation (not w/ K-folds), not implemented yet

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

def main():
    filename = input("Welcome to Sweden's Feature Selection Algorithm! Which set are you testing? ")

    print("\nType the number of the algorithm you want to run.")
    print("\n\t1) Forward Selection")
    print("\n\t2) Backward Elimination\n")

    algorithm = int(input())

    
    # Load data (all the columns after the class labels are different features)
    data = np.loadtxt(filename)
    y = data[:, 0]
    x = data[:, 1:]

    instances, features = X.shape
    print(f"\nThis dataset has {features} features (not including the class attribute), with {instances} instances.\n")

    # First we include all features to have a default rate to measure our search algorithms against
    default_rate = leave_one_out_cross_validation(x, y)
    print(f"\nRunning nearest neighbor with all {features}, using \"leave-one-out\" evaluation, I get an accuracy of {default_rate*100:.1f}%")

    print("\nBeginning search.\n")

    # Choose your algorithm
    if algorithm == 1:
        selected, best_acc_so_far = forward_selection(x, y, features)


if __name__ == "__main__":
    main()
