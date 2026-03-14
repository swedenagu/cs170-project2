from numba import njit
import pandas as pd
import math
import numpy as np

# data = pd.read_csv("CS170_Small_DataSet__62.txt")
data = pd.read_csv("SanityCheck_DataSet__1.txt")

@njit
def leave_one_out_cross_validation(data, current_set, feature_to_add=None):
    # Add the candidate to the current set if one is passed in
    if feature_to_add is not None:
        current_set = np.append(current_set, feature_to_add).astype(np.int64)

    # We slice the columns where the data selected is all the rows in the feature set (x) and the current set (y) is the class label column
    x = data[:, current_set] # if len(current_set) else data[:, 1]
    y = data[:, 0]

    # Keep track of features we correctly identify
    number_correctly_classified = 0

    # Go through all the features and find the nearest neighbor distance for all of them
    for i in range(len(x)):
        object_to_classify = x[i]
        label_object_to_classify = y[i]

        # Track nearest neighbor -- initially nearest neighbor distance for each feature shouldn't be set
        nearest_neighbor_distance = np.inf
        nearest_neighbor_label = -1.0 # numba can't handle variables not initialized as floats (no inference of None types)

        for k in range(len(x)):
            if k != i:
                # Use Euclidean distance formula to calculate distance to neighbor
                distance = math.sqrt(np.sum((object_to_classify - x[k]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_label = y[k]

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

    accuracy = number_correctly_classified / len(x)
    return accuracy

@njit
def forward_selection(data, current_set, feature_to_add):
    selected = []

    # 1-indexed
    remaining = list(range(1, feature_to_add + 1))

    # The accuracy and initial set of features we have should converge to the closest to "best" values by the end of the search
    accuracy = 0
    best_features = []

    for step in range(feature_to_add):
        # current_accuracy = 
        pass

@njit
def backward_elimination(data, current_set, feature_to_add):
    pass

# accuracy = leave_one_out_cross_validation(data, current_set, feature_to_add=None) # cross-validation (not w/ K-folds), not implemented yet

# number_correctly_classified = 0

# for i in range(1, len(data)): # probably needs revision to properly parse file (adapted from MATLAB)
#     object_to_classify = data[2:] # should start from second column and go to end in row i -- how w/ pandas?
#     label_object_to_classify = data[i] # label in column 1 of row i in data

#     nearest_neigbor_distance = None # or infinity
#     nearest_neighbor_location = None # or infinity
#     for k in range(len(data)):
#         if k != i:
#             distance = math.sqrt(sum(object_to_classify - data[2:])**2)
#             if distance < nearest_neigbor_distance:
#                 nearest_neigbor_distance = distance
#                 nearest_neighbor_location = k
#                 nearest_neighbor_label = data[nearest_neighbor_location] # should be item in column 1 of row (nearest_neighbor_location)
#     if label_object_to_classify == nearest_neighbor_label:
#         number_correctly_classified += 1

# accuracy = number_correctly_classified / len(data)

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

    instances, features = x.shape
    print(f"\nThis dataset has {features} features (not including the class attribute), with {instances} instances.\n")

    # First we include all features to have a default rate to measure our search algorithms against
    default_rate = leave_one_out_cross_validation(
        np.hstack([y.reshape(-1, 1), x]),
        np.arange(1, x.shape[1]+ 1, dtype=np.int64)
    )
    print(f"\nRunning nearest neighbor with all {features}, using \"leave-one-out\" evaluation, I get an accuracy of {default_rate*100:.1f}%")

    print("\nBeginning search.\n")

    # Choose your algorithm
    if algorithm == 1:
        selected, best_acc_so_far = forward_selection(x, y, features)
    elif algorithm == 2:
        selected, best_acc_so_far = backward_elimination(x, y, features)

    # Use this to test nearest neighbor with sanity check
    elif algorithm == 3:
        selected, best_acc_so_far = leave_one_out_cross_validation(x, y, features)

    # Output the results of our search
    print(f"\nFinished search! The best feature subset is {{{','.join(map(str, sorted(selected)))}}}, which has an accuracy of {best_acc_so_far*100:.1f}%")


if __name__ == "__main__":
    main()
