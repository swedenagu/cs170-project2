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
        indices = np.append(current_set, feature_to_add).astype(np.int64)
    else:
        indices = current_set.astype(np.int64)

    # We slice the columns where the data selected is all the rows in the feature set (x) and the current set (y) is the class label column
    x = data[:, indices] # if len(current_set) else data[:, 1]
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
def forward_selection(data, current_set):
    selected = np.empty(0, dtype=np.int64)

    # 1-indexed
    remaining = np.arange(1, current_set + 1, dtype=np.int64)

    # The accuracy and initial set of features we have should converge to the closest to "best" values by the end of the search
    best_accuracy = 0.0
    best_features = np.empty(0, dtype=np.int64)

    # Look at all the features and update the accuracy and best feature at each step of the search
    for step in range(current_set):
        current_best_accuracy = 0.0
        # Initially we don't have a "best" feature if we haven't compared to another yet
        current_best_feature = -1

        # Now examine the rest of the features besides the current one
        for i in range(len(remaining)):
            feature = remaining[i]
            accuracy = leave_one_out_cross_validation(data, selected, feature_to_add=feature)

            # How accurate is each candidate (or set of candidates if we're not on the first one)?
            print("Using feature(s) ", feature, ", accuracy is ", accuracy * 100, "%")

            if accuracy > current_best_accuracy:
                current_best_accuracy = accuracy
                current_best_feature = feature

        # The best feature we found so far should be added to the set of selected features and removed from the ones remaining we need to check
        selected = np.append(selected, current_best_feature)
        deletion_mask = remaining != current_best_feature
        remaining = remaining[deletion_mask]

        # How do we know if the feature we just added makes the accuracy worse?
        if current_best_accuracy < best_accuracy:
            print("Warning: Accuracy has decreased! Continuing search in case of local maxima")
        else:
            best_accuracy = current_best_accuracy
            # We want a shallow copy of the current best features we picked in order to avoid changing them later on accident
            best_features = selected.copy()
        
        # print(f"Feature set {{{','.join(map(str, sorted(best_features)))}}} was best, accuracy is {current_best_accuracy * 100:.1f}")
        print("Feature set [",  best_features, "] was best, accuracy is ", current_best_accuracy * 100, "%")
    
    return best_features, best_accuracy

@njit
def backward_elimination(data, current_set):
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
    full_data = np.hstack([y.reshape(-1, 1), x])
    full_features = np.arange(1, x.shape[1] + 1, dtype=np.int64)
    default_rate = leave_one_out_cross_validation(
        full_data, full_features
    )
    print(f"\nRunning nearest neighbor with all {features}, using \"leave-one-out\" evaluation, I get an accuracy of {default_rate*100:.1f}%")

    print("\nBeginning search.\n")

    # Choose your algorithm
    if algorithm == 1:
        selected, best_acc_so_far = forward_selection(full_data, features)
    elif algorithm == 2:
        selected, best_acc_so_far = backward_elimination(full_data, features)

    # Use this to test nearest neighbor with sanity check
    elif algorithm == 3:
        subset = np.array([7, 10, 12], dtype=np.int64)
        best_acc_so_far = leave_one_out_cross_validation(x, y, features)
        selected = subset

    # Output the results of our search
    print(f"\nFinished search! The best feature subset is {{{','.join(map(str, sorted(selected)))}}}, which has an accuracy of {best_acc_so_far*100}%")


if __name__ == "__main__":
    main()
