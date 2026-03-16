from numba import njit, prange
import time
import math
import numpy as np

@njit(parallel=True)
def leave_one_out_cross_validation(data, current_set, feature_to_add=None):
    # Add the candidate to the current set if one is passed in
    if feature_to_add is not None:
        indices = np.append(current_set, feature_to_add).astype(np.int64)
    else:
        indices = current_set.astype(np.int64)

    # We slice the columns where the data selected is all the rows
    #   in the feature set (x) and the current set (y) is the class label column
    x = data[:, indices]  # if len(current_set) else data[:, 1]
    y = data[:, 0]

    # Keep track of features we correctly identify
    number_correctly_classified = 0

    # Go through all the features and find the nearest neighbor distance for all of them
    # We use prange (from numba library) to parallelize computation
    # The goal is to speed up nearest neighbor by running different for loop iterations on different CPU threads
    for i in prange(len(x)):
        object_to_classify = x[i]
        label_object_to_classify = y[i]

        # Track nearest neighbor -- initially nearest neighbor distance for each feature shouldn't be set
        nearest_neighbor_distance = np.inf
        nearest_neighbor_label = (
            -1.0
        )  # numba can't handle variables not initialized as floats (no inference of None types)

        for k in range(len(x)):
            if k != i:
                # Use Euclidean distance formula to calculate distance to neighbor
                diff = object_to_classify - x[k]
                # We still square the distance between the current feature and 
                #   object we're trying to classify but with a diff variable instead of computing it in-place
                # Computing the matrix dot product is faster than doing a sum across all elements
                distance = math.sqrt(np.dot(diff, diff))
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
    # Each call to np.arange is similar to range from typical Python. It returns a numpy ndarray.
    # We also cast the type of each element in the np array to a 64-bit integer.
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
            accuracy = leave_one_out_cross_validation(
                data, selected, feature_to_add=feature
            )

            # How accurate is each candidate (or set of candidates if we're not on the first one)?
            print("Using feature(s) ", feature, ", accuracy is ", round(accuracy * 100, 2), "%")

            if accuracy > current_best_accuracy:
                current_best_accuracy = accuracy
                current_best_feature = feature

        # The best feature we found so far should be added to the set of 
        #   selected features and removed from the ones remaining we need to check
        selected = np.append(selected, current_best_feature)
        remaining = remaining[remaining != current_best_feature]

        # How do we know if the feature we just added makes the accuracy worse?
        if current_best_accuracy < best_accuracy:
            print(
                "Warning: Accuracy has decreased! Continuing search in case of local maxima"
            )
        else:
            best_accuracy = current_best_accuracy
            # We want a deep copy of the current best features we picked in order to avoid changing them later on accident
            best_features = selected.copy()

        print("Feature set [", best_features, "] was best, accuracy is ", round(current_best_accuracy * 100, 2), "%")

    return best_features, best_accuracy

@njit
def backward_elimination(data, current_set):
    # We can ignore some features we picked in the forward search. Initally we look at all of them.
    selected = np.arange(1, current_set + 1, dtype=np.int64)

    # The best accuracy we converge to should continuously update as we 
    #   remove features rather than adding them like in forward search. The selected 
    #   set also should be deep copied to avoid accidentally changing it during each update.
    best_accuracy = leave_one_out_cross_validation(data, selected)
    best_features = selected.copy()

    # The current number of features (current_set) should be an integer; we stop once there's a single feature left
    for step in range(current_set - 1):
        current_best_accuracy = 0.0
        current_worst_feature = -1

        for i in range(len(selected)):
            feature = selected[i]
            # Use the same mask (like in line 78) to try removing the current feature
            candidate = selected[selected != feature]
            accuracy = leave_one_out_cross_validation(data, candidate)

            print(
                "Using feature(s)", candidate, " accuracy is ", round(accuracy * 100, 2), "%"
            )

            if accuracy > current_best_accuracy:
                current_best_accuracy = accuracy
                current_worst_feature = feature

        # Now get rid of the "worst" feature
        selected = selected[selected != current_worst_feature]

        # Update the best accuracy if the accuracy is better after removing a feature
        if current_best_accuracy < best_accuracy:
            print(
                "Warning: Accuracy has decreased! Continuing search in case of local maxima"
            )
        else:
            best_accuracy = current_best_accuracy
            best_features = selected.copy()

        print("Feature set [", selected, "] was best, accuracy is ", round(current_best_accuracy * 100, 2), "%")

    return best_features, best_accuracy

def main():
    filename = input(
        "Welcome to Sweden's Feature Selection Algorithm! Which set are you testing? "
    )

    print("\nType the number of the algorithm you want to run.")
    print("\n\t1) Forward Selection")
    print("\n\t2) Backward Elimination\n")

    algorithm = int(input())

    # Load data (all the columns after the class labels are different features)
    data = np.loadtxt(filename)
    y = data[:, 0]
    x = data[:, 1:]

    instances, features = x.shape
    print(
        f"\nThis dataset has {features} features (not including the class attribute), with {instances} instances.\n"
    )

    # First we include all features to have a default rate to measure our search algorithms against

    # We need this numpy function so that all the columns of features are combined into one column
    # When we do operations later on (e.g. slicing up the feature sets), we can end up forming non-contiguous arrays
    # We want to maximize the speed of numpy operations by allowing for 
    #   vectorization and keeping them C-contiguous where all data is stored row-by-row
    full_data = np.ascontiguousarray(np.hstack([y.reshape(-1, 1), x]))
    full_features = np.arange(1, x.shape[1] + 1, dtype=np.int64)
    default_rate = leave_one_out_cross_validation(full_data, full_features)
    print(
        f'\nRunning nearest neighbor with all {features}, using "leave-one-out" evaluation," I get an accuracy of {default_rate*100:.2f}%'
    )

    print("\nBeginning search.\n")
    # Track how long each search takes to run

    # high-res clock from Python time library that acts as a monotonic performance counter
    start_time = (time.perf_counter())

    # Choose your algorithm
    if algorithm == 1:
        selected, best_acc_so_far = forward_selection(full_data, features)
    elif algorithm == 2:
        selected, best_acc_so_far = backward_elimination(full_data, features)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time was {elapsed_time:.4f} seconds")

    # Output the results of our search
    print(
        f"\nFinished search! The best feature subset is {{{','.join(map(str, sorted(selected)))}}}, which has an accuracy of {best_acc_so_far*100:.2f}%"
    )

if __name__ == "__main__":
    main()
