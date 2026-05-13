import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import BallTree, KDTree

#  1. LOAD THE DATA
train_data = np.load('data/train_data.npy')
train_labels = np.load('data/train_labels.npy')
test_data = np.load('data/test_data.npy')
test_labels = np.load('data/test_labels.npy')

# Print dimensions and class distribution
print("Training dataset dimensions: ", train_data.shape)
print("Number of training labels: ", len(train_labels))
print("Testing dataset dimensions: ", test_data.shape)
print("Number of testing labels: ", len(test_labels))

train_digits, train_counts = np.unique(train_labels, return_counts=True)
print("Training set distribution:")
print(dict(zip(train_digits, train_counts)))

test_digits, test_counts = np.unique(test_labels, return_counts=True)
print("Test set distribution:")
print(dict(zip(test_digits, test_counts)))

#  2. VISUALIZING THE DATA
def show_digit(x):
    """Display a single 28x28 digit image."""
    plt.axis('off')
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

def vis_image(index, dataset="train"):
    """Visualize a digit by index and print its true label."""
    if dataset == "train":
        show_digit(train_data[index])
        label = train_labels[index]
    else:
        show_digit(test_data[index])
        label = test_labels[index]
    print("Actual Label:", label)
    return

# View first images
print("\n=== First training image ===")
vis_image(0, "train")
print("\n=== First test image ===")
vis_image(0, "test")

#  3. DISTANCE FUNCTION
def squared_dist(x, y):
    """Squared Euclidean distance between two vectors."""
    return np.sum(np.square(x - y))

# Intuitive distance examples (7 vs 1, 7 vs 2, 7 vs 7)
print("\n=== Distance examples ===")
vis_image(4, "train")  # a '7'
vis_image(5, "train")  # a '1'
print("Distance from 7 to 1:", squared_dist(train_data[4], train_data[5]))

vis_image(4, "train")
vis_image(1, "train")  # a '2'
print("Distance from 7 to 2:", squared_dist(train_data[4], train_data[1]))

vis_image(4, "train")
vis_image(7, "train")  # another '7'
print("Distance from 7 to different 7:", squared_dist(train_data[4], train_data[7]))

vis_image(4, "train")
vis_image(4, "train")
print("Distance from 7 to itself:", squared_dist(train_data[4], train_data[4]))

#  4. NEAREST NEIGHBOR CLASSIFIER (IMPROVED)
def find_NN(x):
    """
    Return index of nearest neighbor in training set.
    VECTORIZED version - much faster than original Python loop!
    """
    # Broadcasting: subtract x from EVERY row of train_data at once
    distances = np.sum((train_data - x) ** 2, axis=1)
    return np.argmin(distances)

def NN_classifier(x):
    """Predict label using 1-NN."""
    idx = find_NN(x)
    return train_labels[idx]

# Success case
print("\n=== Success case ===")
print("NN classification:", NN_classifier(test_data[0]))
print("True label:", test_labels[0])
print("Test image:")
vis_image(0, "test")
print("Nearest neighbor image:")
vis_image(find_NN(test_data[0]), "train")

# Failure case (typical confusion, e.g. 4 vs 9 or similar-looking digits)
print("\n=== Failure case ===")
print("NN classification:", NN_classifier(test_data[39]))
print("True label:", test_labels[39])
print("Test image:")
vis_image(39, "test")
print("Nearest neighbor image:")
vis_image(find_NN(test_data[39]), "train")

#  5. FULL TEST SET EVALUATION
print("\n=== Evaluating on full test set (brute force) ===")
t_before = time.time()
test_predictions = np.array([NN_classifier(test_data[i]) for i in range(len(test_labels))])
t_after = time.time()

error = np.mean(test_predictions != test_labels)
print("Error of nearest neighbor classifier: {:.4f}".format(error))
print("Classification time (seconds): {:.2f}".format(t_after - t_before))

# 6. FASTER NEAREST NEIGHBOR STRUCTURES
print("\n=== Using BallTree (faster exact NN) ===")
t_before = time.time()
ball_tree = BallTree(train_data)
t_after = time.time()
print("Time to build BallTree (seconds): {:.2f}".format(t_after - t_before))

t_before = time.time()
test_neighbors = np.squeeze(ball_tree.query(test_data, k=1, return_distance=False))
ball_tree_predictions = train_labels[test_neighbors]
t_after = time.time()
print("Time to classify test set with BallTree (seconds): {:.2f}".format(t_after - t_before))
print("BallTree matches brute-force predictions?", np.array_equal(test_predictions, ball_tree_predictions))

print("\n=== Using KDTree (faster exact NN) ===")
t_before = time.time()
kd_tree = KDTree(train_data)
t_after = time.time()
print("Time to build KDTree (seconds): {:.2f}".format(t_after - t_before))

t_before = time.time()
test_neighbors = np.squeeze(kd_tree.query(test_data, k=1, return_distance=False))
kd_tree_predictions = train_labels[test_neighbors]
t_after = time.time()
print("Time to classify test set with KDTree (seconds): {:.2f}".format(t_after - t_before))
print("KDTree matches brute-force predictions?", np.array_equal(test_predictions, kd_tree_predictions))

print("\n=== Done! ===")
print("Typical 1-NN error on raw MNIST pixels is around 3-5%. You should see similar here.")