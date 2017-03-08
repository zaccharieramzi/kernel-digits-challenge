import numpy as np
from tqdm import tqdm

from tools.data_loading import load_images, load_labels, dummy_code
from tools.hog import hog
from tools.kernels import kernel_matrix
from tools.optimization import find_f
from tools.prediction import pred
from tools.submission import labels_to_csv
from tools.visualization import reshape_as_images


# Data loading
print("--- Data Loading ---")
# we load big images
print("Loading training images, this can take a while...")
X_train = load_images(type="train63")
n_train = X_train.shape[0]
# we load labels
print("Loading the labels")
Y_labels_train = load_labels()
# we dummy code the labels
print("Dummy coding the labels")
Y_train = dummy_code(Y_labels_train)
n_classes = Y_train.shape[1]
# we normalize the images
X_train = X_train - X_train.min(axis=0)
X_train = X_train / X_train.max(axis=0)
# we reshape the images
image_list = reshape_as_images(X_train)
# Visual features parameters
print("--- Visual features ---")
filter_sigma = 0.1
filter_shape = 5
hog_cell_size = 7
disc_grid = 16
n_images = image_list.shape[0]
# we compute the hogs for each image
hog_list = []
for i in tqdm(range(n_images), desc="Training images HoG computation"):
    image = image_list[i, :, :, :]
    hog_list.append(
        hog(
            image,
            filter_sigma=filter_sigma,
            filter_shape=filter_shape,
            hog_cell_size=hog_cell_size,
            disc_grid=disc_grid,
            color_grad=True))
n_features = hog_list[0].size
X_hog = np.array(hog_list).reshape((n_images, n_features))
# Data separation
print("--- Data separation ---")
training_idx = []
test_idx = []
train_frac = 1  # 1 if you we to make a submission
if train_frac == 1:
    print("Since train_frac is set to 1, there will be no test set")
for dig in tqdm(range(n_classes), desc="Separating train from test set"):
    # we do the data separation for each class in order to keep the correct
    # distribution
    selected_indices = np.random.permutation(np.sum(Y_labels_train == dig))
    n_class = len(selected_indices)
    class_indices = np.where(Y_labels_train == dig)[0]
    training_idx += list(
        map(int, class_indices[selected_indices[:int(train_frac * n_class)]]))
    test_idx += list(
        map(int, class_indices[selected_indices[int(train_frac * n_class):]]))
# training set within the training set
X_sample = X_hog[training_idx, :]
n_sample = X_sample.shape[0]
Y_sample = Y_train[training_idx, :]
Y_labels_sample = Y_labels_train[training_idx]
# test set within the training set, useful to do grid search
X_test = X_hog[test_idx, :]
n_test = X_test.shape[0]
Y_labels_test = Y_labels_train[test_idx]
# Training
print("--- Beginning training ---")
# Classifier choice
classifier_type = "fast_svm"
n_iter = 60000
# Kernel choice
kernel_type = "rbf"
sigma = 20
lamb = 1e-8
# Kernel computation
print("Computing kernel")
K_sample = kernel_matrix(X_sample, kernel_type=kernel_type, sigma=sigma)
# Optimization
alpha = np.zeros((n_classes, n_sample))
for dig in tqdm(range(n_classes), desc="Coordinate ascent for each class"):
    alpha[dig, :] = find_f(
        K_sample, Y_sample[:, dig],
        prob_type=classifier_type, lamb=lamb, n_iter=n_iter)
# Submission images loading
print("--- Submission part ---")
print("Loading submission data, this may take a while...")
X_eval = load_images(type="test63")
n_eval = X_eval.shape[0]
X_eval = X_eval - X_eval.min(axis=0)
X_eval = X_eval / X_eval.max(axis=0)
# Submission images reshaping
image_list_eval = reshape_as_images(X_eval)
# Hogs computation for visual images
print("--- Visual features ---")
hog_list_eval = []
for i in tqdm(range(n_eval), desc="Submission images HoG computation"):
    image = image_list_eval[i, :, :, :]
    hog_list_eval.append(
        hog(
            image,
            filter_sigma=filter_sigma,
            filter_shape=filter_shape,
            hog_cell_size=hog_cell_size,
            disc_grid=disc_grid,
            color_grad=True))
n_features = hog_list_eval[0].size
X_hog_eval = np.array(hog_list_eval).reshape((n_eval, n_features))
# Prediction for submission images
print("--- Prediction for submission images ---")
Y_eval = pred(
    X_sample, X_hog_eval, alpha, kernel_type=kernel_type, sigma=sigma)
Y_labels_eval = np.argmax(Y_eval, axis=1)
# Submission file
labels_to_csv(Y_labels_eval, file_name="Yte.csv")
