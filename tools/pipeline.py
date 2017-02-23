import numpy as np

from .data_loading import load_images, load_labels, dummy_code
from .kernels import kernel_matrix
from .optimization import find_f
from .prediction import pred
from .process_images import process_images
from .quantization import kmeans
from .submission import labels_to_csv
from .visualization import imshow, dump_as_png

# Data loading
X_train = load_images(type="train")
n_train = X_train.shape[0]
Y_labels_train = load_labels()
Y_train = dummy_code(Y_labels_train)
n_classes = Y_train.shape[1]

indices = np.random.permutation(X_train.shape[0])
training_idx, test_idx = indices[:int(0.9*n_train)], indices[int(0.9*n_train):]

# data exploration:
# - find a way to visualize images
# # matplotlib style :
# imshow(X_train[0])
# imshow(X_train[2])
# # png file dump
# dump_as_png(type="train")
# - check balance of classes

# Visual features
pins_dict_train = pins_generation(training_idx=training_idx)
pins_train = pins_dict_train["pins"]
train_pins = pins_dict_train["train_pins"]
pin_to_im_train = pins_dict_train["pin_to_im"]
pins_mat = np.vstack(train_pins)
visual_features = kmeans(pins_mat, 70)

# Data processing
X_train = process_images(n_train, visual_features, pins_train, pin_to_im_train)
n_train, n_var = X_train.shape

# Data separation

X_sample = X_train[training_idx, :]
n_sample = X_sample.shape[0]
X_test = X_train[test_idx, :]
n_test = X_test.shape[0]
Y_sample = Y_train[training_idx, :]


# Training
kernel_type = "linear"
K_sample = kernel_matrix(X_sample, kernel_type=kernel_type)

classifier_type = "linear regression"
alpha = np.zeros((n_classes, n_sample))
for dig in range(n_classes):
    alpha[dig, :] = find_f(K_sample, Y_sample[:, dig],
                           prob_type=classifier_type, lamb=1.0)

# Evaluation
Y_pred = np.zeros((X_test.shape[0], n_classes))
for dig in range(n_classes):
    Y_pred[:, dig] = pred(X_sample, X_test, alpha[dig, :],
                          kernel_type=kernel_type)


Y_labels_pred = np.argmax(Y_pred, axis=1)
prec = np.mean(Y_labels_pred == Y_labels_train[test_idx])
print("The precision on the test set is of {}".format(prec))

# Prediction
X_eval = load_images(type="test")
n_eval = X_eval.shape[0]

# Visual features for submission
pins_dict_eval = pins_generation(data_type="test")
pins_eval = pins_dict_eval["pins"]
pin_to_im_eval = pins_dict_eval["pin_to_im"]


# Data processing
X_eval = process_images(n_eval, visual_features, pins_eval, pin_to_im_eval)
n_eval, n_var = X_eval.shape

Y_eval = np.zeros((n_eval, n_classes))
for dig in range(n_classes):
    Y_eval[:, dig] = pred(X_sample, X_eval, alpha[dig, :],
                          kernel_type=kernel_type)


Y_labels_eval = np.argmax(Y_eval, axis=1)


# Submission
labels_to_csv(Y_labels_eval)
