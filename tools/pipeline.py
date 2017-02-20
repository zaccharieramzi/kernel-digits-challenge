import numpy as np

from tools.submission import labels_to_csv
from tools.kernels import kernel_matrix

# Data loading
X_train = load_images(type="train")  # TODO code load_images
Y_labels_train = load_labels()  # TODO code load_labels
Y_train = dummy_code(Y_labels_train)  # TODO code dummy_code
n_classes = Y_train.shape[1]  # TODO check correctness of order

# TODO data exploration:
# - find a way to visualize images
# - check balance of classes

# Data processing
X_train = process_images(X_train)  # TODO find correct image processing and code it
n_train, n_var = X_train.shape  # TODO check correctness of order

# Data separation
indices = np.random.permutation(X_train.shape[0])
training_idx, test_idx = indices[:int(0.9*n_train)], indices[int(0.9*n_train):]
X_sample = X_train[training_idx, :]
X_test = X_train[test_idx, :]
Y_sample = Y_train[training_idx, :]


# Training
kernel_type = "linear"
K_sample = kernel_matrix(X_sample, kernel_type=kernel_type, **kwargs)

classifier_type = "linear regression"
alpha = np.zeros(X_train.shape)
for dig in range(n_classes):
    alpha[dig, :] = find_f(K_sample, Y_sample[:, dig],
                           prob_type=classifier_type, **kwargs)

# Evaluation
Y_pred = np.zeros((X_test.shape[0], n_classes))
for dig in range(n_classes):
    Y_pred[:, dig] = pred(X_sample, X_test, alpha[dig, :],
                          kernel_type=kernel_type, **kwargs)


Y_labels_pred = np.argmax(Y_pred, axis=1)
prec = np.mean(Y_labels_pred == Y_labels_train[test_idx, :])

# Prediction
X_eval = load_images(type="test")
n_eval = X_eval.shape[0]

X_eval = process_image(X_eval)

Y_eval = np.zeros((n_eval, n_classes))
for dig in range(n_classes):
    Y_eval[:, dig] = pred(X_sample, X_eval, alpha[dig, :],
                          kernel_type=kernel_type, **kwargs)


Y_labels_eval = np.argmax(Y_eval, axis=1)


# Submission
labels_to_csv(Y_labels_eval)
