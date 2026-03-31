"""
Class for a classification algorithm.
"""

import numpy as np
SUPPORTED_CLASSIFIERS = (
    "naive_bayes",
    "logistic_regression",
    "decision_tree",
)


class Classifier:

    def __init__(self, classifier_type, **kwargs):
        """
        Initializer. Classifier_type should be a string which refers
        to the specific algorithm the current classifier is using.
        Use keyword arguments to store parameters
        specific to the algorithm being used. E.g. if you were
        making a neural net with 30 input nodes, hidden layer with
        10 units, and 3 output nodes your initalization might look
        something like this:

        neural_net = Classifier(weights = [], num_input=30, num_hidden=10, num_output=3)

        Here I have the weight matrices being stored in a list called weights (initially empty).
        """
        if classifier_type not in SUPPORTED_CLASSIFIERS:
            raise ValueError(
                "Unsupported classifier_type '{}'. Expected one of: {}".format(
                    classifier_type, ", ".join(SUPPORTED_CLASSIFIERS)
                )
            )

        self.classifier_type = classifier_type
        self.params = dict(kwargs)
        """
        The kwargs you inputted just becomes a dictionary, so we can save
        that dictionary to be used in other methods.
        """
        self.model_state = {}

    def train(self, training_data):
        """
        Data should be nx(m+1) numpy matrix where n is the 
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        I recommend implementing the specific algorithms in a
        seperate module and then determining which method to call
        based on classifier_type. E.g. if you had a module called
        neural_nets:

        if self.classifier_type == 'neural_net':
            import neural_nets
            neural_nets.train_neural_net(self.params, training_data)

        Note that your training algorithms should be modifying the parameters
        so make sure that your methods are actually modifying self.params

        You should print the accuracy, precision, and recall on the training data.
        """
        self._validate_dataset(training_data, dataset_name="training_data")
        self._store_dataset_metadata(training_data)

        if self.classifier_type == "naive_bayes":
            self._train_naive_bayes(training_data)
        elif self.classifier_type == "logistic_regression":
            self._train_logistic_regression(training_data)
        elif self.classifier_type == "decision_tree":
            self._train_decision_tree(training_data)

        y_true, y_pred = self._predict_dataset(training_data)
        metrics = self._compute_metrics(y_true, y_pred)
        self._print_metrics(metrics, split_name="training")
        return metrics

    def predict(self, data):
        """
        Predict class of a single data vector
        Data should be 1x(m+1) numpy matrix where m is the number of features
        (recall that the first element of the vector is the label).

        I recommend implementing the specific algorithms in a
        seperate module and then determining which method to call
        based on classifier_type.

        This method should return the predicted label.
        """
        sample = self._validate_sample(data)

        if self.classifier_type == "naive_bayes":
            return self._predict_naive_bayes(sample)
        if self.classifier_type == "logistic_regression":
            return self._predict_logistic_regression(sample)
        if self.classifier_type == "decision_tree":
            return self._predict_decision_tree(sample)

        raise ValueError("Unsupported classifier_type '{}'".format(self.classifier_type))

    def test(self, test_data):
        """
        Data should be nx(m+1) numpy matrix where n is the 
        number of examples and m is the number of features
        (recall that the first element of the vector is the label).

        You should print the accuracy, precision, and recall on the test data.
        """
        self._validate_dataset(test_data, dataset_name="test_data")
        y_true, y_pred = self._predict_dataset(test_data)
        metrics = self._compute_metrics(y_true, y_pred)
        self._print_metrics(metrics, split_name="test")
        return metrics

    def _validate_dataset(self, data, dataset_name):
        if not isinstance(data, np.ndarray):
            raise TypeError("{} must be a numpy array".format(dataset_name))
        if data.ndim != 2:
            raise ValueError("{} must be a 2D numpy array".format(dataset_name))
        if data.shape[1] < 2:
            raise ValueError("{} must contain a label and at least one feature".format(dataset_name))

    def _validate_sample(self, data):
        sample = np.asarray(data)
        if sample.ndim != 1:
            raise ValueError("Each sample must be a 1D numpy vector")
        if sample.shape[0] < 2:
            raise ValueError("Each sample must contain a label and at least one feature")
        return sample

    def _store_dataset_metadata(self, training_data):
        self.model_state["num_features"] = training_data.shape[1] - 1
        self.model_state["classes"] = np.unique(training_data[:, 0])

    def _predict_dataset(self, data):
        y_true = []
        y_pred = []
        for row in data:
            y_true.append(row[0])
            y_pred.append(self.predict(row))
        return np.array(y_true), np.array(y_pred)

    def _compute_metrics(self, y_true, y_pred):
        """
        Placeholder for shared metric computation.
        Implement accuracy / precision / recall here later.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        labels = np.unique(np.concatenate((y_true, y_pred)))
        accuracy = float(np.mean(y_true == y_pred))

        precision_scores = []
        recall_scores = []

        for label in labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            false_positive = np.sum((y_true != label) & (y_pred == label))
            false_negative = np.sum((y_true == label) & (y_pred != label))

            if true_positive + false_positive == 0:
                precision_scores.append(0.0)
            else:
                precision_scores.append(
                    float(true_positive) / float(true_positive + false_positive)
                )

            if true_positive + false_negative == 0:
                recall_scores.append(0.0)
            else:
                recall_scores.append(
                    float(true_positive) / float(true_positive + false_negative)
                )

        metrics = {
            "accuracy": accuracy,
            "precision": float(np.mean(precision_scores)),
            "recall": float(np.mean(recall_scores)),
        }
        return metrics

    def _print_metrics(self, metrics, split_name):
        """
        Placeholder for consistent train/test metric output formatting.
        """
        print("{} metrics:".format(split_name.capitalize()))
        print("Accuracy: {:.4f}".format(metrics["accuracy"]))
        print("Precision: {:.4f}".format(metrics["precision"]))
        print("Recall: {:.4f}".format(metrics["recall"]))

    def _train_naive_bayes(self, training_data):
        labels = training_data[:, 0]
        features = training_data[:, 1:]
        classes = self.model_state["classes"]

        #smoothing 避免机率变成 0
        smoothing = 1

        #判断资料是连续还是离散
        num_unique = np.mean([len(np.unique(features[:, j])) for j in range(features.shape[1])])
        is_continuous = num_unique > 10
        self.model_state["is_continuous"] = is_continuous

        #算每个 class 的机率 P(y)
        class_priors = {}
        for c in classes:
            class_priors[c] = np.sum(labels == c) / len(labels)
        self.model_state["class_priors"] = class_priors

        if is_continuous:
            #如果是连续资料，存每個 class 每個 feature 的 mean 和 std
            means = {}
            stds = {}
            for c in classes:
                class_features = features[labels == c]
                means[c] = np.mean(class_features, axis=0)
                #算标准差，代表分布的范围，加 1e-9 避免除以 0
                stds[c] = np.std(class_features, axis=0) + 1e-9
            self.model_state["means"] = means
            self.model_state["stds"] = stds

        else:
            #如果是离散资料，记录每个 feature 可能出现的值
            feature_values = {}
            for j in range(features.shape[1]):
                feature_values[j] = np.unique(features[:, j])
            self.model_state["feature_values"] = feature_values

            #存条件机率 P(x|y)
            cond_probs = {}
            for c in classes:
                cond_probs[c] = {}
                class_features = features[labels == c]
                for j in range(features.shape[1]):
                    cond_probs[c][j] = {}
                    values = feature_values[j]
                    for v in values:
                        count = np.sum(class_features[:, j] == v)
                        total = len(class_features)
                        #加 smoothing，避免机率为 0
                        prob = (count + smoothing) / (total + smoothing * len(values))
                        cond_probs[c][j][v] = prob
            self.model_state["cond_probs"] = cond_probs


    def _predict_naive_bayes(self, sample):
        features = sample[1:]
        classes = self.model_state["classes"]
        class_priors = self.model_state["class_priors"]

        #看 training 时判断是连续还是离散
        is_continuous = self.model_state["is_continuous"]

        #用来记录最佳答案
        best_class = None
        best_log_prob = -np.inf

        for c in classes:
            log_prob = np.log(class_priors[c])

            if is_continuous:
                #如果是连续资料，用常态分佈算机率
                mu = self.model_state["means"][c]
                sigma = self.model_state["stds"][c]
                log_prob += np.sum(
                    -0.5 * np.log(2 * np.pi * sigma ** 2)
                    - ((features - mu) ** 2) / (2 * sigma ** 2)
                )
            else:
                #如果是离散资料，就直接查表
                cond_probs = self.model_state["cond_probs"]
                #如果这个值在 training 有出现过，就加上对应机率
                for j, v in enumerate(features):
                    if v in cond_probs[c][j]:
                        log_prob += np.log(cond_probs[c][j][v])

            #找出机率最大的 class
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        return best_class

    def _train_logistic_regression(self, training_data):
        """
        Train the Logistic Regression model and store learned parameters in self.model_state.
        """
        raise NotImplementedError("Logistic Regression training has not been implemented yet.")

    def _predict_logistic_regression(self, sample):
        """
        Predict a label for one sample using the trained Logistic Regression model.
        """
        raise NotImplementedError("Logistic Regression prediction has not been implemented yet.")

    def _train_decision_tree(self, training_data):
        """
        Train the Decision Tree model and store the tree in self.model_state.
        """
        raise NotImplementedError("Decision Tree training has not been implemented yet.")

    def _predict_decision_tree(self, sample):
        """
        Predict a label for one sample using the trained Decision Tree model.
        """
        raise NotImplementedError("Decision Tree prediction has not been implemented yet.")
