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
        """
        Train the Naive Bayes model and store learned parameters in self.model_state.
        """
        raise NotImplementedError("Naive Bayes training has not been implemented yet.")

    def _predict_naive_bayes(self, sample):
        """
        Predict a label for one sample using the trained Naive Bayes model.
        """
        raise NotImplementedError("Naive Bayes prediction has not been implemented yet.")

    def _train_logistic_regression(self, training_data):
        """
        Train the Logistic Regression model and store learned parameters in self.model_state.
        """
        x_raw = np.asarray(training_data[:, 1:], dtype=float)
        y_raw = np.asarray(training_data[:, 0])

        classes = np.array(self.model_state["classes"])
        n_classes = len(classes)

        learning_rate = self.params.get("learning_rate")
        if learning_rate is None:
            learning_rate = 0.1

        # 没给参数用默认值
        num_iterations = self.params.get("num_iterations")
        if num_iterations is None:
            num_iterations = 3000

        reg_strength = self.params.get("reg_strength")
        if reg_strength is None:
            reg_strength = 0.001

        categorical_unique_threshold = self.params.get("categorical_unique_threshold")
        if categorical_unique_threshold is None:
            categorical_unique_threshold = 20

        # 记录处理方式
        feature_info = []
        transformed_parts = []

        # 判断资料要怎么处理
        for j in range(x_raw.shape[1]):
            col = x_raw[:, j]
            unique_vals = np.unique(col)

            is_integer_like = np.allclose(col, np.round(col))
            is_categorical = is_integer_like and len(unique_vals) <= categorical_unique_threshold

            if is_categorical:
                feature_info.append({
                    "type": "categorical",
                    "categories": unique_vals
                })

                one_hot = np.zeros((x_raw.shape[0], len(unique_vals)), dtype=float)
                for idx, cat in enumerate(unique_vals):
                    one_hot[:, idx] = (col == cat).astype(float)
                transformed_parts.append(one_hot)
            else:

                mean = float(np.mean(col))
                std = float(np.std(col))
                if std < 1e-12:
                    std = 1.0

                feature_info.append({
                    "type": "continuous",
                    "mean": mean,
                    "std": std
                })

                values = ((col.reshape(-1, 1) - mean) / std).astype(float)
                transformed_parts.append(values)

        # 把所有处理好的合在一起
        x = np.hstack(transformed_parts)

        bias = np.ones((x.shape[0], 1), dtype=float)
        x = np.hstack((bias, x))

        # predict用同样规则处理资料
        self.model_state["logreg_feature_info"] = feature_info

        class_to_index = {label: idx for idx, label in enumerate(classes)}
        y_indices = np.array([class_to_index[label] for label in y_raw], dtype=int)

        y = np.zeros((x.shape[0], n_classes), dtype=float)
        y[np.arange(x.shape[0]), y_indices] = 1.0

        w = np.zeros((x.shape[1], n_classes), dtype=float)

        for _ in range(int(num_iterations)):
            scores = np.dot(x, w)

            shifted = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            grad = np.dot(x.T, (probs - y)) / float(x.shape[0])

            # 加L2 regularization
            reg_term = (reg_strength / float(x.shape[0])) * w
            reg_term[0, :] = 0.0
            grad += reg_term

            w -= learning_rate * grad

        self.model_state["logreg_weights"] = w

    def _predict_logistic_regression(self, sample):
        """
        Predict a label for one sample using the trained Logistic Regression model.
        """
        if "logreg_weights" not in self.model_state:
            raise ValueError("Logistic Regression model has not been trained yet.")

        x_raw = np.asarray(sample[1:], dtype=float)
        feature_info = self.model_state["logreg_feature_info"]

        transformed_parts = []
        # 用training时同样的方法处理资料
        for j, info in enumerate(feature_info):
            value = x_raw[j]

            if info["type"] == "categorical":

                categories = info["categories"]
                one_hot = np.zeros((1, len(categories)), dtype=float)

                for idx, cat in enumerate(categories):
                    if value == cat:
                        one_hot[0, idx] = 1.0

                transformed_parts.append(one_hot)
            else:

                val = np.array([[float(value)]], dtype=float)
                val = (val - info["mean"]) / info["std"]
                transformed_parts.append(val)

        # 把所有处理好的合在一起
        x = np.hstack(transformed_parts)

        bias = np.ones((1, 1), dtype=float)
        x = np.hstack((bias, x))

        # 算每个 class 的分数
        w = self.model_state["logreg_weights"]
        scores = np.dot(x, w)

        # 选分数最高的 class
        pred_index = int(np.argmax(scores, axis=1)[0])

        return self.model_state["classes"][pred_index]

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
