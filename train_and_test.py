import sys

import load_data
from classifier import Classifier, SUPPORTED_CLASSIFIERS

"""
This is the main python method that will be run.
You should determine what sort of command line arguments
you want to use. But in this module you will need to 
1) initialize your classifier and its params 
2) load training/test data 
3) train the algorithm
4) test it and output the desired statistics.
"""


SUPPORTED_DATASETS = (
    "iris",
    "congress",
    "monks1",
    "monks2",
    "monks3",
)


def parse_args(argv):
    if len(argv) < 3:
        raise ValueError(
            "Usage: python train_and_test.py <classifier_type> <dataset_name>"
        )

    classifier_type = argv[1]
    dataset_name = argv[2]

    if classifier_type not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            "Unsupported classifier_type '{}'. Expected one of: {}".format(
                classifier_type, ", ".join(SUPPORTED_CLASSIFIERS)
            )
        )

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            "Unsupported dataset_name '{}'. Expected one of: {}".format(
                dataset_name, ", ".join(SUPPORTED_DATASETS)
            )
        )

    return {
        "classifier_type": classifier_type,
        "dataset_name": dataset_name,
    }


def build_classifier_config(classifier_type):
    configs = {
        "naive_bayes": {
            "smoothing": 1.0,
        },
        "logistic_regression": {
            "learning_rate": None,
            "num_iterations": None,
        },
        "decision_tree": {
            "split_criterion": None,
            "max_depth": None,
        },
    }
    return configs[classifier_type]


def load_selected_dataset(dataset_name):
    if dataset_name == "iris":
        return load_data.load_iris(training_ratio=0.7)
    if dataset_name == "congress":
        return load_data.load_congress_data(training_ratio=0.7)
    if dataset_name == "monks1":
        return load_data.load_monks(1)
    if dataset_name == "monks2":
        return load_data.load_monks(2)
    if dataset_name == "monks3":
        return load_data.load_monks(3)

    raise ValueError("Unsupported dataset_name '{}'".format(dataset_name))


def run_experiment(classifier_type, dataset_name):
    classifier_config = build_classifier_config(classifier_type)
    training_data, test_data = load_selected_dataset(dataset_name)

    classifier = Classifier(classifier_type, **classifier_config)
    training_metrics = classifier.train(training_data)
    test_metrics = classifier.test(test_data)

    return {
        "classifier_type": classifier_type,
        "dataset_name": dataset_name,
        "training_metrics": training_metrics,
        "test_metrics": test_metrics,
    }


def main(argv=None):
    if argv is None:
        argv = sys.argv

    config = parse_args(argv)
    run_experiment(
        classifier_type=config["classifier_type"],
        dataset_name=config["dataset_name"],
    )


if __name__ == "__main__":
    main()
