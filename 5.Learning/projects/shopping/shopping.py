import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        evidencekey = next(reader)
        month = {"Jan": 1, "Feb": 2, "Mar": 3, "Ari": 4, "May": 5, "June": 6, "Jul": 7, "Aug": 8,
                 "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        intEvidence = ["Administrative", "Informational", "ProductRelated", "OperatingSystems", "Browser",
                       "Region", "TrafficType"]
        floatEvidence = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
                         "BounceRates", "ExitRates", "PageValues", "SpecialDay"]

        evidence = []
        label = []
        for raw in reader:
            rawEvidence = []
            for i in range(len(raw)):
                if evidencekey[i] in intEvidence:
                    rawEvidence.append(int(raw[i]))
                elif evidencekey[i] in floatEvidence:
                    rawEvidence.append(float(raw[i]))
                elif evidencekey[i] == "Month":
                    rawEvidence.append(month[raw[i]])
                elif evidencekey[i] == "VisitorType":
                    rawEvidence.append(1 if raw[i] == "Returning_Visitor" else 0)
                elif evidencekey[i] == "Weekend":
                    rawEvidence.append(1 if raw[i] == "TRUE" else 0)
                else:
                    evidence.append(rawEvidence)
                    label.append(1 if raw[i] == "TRUE" else 0)
        return evidence, label


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    return KNeighborsClassifier(n_neighbors=1).fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive_amount = 0
    negative_amount = 0
    true_positive_amount = 0
    false_negative_amount = 0
    for result in zip(labels, predictions):
        if result[0] == 1:
            positive_amount += 1
            if result[1] == 1:
                true_positive_amount += 1
        else:
            negative_amount += 1
            if result[1] == 0:
                false_negative_amount += 1

    return true_positive_amount / positive_amount, false_negative_amount / negative_amount


if __name__ == "__main__":
    main()
