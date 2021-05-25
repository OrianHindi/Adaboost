from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import itertools
import numpy as np
from Point import Point


class Rule:
    def __init__(self, p1, p2, side, error, prediction):
        self.p1 = p1
        self.p2 = p2
        self.side = side
        self.error = error
        self.predictions = prediction

# compute weighted error of a rule
def compute_weighted_error(w, predicted, ground_truth):
    error = 0
    for i in range(len(predicted)):
        if predicted[i] != ground_truth[i]:
            a = 1
        else:
            a = 0
        error += w[i] * a

    return error


# predict point, side represent the value the rule gave to points who above the line
def pred_point(p1, p2, p3, side):
    if is_above(p1, p2, p3) < 0:
        pred = side
    else:
        pred = -side

    return pred


def is_above(p1, p2, p3):
    return (p3.get_x() - p1.get_x()) * (p2.get_y() - p1.get_y()) - (p3.get_y() - p1.get_y()) * (p2.get_x() - p1.get_x())


# predict all train points with given rule(line), truth parameter is to indicate if the rule see -1 fromm all points
# above it or 1
def get_predicted_points(train, p1, p2, truth):
    pred = np.zeros_like(train)
    if truth:
        for i in range(len(train)):
            if is_above(p1, p2, train[i]) < 0:
                pred[i] = 1
            else:
                pred[i] = -1
    else:
        for i in range(len(train)):
            if is_above(p1, p2, train[i]) < 0:
                pred[i] = -1
            else:
                pred[i] = 1
    return pred


# for all possible lines computer their predict the points with each line and compute wighted error
# of all those possible lines take the line with lowest error,save both points of that line, which val the line see
#  above it and the predicted values.
#
def get_ht_line(train, w, truth):
    error = np.inf
    rule = Rule(None, None, 0, 0, None)
    for p in itertools.combinations(train, 2):
        p1 = p[0]
        p2 = p[1]
        if p1.get_y() > p2.get_y():
            p1 = p[1]
            p2 = p[0]
        pred = get_predicted_points(train, p1, p2, True)
        ht = compute_weighted_error(w, pred, truth)
        if ht < error:
            rule.p1 = p1
            rule.p2 = p2
            rule.side = 1
            rule.error = ht
            rule.predictions = pred
            error = ht

    for p in itertools.combinations(train, 2):
        p1 = p[0]
        p2 = p[1]
        if p1.get_y() > p2.get_y():
            p1 = p[1]
            p2 = p[0]
        pred = get_predicted_points(train, p1, p2, False)
        ht = compute_weighted_error(w, pred, truth)
        if ht < error:
            rule.p1 = p1
            rule.p2 = p2
            rule.side = -1
            rule.error = ht
            rule.predictions = pred
            error = ht

    return rule


# read all the dataset
def read_dataset():
    x = []
    y = []
    file = open("rectangle.txt", "r")
    for line in file:
        x1, y2, val = line.split()
        p = Point(float(x1), float(y2))
        x.append(p)
        y.append(float(val))
    return x, y


def adaboost(rounds):
    X, Y = read_dataset()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    w = np.ones(len(X_train)) / len(X_train)
    test_acc = np.zeros(8)
    train_acc = np.zeros(8)
    Hs = []
    As = []
    for i in range(rounds):
        total_sum = 0
        rule = get_ht_line(X_train, w, Y_train)
        Hs.append(rule)
        at = 0.5 * np.log((1-rule.error) / rule.error)
        As.append(at)
        for x in range(len(X_test)):
            if sign(X_test[x], Hs, As, i,False, x) != Y_test[x]:
                test_acc[i] += 1

        for x in range(len(X_train)):
            if sign(X_train[x], Hs, As, i, False, x) != Y_train[x]:
                train_acc[i] += 1

        for j in range(len(w)):
            w[j] *= np.exp(-at * rule.predictions[j] * Y_train[j])
            total_sum += w[j]

        w = w / total_sum
    test_acc /= len(X_test)
    train_acc /= len(X_train)
    return train_acc, test_acc


# Sign function, is train is to represent if we check on train if yes we got the rule predicted values just calcualte.
# if we came from test we need to predict the point value
def sign(p, HS, AS, len,is_train, idx):
    sum = 0
    if is_train:
        for i in range(len):
            sum += AS[i] * HS[i].predictions[idx]
    else:
        for i in range(len):
            sum += AS[i] * pred_point(HS[i].p1, HS[i].p2, p, HS[i].side)
    if sum > 0:
        return 1
    else:
        return -1


if __name__ == '__main__':
    train_acc = np.zeros(8)
    test_acc = np.zeros(8)
    for j in range(100):
        temp1, temp2 = adaboost(8)
        test_acc += temp2
        train_acc += temp1
        print("iteration ", j)
    test_acc /= 100.0
    train_acc /= 100.0

    print("Train:")
    for i, j in enumerate(train_acc):
        print(str(i), ":", j)
    print("Test:")
    for i, j in enumerate(test_acc):
        print(str(i), ":", j)




