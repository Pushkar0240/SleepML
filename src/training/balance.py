"""
SleepTransitionCNN
==================

Dataset Balancer

Supports

1. Undersampling

2. Oversampling

3. Class Weight Calculation


"""

from __future__ import annotations

from collections import Counter

import numpy as np

from sklearn.utils.class_weight import compute_class_weight


class DatasetBalancer:

    # -------------------------------------------------
    # Class Weights
    # -------------------------------------------------

    def class_weights(self, labels):

        classes = np.unique(labels)

        weights = compute_class_weight(

            class_weight="balanced",

            classes=classes,

            y=labels

        )

        return {

            int(c): float(w)

            for c, w in zip(classes, weights)

        }

    # -------------------------------------------------
    # Oversample
    # -------------------------------------------------

    def oversample(self, X, y):

        counter = Counter(y)

        maximum = max(counter.values())

        X_new = []

        y_new = []

        for label in counter:

            idx = np.where(y == label)[0]

            repeat = maximum - len(idx)

            X_new.extend(X[idx])

            y_new.extend(y[idx])

            if repeat > 0:

                chosen = np.random.choice(

                    idx,

                    repeat,

                    replace=True

                )

                X_new.extend(X[chosen])

                y_new.extend(y[chosen])

        return np.asarray(X_new), np.asarray(y_new)

    # -------------------------------------------------
    # Undersample
    # -------------------------------------------------

    def undersample(self, X, y):

        counter = Counter(y)

        minimum = min(counter.values())

        X_new = []

        y_new = []

        for label in counter:

            idx = np.where(y == label)[0]

            chosen = np.random.choice(

                idx,

                minimum,

                replace=False

            )

            X_new.extend(X[chosen])

            y_new.extend(y[chosen])

        return np.asarray(X_new), np.asarray(y_new)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def summary(self, labels):

        print("\nClass Distribution")

        print("-" * 40)

        counter = Counter(labels)

        for label, count in counter.items():

            print(f"{label:10} {count}")

        print("-" * 40)
