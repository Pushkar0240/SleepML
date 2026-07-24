import numpy as np

from src.training.balance import DatasetBalancer

X = np.random.randn(200, 2, 6000)

y = np.array(

    [0] * 170 +

    [1] * 30

)

balancer = DatasetBalancer()

balancer.summary(y)

weights = balancer.class_weights(y)

print("\nClass Weights")

print(weights)

X2, y2 = balancer.oversample(X, y)

print("\nAfter Oversampling")

balancer.summary(y2)

X3, y3 = balancer.undersample(X, y)

print("\nAfter Undersampling")

balancer.summary(y3)
