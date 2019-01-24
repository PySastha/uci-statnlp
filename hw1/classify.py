#!/bin/python

def train_classifier(X, y):
	from sklearn.linear_model import LogisticRegression
	C_value = 0.65
	cls = LogisticRegression(C=C_value)
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls):
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy", acc)
