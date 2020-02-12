import pandas as pd
import numpy as np
import math

# ----------------------------------------------------------------------------
# 							SIMPLE LINEAR REGRESSION CLASS
# ----------------------------------------------------------------------------

class linearRegressionGradientDescent:
	def __init__(self, x, y, learning_rate, epochs):
		self.x = x
		self.y = y
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.slope = 0
		self.intercept = 0

	def getSlope(self):
		return self.slope

	def getIntercept(self):
		return self.intercept

	def getResidualSquareSum(self):
		residual_square_sum = 0
		for X, Y in zip(self.x, self.y):
			residual_square_sum += (self.estimateDependentVariable(self.slope, self.intercept, X) - Y) ** 2
		return residual_square_sum / len(self.x)

	def setSlope(self, slope):
		self.slope = slope

	def setIntercept(self, intercept):
		self.intercept = intercept

	def estimateDependentVariable(self, slope, intercept, x):
		return slope * x + intercept

	def calculateNewSlope(self):
		nb_observations = len(self.x)
		new_slope = 0
		for observation_x_value, observation_y_value in zip(self.x, self.y):
			new_slope += (self.estimateDependentVariable(self.slope, self.intercept, observation_x_value) - observation_y_value) * observation_x_value
		new_slope = self.slope - new_slope * (self.learning_rate / nb_observations)
		return new_slope

	def calculateNewIntercept(self):
		nb_observations = len(self.x)
		new_intercept = 0
		for observation_x_value, observation_y_value in zip(self.x, self.y):
			new_intercept += (self.estimateDependentVariable(self.slope, self.intercept, observation_x_value) - observation_y_value)
		new_intercept = self.slope - new_intercept * (self.learning_rate / nb_observations)
		return new_intercept

	def calculateWeights(self):
		tmp_slope = 0
		tmp_intercept = 0
		for _ in range(self.epochs):
			tmp_slope = self.calculateNewSlope()
			tmp_intercept = self.calculateNewIntercept()
			self.setSlope(tmp_slope)
			self.setIntercept(tmp_intercept)

# ----------------------------------------------------------------------------
# 							LOGISTIC REGRESSION CLASS
# ----------------------------------------------------------------------------

# class logisticRegression():
# 	def __init__(self, epochs, learning_rate):

# ----------------------------------------------------------------------------
# 							NORMALIZATION CLASS
# ----------------------------------------------------------------------------

class normalizeClass:
	def __init__(self, data_set, min_value, max_value):
		"""This class is used to normalize sets of data, or single data (inheriting from given data-set.)

		Args:
			data-set: type = LIST | description = Data-set given as a list.
			min_value: type = FLOAT64 | description = Minimum value of given data-set
			max_value: type = FLOAT64 | description = Maximum value of given data-set

		"""
		self.data_set = data_set
		self.min_value = min_value
		self.max_value = max_value

	def getNormalizedDataSet(self):
		return self.data_set

	def normalizeDataSet(self):
		for i in range(0, len(self.data_set)):
			self.data_set[i] = float((self.data_set[i] - self.min_value) / (self.max_value - self.min_value))

	def normalizeSingleData(self, data):
		return float((data - self.min_value) / (self.max_value - self.min_value))