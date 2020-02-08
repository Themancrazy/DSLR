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

class logisticRegressionGradientDescent():
	def __init__(self, csv_file_name, class_category_name, classes_possibilities, feature_names, learning_rate, epochs):
		self.file = csv_file_name
		self.data_frame = self.assignDataFrame()
		self.data_frame.dropna(inplace=True)
		self.feature_names = self.assignFeatures(feature_names)
		self.classes_possibilities = classes_possibilities
		self.class_category_name = class_category_name
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.weights = []
		self.assignWeights()
		self.theta_0 = 1

	def assignWeights(self):
		if len(self.weights) != len(self.feature_names):
			self.weights = []
			for _ in self.classes_possibilities:
				tmp_weights = []
				for _ in self.feature_names:
					tmp_weights.append(0)
				self.weights.append(tmp_weights)

	def assignDataFrame(self):
		file = open(self.file)
		return pd.read_csv(file)

	def assignFeatures(self, featureNames):
		newFeature = []
		for f in featureNames:
			newFeature.append(np.array(self.data_frame[f]))
		return newFeature

	def getWeights(self):
		return self.weights

	def hypothesesisFunction(self, row_number, class_number):
		row_vector_x = []
		for f in self.feature_names:
			row_vector_x.append(float(f[row_number]))
		return self.theta_0 + np.dot(self.weights[class_number], row_vector_x)

	def sigmoidFunction(self, row_number, class_number):
		e = 2.71828
		return 1 / (1 + e ** (self.hypothesesisFunction(row_number, class_number)))

	def estimateProbability(self, row_number, class_number):
		return self.sigmoidFunction(row_number, class_number)

	def lossFunction(self, target_class_name, target_class_index):
		current_y = 0
		error_sum = 0
		nb_samples = len(self.feature_names[0])
		for row in range(nb_samples):
			# if self.data_frame[self.class_category_name][row] == target_class_name:
			if self.data_frame.loc[row, 1] == target_class_name:
				current_y = 1
			else:
				current_y = 0
			error_sum += ((-1 * current_y) * math.log(self.estimateProbability(row, target_class_index))) + ((1 - current_y) * math.log(1 - self.estimateProbability(row, target_class_index)))
		error_sum = error_sum / nb_samples

	def sumFunction(self, target_class_name, target_class_index, weight_index):
		current_y = 0
		error_sum = 0
		nb_samples = len(self.feature_names[0])
		for row in range(nb_samples):
			print("ROW:", row, "\tNB_SAMPLE:", nb_samples)
			print("House of row:", np.array(self.data_frame[self.class_category_name][row]), "\tCurrent House:", target_class_name)
			print("error_sum:", error_sum)
			# if self.data_frame[self.class_category_name][row] == target_class_name:
			if self.data_frame.loc[row, 1] == target_class_name:
				current_y = 1
			else:
				current_y = 0
			error_sum += (self.estimateProbability(row, target_class_index) - current_y) * self.feature_names[weight_index][row]
		error_sum = error_sum

	def trainModel(self):
		for index_class, name_class in zip(range(len(self.classes_possibilities)), self.classes_possibilities): # for all class
			print(index_class)
			for _ in range(self.epochs):
				tmp_weights_list = []
				for index_weight, weight in zip(range(len(self.weights[index_class])), self.weights[index_class]): # for each weight
					tmp_weights_list.append(weight - (self.learning_rate * self.sumFunction(name_class, index_class, index_weight))) # train all weights in this loop
				self.weights[index_class] = tmp_weights_list
			print("\x1b[91mweights for", name_class, "are: \x1b[0m", self.weights[index_class])

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