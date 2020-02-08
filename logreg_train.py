import sys
import pandas as pd
import numpy as np
import seaborn as sns
import linear_regression as lr
from matplotlib import pyplot as plt

class logisticRegression:
	def __init__(self, data, epochs, learning_rate):
		data.insert(0, "theta_0", np.ones(data.shape[0]))
		self.features = ["Hogwarts House", "theta_0", "Charms", "Herbology", "Ancient Runes", "Astronomy", "Defense Against the Dark Arts"]
		self.data = data.loc[:, self.features]
		self.data = self.data.dropna()
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.houses = self.data.loc[:, "Hogwarts House"]
		self.houses_possibilities = ["Hufflepuff", "Ravenclaw", "Slytherin", "Gryffindor"]

	def precisionMatrix(self, x, y, thetas):
		success = 0
		bach = 1400
		for row in range(bach):
			x_row = np.array(x.iloc[row])
			linear_model = np.dot(thetas, x_row)
			probability = self.sigmoidFunction(linear_model)
			if probability > 0.5:
				probability = 1
			else:
				probability = 0
			if probability == y[row]:
				success += 1
		print((success / bach) * 100)

	def sigmoidFunction(self, z):
		return 1 / (1 + np.exp(-z))

	def standardizeData(self, matrix):
		return (matrix - matrix.mean()) / matrix.std()

	def trainModel(self):
		x = self.standardizeData(self.data.loc[:, self.features[2:]])
		x.insert(0, "theta_0", self.data.loc[:, "theta_0"])
		thetas_per_house = {}
		nb_rows = x.shape[0]
		for house in self.houses_possibilities:
			cost = []
			thetas = np.ones(x.shape[1])
			y = (np.where(self.houses == house, 1, 0))
			for _ in range(self.epochs):
				z = np.dot(x, thetas)
				h = self.sigmoidFunction(z)
				j = (np.dot((-y).T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) / nb_rows
				cost.append(j)
				gradient = np.dot(x.T, (h - y)) / nb_rows
				thetas -= self.learning_rate * gradient
			plt.figure(1)
			plt.plot(cost, label=house)
			self.precisionMatrix(x, y, thetas)
			thetas_per_house[house] = thetas
			print("\x1b[1mProbabilities for\x1b[93m", house, "\x1b[0m\x1b[1m\t->\t\x1b[92mCalculated!", "\x1b[0m")
		plt.legend()
		plt.show()

def logistic_regression():
	file = open("resources/dataset_train.csv")
	df = pd.read_csv(file)
	logisticRegression(df, 5000, 0.01).trainModel()

if __name__ == "__main__":
	logistic_regression()