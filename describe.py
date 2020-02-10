# This program will take a dataset as a parameter.
# All it has to do is to display information for all numerical features in a dataset.

import pandas as pd
import numpy as np
import re
import sys
from math import sqrt
from math import floor
from math import ceil

class dataAnalysis:
	def dataAnalysisCount(self):
		return len(self.column)

	def dataAnalysisMean(self):
		total = 0
		for element in self.column:
			total += element
		total /= self.count
		return total

	def dataAnalysisStd(self):
		var = 0
		for nb in self.column:
			var += ((nb - self.mean) ** 2)
		std = sqrt(var / self.count)
		return std

	def dataAnalysisMin(self):
		minimumNb = self.column[0]
		for nb in self.column:
			if nb < minimumNb:
				minimumNb = nb
		return minimumNb

	def dataAnalysisQuant(self, percent):
		len_col = len(self.column) - 1
		column_as_array = np.array(self.column)
		index = len_col * percent
		floored_index = floor(index)
		ceiled_index = ceil(index)
		if floored_index == ceiled_index:
			return column_as_array[floored_index]
		delta_0 = column_as_array[floored_index] * (ceiled_index - index)
		delta_1 = column_as_array[ceiled_index] * (index - floored_index)
		return delta_0 + delta_1

	def dataAnalysisMax(self):
		maximumNb = self.column[0]
		for nb in self.column:
			if nb > maximumNb:
				maximumNb = nb
		return maximumNb

	def analyseColumn(self, column, count, mean, std, minimum, quad25, quad50, quad75, maximum):
		self.column = column
		self.count = self.dataAnalysisCount()
		self.mean = self.dataAnalysisMean()
		self.std = self.dataAnalysisStd()
		self.min = self.dataAnalysisMin()
		self.quad25 = self.dataAnalysisQuant(1/4)
		self.quad50 = self.dataAnalysisQuant(1/2)
		self.quad75 = self.dataAnalysisQuant(3/4)
		self.max = self.dataAnalysisMax()
		count.append(str(self.count))
		mean.append(str(self.mean))
		std.append(str(self.std))
		minimum.append(str(self.min))
		quad25.append(str(self.quad25))
		quad50.append(str(self.quad50))
		quad75.append(str(self.quad75))
		maximum.append(str(self.max))

	def printRowsHeader(self, table):
		for row in table:
			for element in row:
				print(element[:6], end="\t")
			print("")

def isRealNumb(elem):
    if (bool(re.match('^[0123456789]+$', elem)) or ((elem[0] == '-' or elem[0] == '+') and bool(re.match('^[.0123456789]+$', elem[1:])))):
        return True
    return False

def displayFeatures(filename):
	file = open(filename)
	df = pd.read_csv(file)
	columns = list(df)
	da = dataAnalysis()
	labelRow = [""]
	countRow = ["Count"]
	meanRow = ["Mean"]
	stdRow = ["std"]
	minRow = ["min"]
	quad25Row = ["25%"]
	quad50Row = ["50%"]
	quad75Row = ["75%"]
	maxRow = ["max"]
	for c in columns:
		tmpCol = df[c]
		if tmpCol.dtypes == float or tmpCol.dtypes == int:
			tmpCol = tmpCol.dropna()
			tmpCol = tmpCol.sort_values(ascending=[True])
			labelRow.append(tmpCol.name)
			da.analyseColumn(tmpCol, countRow, meanRow, stdRow, minRow, quad25Row, quad50Row, quad75Row, maxRow)
	daTable = []
	daTable.extend((labelRow, countRow, meanRow, stdRow, minRow, quad25Row, quad50Row, quad75Row, maxRow))
	da.printRowsHeader(daTable)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("\x1b[91musage: python3 histogram.py <file.csv>\x1b[0m")
		exit(1)
	df = pd.read_csv(sys.argv[1])
	print(df.describe())
	displayFeatures(sys.argv[1])
