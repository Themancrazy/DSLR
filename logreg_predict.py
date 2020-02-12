import sys
import json
import csv
import math
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def standardize_data(mean, std, data):
	return (data - mean) / std

def write_result_in_file(result):
	file = open("houses.csv", 'w+')
	wr = csv.writer(file, lineterminator='\n')
	wr.writerow(["Index", "Hogwarts House"])
	for i, house in zip(range(len(result)), result):
		wr.writerow([i, house])

def replace_nan(column):
	mean = (column.dropna()).mean()
	column = column.fillna(mean)
	return column

def predict_houses(df, infos):
	mean = infos["means"]
	std = infos["std"]
	houses = ["Hufflepuff", "Ravenclaw", "Slytherin", "Gryffindor"]
	courses = ["theta_0", "Charms", "Herbology", "Ancient Runes", "Astronomy", "Defense Against the Dark Arts"]
	x = standardize_data(mean, std, replace_nan(df.loc[:, courses[1:]]))
	x.insert(0, "theta_0", np.ones(df.shape[0]))
	result = np.zeros(x.shape[0])
	for house in houses:
		y = 1 / (1 + np.exp(-1 * np.dot(x, infos[house])))
		result = (np.where(y > 0.5, house, result))
	write_result_in_file(result)

if __name__ == "__main__":
	file = open("resources/dataset_test.csv")
	weights_file = open("weights.json")
	df = pd.read_csv(file)
	df.insert(0, "theta_0", np.ones(df.shape[0]))
	thetas = json.load(weights_file)
	predict_houses(df, thetas)
