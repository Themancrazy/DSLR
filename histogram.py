import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as figure

def normalize_column(c):
	return ((c - c.min()) / (c.max() - c.min()))

def create_graph(df, course, houses, axs, row, col):
	colors = ["red", "blue", "green", "deeppink"]
	column = normalize_column(df[course])
	for house, color in zip(houses, colors):
		axs[row, col].hist(column.loc[df["Hogwarts House"] == house], 15, facecolor=color, label=house)
	axs[row, col].set_title(course + " Homogeneity")
	axs[row, col].legend()
	for ax in axs.flat:
 	   ax.set(xlabel='Score', ylabel='Amount of students')

def make_histogram(df):
	col = 0
	row = 0
	houses = ["Hufflepuff", "Ravenclaw", "Slytherin", "Gryffindor"]
	courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
	fig, axs = plt.subplots(4, 4)
	for course in courses:
		create_graph(df, course, houses, axs, row, col)
		if col == 3:
			col = 0
			row += 1
		else:
			col += 1
	plt.show()


if __name__ == "__main__":
	if len(sys.argv) != 1:
		print("histogram doesn't take arguments.")
		exit(1)
	file = open("resources/dataset_train.csv")
	df = pd.read_csv(file)
	make_histogram(df)