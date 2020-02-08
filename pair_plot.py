import sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def makeScattering():
	file = open('resources/dataset_train.csv')
	df = pd.read_csv(file)
	df = df.loc[:, ("Hogwarts House", "Charms", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes")]
	df = df.dropna(how='any', axis=0)
	sns.pairplot(df, hue="Hogwarts House")
	plt.show()

if __name__ == "__main__":
	if len(sys.argv) != 1:
		print("scatter doesn't take arguments.")
		exit(1)
	makeScattering()   
