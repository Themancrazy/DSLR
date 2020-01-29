import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as figure

def plotScatter(c1, c2):
    plt.scatter(c1, c2, label="Scores proportion", color='r')
    plt.xlabel('Astronomy')
    plt.ylabel('Defense Against the Dark Arts')
    plt.legend()
    plt.show()

def makeScattering():
    file = open('resources/dataset_train.csv')
    df = pd.read_csv(file)
    # course1 = df.loc[df['Hogwarts House'] == "Slytherin", 'Astronomy']
    # course2 = df.loc[df['Hogwarts House'] == "Slytherin", 'Defense Against the Dark Arts']
    course1 = df['Astronomy']
    course2 = df['Defense Against the Dark Arts']
    plotScatter(course1, course2)

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("scatter doesn't take arguments.")
        exit(1)
    makeScattering()    