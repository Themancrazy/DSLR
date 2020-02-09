# Display mean of score of each house per course.

import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as figure

def plotHistogram(courses, houses, ravenMeans, slythMeans, gryffMeans, hufflMeans):
    df = pd.DataFrame({'Ravenclaw': ravenMeans, 'Slytherin': slythMeans, 'Gryffindor': gryffMeans, 'Hufflepuff': hufflMeans}, index=courses)
    ax = df.plot.bar(rot=0)
    plt.xlabel("Subjects")
    plt.ylabel("Scores")
    plt.title("Each class's average score per house")
    plt.show()

def normalizeValue(value, minVal, maxVal):
    return (value - minVal) / (maxVal - minVal)

def findMean(course, house, df):
    data = df.loc[df['Hogwarts House'] == house, course]
    return normalizeValue(data.mean(), data.min(), data.max())

def makeHistogram():
    file = open("resources/dataset_train.csv")
    df = pd.read_csv(file)
    courses = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]
    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    ravenMeans = []
    slythMeans = []
    gryffMeans = []
    hufflMeans = []
    for c in courses:
        meanForClass = []
        for h in houses:
            meanForClass.append(findMean(c, h, df))
        ravenMeans.append(meanForClass[0])
        slythMeans.append(meanForClass[1])
        gryffMeans.append(meanForClass[2])
        hufflMeans.append(meanForClass[3])
    plotHistogram(courses, houses, ravenMeans, slythMeans, gryffMeans, hufflMeans)
        

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("histogram doesn't take arguments.")
        exit(1)
    makeHistogram()