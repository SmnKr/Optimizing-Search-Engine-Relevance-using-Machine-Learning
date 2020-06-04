import os
import sys
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


############################## Preprocessings Function ##############################
def preprocessing(inputFileDir,FeatureNum):

	dataframe = pd.read_csv(inputFileDir, sep=",")
	array = dataframe.values
	X = array[:,1:FeatureNum]#Assumption1: columns [1:lastFeatureIndex] are the features in the input data
	Y = array[:,0]#Assumption2: The first column is annotation column in the input data

	#splitting into train and test data
	X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
	trainData = np.concatenate((y_train[:, None],X_train[:,0:12]), axis=1)	
	testData = np.concatenate((y_test[:, None],X_test[:,0:12]), axis=1)

	with open("Train.csv","w") as my_csv:
		csvWriter = csv.writer(my_csv,delimiter=',')
		csvWriter.writerows(trainData)
	with open("Test.csv","w") as my_csv:
		csvWriter = csv.writer(my_csv,delimiter=',')
		csvWriter.writerows(testData)


	for fileName in ["Train","Test"]:
		outputFileDir = "SVMRANK_"+fileName+".dat"
		InFile = open(fileName+".csv",'r')
		OutFile = open(outputFileDir,'w')
		InFile.readline()#headers
		for line in InFile:
			line = line.strip()
			st = line.split(',')
			newLine = st[0]+" qid:"+st[1]
			for i in range(2,FeatureNum):
				newLine = newLine + " "+ str(i-1)+":"+st[i]
			OutFile.write(newLine+"\n")


if __name__ == "__main__":
	preprocessing("input_Dataset.csv",FeatureNum)
	print("Test and Train Files are created according to the SVMRank tool input format.")