import pandas as pd 
import os
import sys
import getopt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import numpy as np
import scipy
import csv
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

############################## Preprocessings Function ##############################
def preprocessing(inputFileDir):

	################## Duplicate Removal ################
	print("start preprocessing!")
	print("inputFileDir: "+str(inputFileDir))
	dataframe = pd.read_csv(inputFileDir, sep=",")
	dataframe.drop_duplicates(subset =['query_id','url_id'], keep = 'first', inplace = True)

	################## Feature Scaling #################
	array = dataframe.values
	# separate array into input and output components
	X = array[:,[2,4,5,6,7,8,9,10,11]]
	Y = array[:,-1]
	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaledX = scaler.fit_transform(X)

	################## Standardization ################
	# Centering the feature columns at mean 0 with standard deviation 
	scaler = StandardScaler().fit(rescaledX)
	StandardizedX = scaler.transform(rescaledX)

	preprocX = np.concatenate((array[:,0:2],StandardizedX[:,[0]],array[:,[3]],StandardizedX[:,1:9],array[:,[12]]), axis=1)	
	preprocessedFileDir = ''.join("Preprocessed_"+inputFileDir)
	with open(preprocessedFileDir,"w") as my_csv:
		csvWriter = csv.writer(my_csv,delimiter=',')
		csvWriter.writerow(['query_id','url_id','query_length','is_homepage','sig1','sig2','sig3','sig4','sig5','sig6','sig7','sig8','relevance'])
		csvWriter.writerows(preprocX)
	return preprocessedFileDir



########################### SVM Classification #############################
def SVMClassification(preprocessedInputDir,outputFolderName,DimRedMethod):

	dataframe = pd.read_csv(preprocessedInputDir, sep=",")
	array = dataframe.values
	X = array[:,0:12]
	Y = array[:,-1]

	#writing to outPutFile
	outFile = open(outputFolderName+"/SVMResults.txt","w")

	#splitting into train and test data
	X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
	
	#### Dimension Reduction using PCA ####
	if(DimRedMethod == "PCA"):
		pca = PCA(.95)
		pca.fit(X_train)
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)

	#### Dimension Reduction using Anova test ####
	if (DimRedMethod == "AnovaTest"):
		test = SelectKBest(score_func=f_classif, k=5)
		fit = test.fit(X_train, y_train)
		X_train = fit.transform(X_train)
		X_test = fit.transform(X_test)

	##### SVM Parameter Tuning #####					
	# Set the parameters by cross-validation
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	scores = ['precision', 'recall']
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		#clf = GridSearchCV(svm.NuSVC(), tuned_parameters, scoring='%s_macro' % score)
		#clf = GridSearchCV(svm.LinearSVC(), tuned_parameters, scoring='%s_macro' % score)
		clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s_macro' % score,n_jobs=3)
		clf.fit(X_train, y_train)
		print()
		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean, std * 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		print()
		outFile.write("the model using optimal parameter value on the test set"+"\n")
		outFile.write(classification_report(y_true, y_pred))
	outFile.close()

##################### Random Forest Classification #########################
def RandomForestClassification(preprocessedInputDir,outputFolderName,DimRedMethod):
	dataframe = pd.read_csv(preprocessedInputDir, sep=",")
	array = dataframe.values
	X = array[:,0:12]
	Y = array[:,-1]
	
	#writing to outPutFile
	outFile = open(outputFolderName+"/RandomForestResults.txt","w")

	#splitting into train and test data
	X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
	
	#### Dimension Reduction using PCA ####
	if(DimRedMethod == "PCA"):
		pca = PCA(.95)
		pca.fit(X_train)
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)

	#### Dimension Reduction using Anova test ####
	if (DimRedMethod == "AnovaTest"):
		test = SelectKBest(score_func=f_classif, k=5)
		fit = test.fit(X_train, y_train)
		X_train = fit.transform(X_train)
		X_test = fit.transform(X_test)

	##### RF Parameter Tuning #####
	#hyper parameter tuning. Selecting best n_estimators
	tuned_parameters = {'bootstrap': [True, False],
 	'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 	'max_features': ['auto', 'sqrt'],
 	'min_samples_leaf': [1, 2, 4],
 	'min_samples_split': [2, 5, 10],
 	'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
	scores = ['precision', 'recall']
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring='%s_macro' % score, n_jobs=3)
		clf.fit(X_train, y_train)
		print()
		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean, std * 2, params))
		print()
		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		#Run the model using optimal parameter value on the test set
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		print()
		outFile.write("the model using optimal parameter value on the test set"+"\n")
		outFile.write(classification_report(y_true, y_pred))
	outFile.close()


##################### Fully Connected MLP Classification #########################
def MLPClassification(preprocessedInputDir,outputFolderName,DimRedMethod):
	dataframe = pd.read_csv(preprocessedInputDir, sep=",")
	array = dataframe.values
	X = array[:,0:12]
	Y = array[:,-1]
	X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
	#writing to outPutFile
	outFile = open(outputFolderName+"/MLPResults.txt","w")

	#### Dimension Reduction using PCA ####
	if(DimRedMethod == "PCA"):
		pca = PCA(.95)
		pca.fit(X_train)
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)
		print("After PCA")
	#### Dimension Reduction using Anova test ####
	elif (DimRedMethod == "AnovaTest"):
		test = SelectKBest(score_func=f_classif, k=5)
		fit = test.fit(X_train, y_train)
		X_train = fit.transform(X_train)
		X_test = fit.transform(X_test)
	
	##### MLP Parameter Tuning #####
	tuned_parameters = {
		'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
		'activation': ['tanh', 'relu'],
		'solver': ['sgd', 'adam'],
		'alpha': [0.0001, 0.05],
		'learning_rate': ['constant','adaptive'],
	}
	scores = ['precision', 'recall']
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		clf = GridSearchCV(MLPClassifier(), tuned_parameters, scoring='%s_macro' % score,n_jobs=3)
		clf.fit(X_train, y_train)
		print()
		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean, std * 2, params))
		print()
		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		#Run the model using optimal parameter value on the test set
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		print()
		outFile.write("the model using optimal parameter value on the test set"+"\n")
		outFile.write(classification_report(y_true, y_pred))
	outFile.close()

############################## Setup Function ##############################
def setups(argv):

	if len(argv) == 0:
		print('You must pass some parameters. Use \"-h\" to help.')
		return
	if len(argv) == 1 and argv[0] == '-h':
		f = open('ReadMe.txt', 'r')
		print(f.read())
		f.close()
		return
	inputFileName = "Dataset.csv"#default value
	outputFolderName = "Results"#default value
	DimRedMethod = "None"#default value
	ClassificationMethod = "RandomForest"#default value
	if not os.path.exists(outputFolderName):
		os.makedirs(outputFolderName)
	opts, args = getopt.getopt(sys.argv[1:],"d:m:")
	for opt,arg in opts:
		if opt == '-d':
			DimRedMethod = arg
		elif opt == '-m':
			ClassificationMethod = arg
		else:
			print("Usage: %s -d DimRedMethod -m ClassificationMethod" % sys.argv[0])

	########### Step One: Preprocessing ############
	print("...............Preprocessings................")
	preprocessedInputDir = preprocessing(inputFileName)
	print("Done!")

	###########  SVM Classification ############
	if "SVM" in ClassificationMethod:
		print("..............SVM Classification...............")
		SVMClassification(preprocessedInputDir,outputFolderName,DimRedMethod)
		print("Done!")


	###########  RandomForest Classification ############
	elif "RandomForest" in ClassificationMethod:
		print("..........Random Forest Classification...........")
		RandomForestClassification(preprocessedInputDir,outputFolderName,DimRedMethod)
		print("Done!")


	###########  MLP Classification ############
	elif "MLP" in ClassificationMethod:
		print("..........MLP Classification...........")
		MLPClassification(preprocessedInputDir,outputFolderName,DimRedMethod)
		print("Done!")


if __name__ == "__main__":
	setups(sys.argv[1:])