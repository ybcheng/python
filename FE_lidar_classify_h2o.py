# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 15:16:12 2018

@author: ybcheng
"""


import sys
import os
import glob

#import exiftool
import pandas as pd
import numpy as np

import base64
import struct
import shutil
import copy
import math
import time

#import gdal
import laspy
import colorsys

#import improc
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator




def OpenLasFile(Filename):
    InputFile = laspy.file.File(Filename, mode='r')
    Data = np.vstack([InputFile.x, InputFile.y, InputFile.z, InputFile.intensity.astype(float),
                      InputFile.red.astype(float), InputFile.green.astype(float), InputFile.blue.astype(float),  
                      InputFile.num_returns.astype(float), InputFile.return_num.astype(float),
                      InputFile.Raw_Classification.astype(int)]).transpose()
    DF = pd.DataFrame(Data, columns = ['X', 'Y', 'Z', 'Intensity', 
                                       'Red', 'Green', 'Blue',
                                       'NumberOfReturns', 'ReturnNumber', 'Classification'])
    Header = InputFile.header
    InputFile.close()
    return DF, Header
    
    
def GetDataFrames(Filenames):
    Count = 0
    for Filename in Filenames:
        if Count == 0:
            DF, Header = OpenLasFile(Filename)
            Count += 1
        else:
            DF.append(OpenLasFile(Filename)[0])
    return DF


#def GetDataFrames(Filepath):
#    Count = 0
#    for Filename in os.listdir(Filepath):
#        if Count == 0:
#            DF, Header = OpenLasFile(Filepath+"\\"+Filename)
#            Count += 1
#        else:
#            DF.append(OpenLasFile(Filepath+"\\"+Filename)[0])
#    return DF
    

#def ConvertToHSV(DF):
#    Values = DF[['Red', 'Green', 'Blue']].values
#    SumOfValues = np.sum(Values, axis = 1)
#    SumOfValues = SumOfValues[:, None]
#    Values = np.divide(Values, SumOfValues, out = np.zeros_like(Values), where = SumOfValues != 0.)
#    HSVValues = [colorsys.rgb_to_hsv(V[0], V[1], V[2]) for V in Values]
#    HSVValues = np.asarray(HSVValues)
#    HSVValues = np.nan_to_num(HSVValues)
#    DF['H'] = HSVValues[:, 0]
#    DF['S'] = HSVValues[:, 1]
#    DF['V'] = HSVValues[:, 2]
#    DF = DF.drop(['Red', 'Green', 'Blue'], axis = 1)
#    return DF
    
    
def classify(in_dir, grnd_fn='grnd', veg_fn='veg', terra=True):
    """
    """


    StartTime = time.time()
    
    #in_dir = in_dir.replace('/','\\')
    # setting up filenames for input, dz, and training
    in_fn = glob.glob(in_dir + '/' + '*las')
    
    dz_dir = in_dir.replace('/INPUT', '/ELEVATIONDATA')
    dz_fn = glob.glob(dz_dir + '/' + '*las')
    
    trn_dir = in_dir.replace('/INPUT', '/TRAINING')
    grnd_fn = glob.glob(trn_dir + '/' + grnd_fn + '*las')
    veg_fn = glob.glob(trn_dir + '/' + veg_fn + '*las')
    
    grnd_df = GetDataFrames(grnd_fn)
    veg_df = GetDataFrames(veg_fn)

    grnd_df['Classification'] = 2
    veg_df['Classification']  = 4

    trn_df = grnd_df.append(veg_df)
    
    trn_df['Classification'] = trn_df['Classification'].astype(int)
    num_cols = np.shape(trn_df)[1]
    col_dtypes = {'X': "float",
                  'Y': "float",
                  'Z': "float",
                  'Intensity': "float",
                  'Red': "float",
                  'Green': "float",
                  'Blue': "float",
                  'NumberOfReturns': "float",
                  'ReturnNumber': "float",
                  'Classification': "categorical"}

    #Elevation data is the point cloud with heights measured relative to ground. "INPUT" is the original point cloud. Both are generated with the same tiling scheme so direct application of classification
    #can occur after it works on the Delta_Z data set. We have to scale the data such that one feature does not overtake the others. To do this we find the minimum and maximum for each feature.
    #Minimums, Maximums = GetMinimumsAndMaximums(in_dir+"ELEVATIONDATA")
    #print ("Minimums: ", Minimums)
    #print ("Maximums: ", Maximums)

    #Vegetation['IsItVeg'] = 4
    #BareEarth['IsItVeg'] = 2

    #Finding which training class has the minimum number of points
    #NumberOfTrainingPoints = np.min(np.array([Vegetation.shape[0], BareEarth.shape[0]]))
    #NumberOfTrainingPoints = 1000000

    #The fraction of random points to take out of the sample so both are represented equally
    #VegFrac = NumberOfTrainingPoints/Vegetation.shape[0]
    #GroundFrac = NumberOfTrainingPoints/BareEarth.shape[0]

    #print ("Fraction of possible veg, fraction of possible ground, min number of training points for each class", VegFrac, GroundFrac, NumberOfTrainingPoints)

    #Vegetation = Vegetation.sample(frac=VegFrac)
    #BareEarth = BareEarth.sample(frac=GroundFrac)
    #TotalTrainingSet = Vegetation.append(BareEarth)

    #In case you want less than the total number of possible training data 
    #NumberOfTraining = 2000000
    #TotalTrainingSet = TotalTrainingSet.sample(frac=NumberOfTraining/TotalTrainingSet.shape[0])
    #y = TotalTrainingSet['IsItVeg'].values
    #TotalTrainingSet = TotalTrainingSet.drop(['IsItVeg'], axis = 1)

    #Scaling the data
    #TotalTrainingSet = ScaleData(TotalTrainingSet, Minimums, Maximums)

    #if UseHeightPercentiles == True:
    #    print ("Getting Percentiles for training")
    #    TrainingScores, TrainingZs = CalculatePercentile(TotalTrainingSet)
    #    TotalTrainingSet['Percentile'] = TrainingScores

    #if UseHSV == True:
    #    TotalTrainingSet = ConvertToHSV(TotalTrainingSet)                 


    #Include NDSI if one so chooses
    #TotalTrainingSet = GetNDSI(TotalTrainingSet)


    #X = TotalTrainingSet.values
    

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

    #print ("Training the neural network")
    #print (y_train.size, " total training data points")
    #Classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(4,), random_state=42, batch_size = 1000)
    #Classifier.fit(X_train, y_train)
    #VegPrediction = Classifier.predict(X_test)
    #print ("Number of test: ", y_test.size)
    #print ("Confusion Matrix for veg/ground: ")
    #print (confusion_matrix(y_test, VegPrediction))
    #print ("Classification Report for veg/ground: ")
    #print (classification_report(y_test, VegPrediction))


    h2o.init()
    h2o.remove_all()

    trn_hf = h2o.H2OFrame(trn_df, column_types=col_dtypes)
    
    x_hf = trn_hf.col_names[2: (num_cols-1)]
    y_hf = trn_hf.col_names[(num_cols-1)]
    
    modelH2O = H2ODeepLearningEstimator(epochs=123456)
    modelH2O.train(x_hf, y_hf, trn_hf)
    
    #modelH2O
    
    
    """
###Drains

    if UseDrain == True:
        Vegetation_ = GetDataFrames(in_dir+"TRAINING\\Vegetation")
        Vegetation_['IsItDrain'] = 0
        Drains['IsItDrain'] = 1

        NumberOfTrainingPoints = np.min(np.array([Vegetation_.shape[0], Drains.shape[0]]))
        VegFrac = NumberOfTrainingPoints/Vegetation_.shape[0]
        DrainsFrac = NumberOfTrainingPoints/Drains.shape[0]

        Vegetation_ = Vegetation_.sample(frac=VegFrac)
        Drains = Drains.sample(frac=DrainsFrac)
        TotalTrainingSet = Vegetation_.append(Drains)



        y = TotalTrainingSet['IsItDrain'].values
        TotalTrainingSet = TotalTrainingSet.drop(['IsItDrain'], axis = 1)
        TotalTrainingSet = ScaleData(TotalTrainingSet, Minimums, Maximums)
    

        if UseHeightPercentiles == True:
            print ("Getting Percentiles for drains")
            DrainPercentiles = CalculatePredictionPercentile(TrainingZs, TotalTrainingSet['Z'].values)
            TotalTrainingSet['Percentile'] = DrainPercentiles


        #TotalTrainingSet = TotalTrainingSet.drop(['Z'], axis = 1)

        if UseHSV == True:
            TotalTrainingSet = ConvertToHSV(TotalTrainingSet)                 

        X = TotalTrainingSet.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

        print ("Training the neural network")
        print (y_train.size, " total training data points")
        Classifier1 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(4,), random_state=42, batch_size = 1000)
        Classifier1.fit(X_train, y_train)
        DrainPrediction = Classifier1.predict(X_test)
        print ("Number of drain test points: ", y_test.size)
        print ("Drain Confusion Matrix: ")
        print (confusion_matrix(y_test, DrainPrediction))
        print ("Drain Classification Report: ")
        print (classification_report(y_test, DrainPrediction))

###########################
    """
 
    for dz in dz_fn:
        print ("Classifying: ", os.path.basename(dz))
        
        #dz_df  = GetDataFrames(in_dir+"\\ELEVATIONDATA\\" + Filename)
        dz_df, header = OpenLasFile(dz)
        dz_df['Classification'] = dz_df['Classification'].astype(int)
        dz_hf  = h2o.H2OFrame(dz_df, column_types=col_dtypes)
        #dz_hf1 = dz_hf.col_names[2:9]
        
        pred = modelH2O.predict(dz_hf[:,2:(num_cols-1)]).as_data_frame(True)
        
        print(pred)
        #Set initial classification of all data to 1
        #Data['Classification'] = 1
        
        #If the return number does not equal the number of returns, that must mean it is vegetation
        #Data[Data['ReturnNumber'] != Data['NumberOfReturns']].loc[:, 'Classification'] = 4
        #InitialClassification = Data['Classification'].values
        #Data = Data.drop(['Classification', 'NumberOfReturns', 'ReturnNumber'], axis = 1)
        #Data = ScaleData(Data, Minimums, Maximums)
        #Data = GetNDSI(Data)

        #if UseHSV == True:
        #    Data = ConvertToHSV(Data)     

        #if UseHeightPercentiles == True:
        #    print ("Getting Percentiles for prediction")
        #    Percentiles = CalculatePredictionPercentile(TrainingZs, Data['Z'].values)
        #    Data['Percentile'] = Percentiles            


        #Classification = []
        #X = Data.values
        
        #Predict probabilities for each class 
        #VegPrediction = Classifier.predict_proba(X)
        
"""     print (InitialClassification.size, VegPrediction.shape)

        #X = Data.drop(['Z'], axis = 1).values
        if UseDrain == True:
            DrainPrediction = Classifier1.predict_proba(X)

        VegetationPrediction = VegPrediction[:, 1]
 
        for i in range(InitialClassification.shape[0]):
            #If it was originally marked as ground in TerraSolid/Global mapper, keep the classification
            if InitialClassification[i] == 2:
                Classification.append(2)
            #If it was originally marked as man-made, keep the classification
            elif (InitialClassification[i] == 6) or (InitialClassification[i] == 10) or (InitialClassification[i] >=13):
                Classification.append(6)
            #If it found that the return number did not equal the number of returns previously, it is marked as vegatation
            elif InitialClassification[i] == 4:
                Classification.append(4)
            else:
                #If the probability that it is vegetation, is less than 1sigma, classify it as ground
                if VegetationPrediction[i] < 0.37:
                    Classification.append(2)
                #else, 
                else:
                    if UseDrain == True:
                        if DrainPrediction[i][1] > 0.67:
                            Classification.append(2)
                        else:
                            Classification.append(4)
                    else:
                        Classification.append(4)

                 
        Classification = np.asarray(Classification)
        unique, counts = np.unique(Classification, return_counts=True)
        print (dict(zip(unique, counts)))
        #We want to copy all features from the non-delta z point cloud and only replace the classification
        InputFile = laspy.file.File(in_dir+"INPUT\\" + Filename, mode = 'r')
        Header = InputFile.header
        #Location of classified las files
        OutputFile = laspy.file.File(in_dir+"OUTPUT\\" + Filename, mode = 'w', header = Header)
        #setting output points to the non-deltaz point cloud
        OutputFile.points = InputFile.points

        print (InputFile.points.shape, InitialClassification.shape, Classification.shape)
        #Overwriting the classification
        OutputFile.raw_classification = Classification
        InputFile.close()
        OutputFile.close()

    EndTime = time.time()
    print ("Execution took: ", (EndTime-StartTime)/60., " minutes")
        
"""