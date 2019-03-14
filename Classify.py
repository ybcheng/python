import os
import pandas as pd
import numpy as np
import laspy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def OpenLasFileWithReturn(Filename):
    InputFile = laspy.file.File(Filename, mode='r')
    Data = np.vstack([InputFile.z, InputFile.intensity.astype(float), InputFile.red.astype(float), InputFile.blue.astype(float), InputFile.green.astype(float), InputFile.num_returns.astype(float), InputFile.return_num.astype(float)]).transpose()
    DF = pd.DataFrame(Data, columns = ['Z', 'Intensity', 'Red', 'Blue', 'Green', 'NumberOfReturns', 'ReturnNumber'])
    Header = InputFile.header
    InputFile.close()
    return DF, Header

def OpenLasFile(Filename):
    InputFile = laspy.file.File(Filename, mode='r')
    Data = np.vstack([InputFile.z, InputFile.intensity.astype(float), InputFile.red.astype(float), InputFile.blue.astype(float), InputFile.green.astype(float)]).transpose()
    DF = pd.DataFrame(Data, columns = ['Z', 'Intensity', 'Red', 'Blue', 'Green'])
    Header = InputFile.header
    InputFile.close()
    return DF, Header

def GetDataFrames(Filepath):
    Count = 0
    for Filename in os.listdir(Filepath):
        if Count == 0:
            DF, Header = OpenLasFile(Filepath+"\\"+Filename)
            Count += 1
        else:
            DF.append(OpenLasFile(Filepath+"\\"+Filename)[0])
    return DF

def GetSingleDataFrame(Path):
    DF, Header = OpenLasFileWithReturn(Path)
    return DF

def GetNDSI(DF):
    DF['NDSI'] = (DF['Red'] - DF['Green'])/(DF['Red'] + DF['Green'])
    DF = DF.fillna(0.)
    return DF

def GetMaxMin(Feature, A):
    A = A[Feature].values.flatten()
    return np.max(A), np.min(A)

def GetMinimumsAndMaximums(Filepath):
    Count = 0
    for Filename in os.listdir(Filepath):
        DF, Header = OpenLasFile(Filepath + "\\" + Filename)
        if Count == 0:
            FeatureList = list(DF)
            Minimum = 1.e6*np.ones([len(FeatureList)])
            Maximum = -1.e6*np.ones([len(FeatureList)])

            for i in range(len(FeatureList)):
                Feature = FeatureList[i]
                Max, Min = GetMaxMin(Feature, DF)
                if Max > Maximum[i]:
                    Maximum[i] = Max
                if Min < Minimum[i]:
                    Minimum[i] = Min
            Count += 1
        else:
            for i in range(len(FeatureList)):
                Feature = FeatureList[i]
                Max, Min = GetMaxMin(Feature, DF)
                if Max > Maximum[i]:
                    Maximum[i] = Max
                if Min < Minimum[i]:
                    Minimum[i] = Min
            Count+= 1
        print (Count)
    return Minimum, Maximum

def ScaleData(DF, Minimums, Maximums):
    Features = list(DF)
    for i in range(len(list(DF))):
        Feature = Features[i]
        DF[Feature] = (DF[Feature] - Minimums[i])/(Maximums[i] - Minimums[i])
    return DF


    
def main():
    BareEarth = GetDataFrames("E:\\NeuralNetworkClassifier\\DATA\\TRAINING\\Bare")
    Vegetation = GetDataFrames("E:\\NeuralNetworkClassifier\\DATA\\TRAINING\\Vegetation")
    print ("Getting minimums and maximums")

    #Elevation data is the point cloud with heights measured relative to ground. "INPUT" is the original point cloud. Both are generated with the same tiling scheme so direct application of classification
    #can occur after it works on the Delta_Z data set. We have to scale the data such that one feature does not overtake the others. To do this we find the minimum and maximum for each feature.
    Minimums, Maximums = GetMinimumsAndMaximums("E:\\NeuralNetworkClassifier\\DATA\\ELEVATIONDATA")
    print ("Minimums: ", Minimums)
    print ("Maximums: ", Maximums)

    Vegetation['IsItVeg'] = 1
    BareEarth['IsItVeg'] = 0

    #Finding which training class has the minimum number of points
    NumberOfTrainingPoints = np.min(np.array([Vegetation.shape[0], BareEarth.shape[0]]))
    #NumberOfTrainingPoints = 1000000

    #The fraction of random points to take out of the sample so both are represented equally
    VegFrac = NumberOfTrainingPoints/Vegetation.shape[0]
    GroundFrac = NumberOfTrainingPoints/BareEarth.shape[0]

    print (VegFrac, GroundFrac, NumberOfTrainingPoints)

    Vegetation = Vegetation.sample(frac=VegFrac)
    BareEarth = BareEarth.sample(frac=GroundFrac)
    TotalTrainingSet = Vegetation.append(BareEarth)

    #In case you want less than the total number of possible training data 
    NumberOfTraining = 2000000
    TotalTrainingSet = TotalTrainingSet.sample(frac=NumberOfTraining/TotalTrainingSet.shape[0])
    y = TotalTrainingSet['IsItVeg'].values
    TotalTrainingSet =TotalTrainingSet.drop(['IsItVeg'], axis = 1)

    #Scaling the data
    TotalTrainingSet = ScaleData(TotalTrainingSet, Minimums, Maximums)

    #Include NDSI if one so chooses
    #TotalTrainingSet = GetNDSI(TotalTrainingSet)


    X = TotalTrainingSet.values
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)

    print ("Training the neural network")
    print (y_train.size, " training data points")
    Classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(3,3), random_state=42)
    Classifier.fit(X_train, y_train)
    Prediction = Classifier.predict(X_test)
    print ("Number of test: ", y_test.size)
    print ("Confusion Matrix: ")
    print (confusion_matrix(y_test, Prediction))
    print ("Classification Report: ")
    print (classification_report(y_test, Prediction))

    for Filename in os.listdir("E:\\NeuralNetworkClassifier\\DATA\\ELEVATIONDATA"):
        Data = GetSingleDataFrame("E:\\NeuralNetworkClassifier\\DATA\\ELEVATIONDATA\\" + Filename)
        #Set initial classification of all data to 1
        Data['Classification'] = 1
        #If the return number does not equal the number of returns, that must mean it is vegetation
        Data[Data['ReturnNumber'] != Data['NumberOfReturns']].loc[:, 'Classification'] = 4
        InitialClassification = Data['Classification'].values
        Data = Data.drop(['Classification', 'NumberOfReturns', 'ReturnNumber'], axis = 1)
        Data = ScaleData(Data, Minimums, Maximums)
        #Data = GetNDSI(Data)


        Classification = []
        X = Data.values
        
        #Predict probabilities for each class 
        Prediction = Classifier.predict_proba(X)
 
        for i in range(Prediction.shape[0]):
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
                #If the probability that it is ground, is greater than 1sigma, classify it as ground
                if Prediction[i][0] > .6827:
                    Classification.append(2)
                #If the probability that it is vegetation is greater than 1 sigma, classify it as vegetation
                elif Prediction[i][1] > .6827:
                    Classification.append(4)
                else:
                #If neither of the above are true, keep it as unclassified for human eyes
                    Classification.append(1)
        Classification = np.asarray(Classification)
        unique, counts = np.unique(Classification, return_counts=True)
        print (dict(zip(unique, counts)))
        #We want to copy all features from the non-delta z point cloud and only replace the classification
        InputFile = laspy.file.File("E:\\NeuralNetworkClassifier\\DATA\\INPUT\\" + Filename, mode = 'r')
        Header = InputFile.header
        #Location of classified las files
        OutputFile = laspy.file.File("E:\\NeuralNetworkClassifier\\DATA\\OUTPUT\\" + Filename, mode = 'w', header = Header)
        #setting output points to the non-deltaz point cloud
        OutputFile.points = InputFile.points
        #Overwriting the classification
        OutputFile.raw_classification = Classification
        InputFile.close()
        OutputFile.close()



        


    

main()
