# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:10:16 2019

@author: Sadman Sakib
"""
import globalVars_q2 as gv 
import pandas as pd
#Split dataframe into positive and negative
dfs_SplitByClasses=[]
for region, df_region in gv.dfTrain.groupby('BUY'):
    dfs_SplitByClasses.append(df_region)
X_Train_Negative=dfs_SplitByClasses[0]
X_Train_Positive=dfs_SplitByClasses[1]
X_Train_Negative=X_Train_Negative.iloc[:, 0:len(gv.dfTrain.columns)-1] 
X_Train_Positive=X_Train_Positive.iloc[:, 0:len(gv.dfTrain.columns)-1] 
trainLengthNegative=X_Train_Negative.shape[0]
trainLengthPositive=X_Train_Positive.shape[0]
#***Find unique values in each feature***
def findUniqueValueDict():
    for colIndex in range(len(gv.allColumns)):
        uniqueValFound_df_fullds=(gv.X[gv.allColumns[colIndex]].value_counts())     
        eachColAllPossValList=list()
        for items in uniqueValFound_df_fullds.iteritems(): 
            eachColAllPossValList.append(items[0])
        gv.valuesFullDS[gv.allColumns[colIndex]]=eachColAllPossValList   
def findExtra(value):
        m=len(value)
        p=1/m    
        return m, p
#***Peroformance Measure***
def performanceCalculation(y_actual, y_predicted):
    TP = FP = TN = FN =  0
    for i in range(len(y_actual)): 
        if y_actual[i]==y_predicted[i]==gv.labels[0]:
           TP += 1
        elif y_actual[i]==y_predicted[i]==gv.labels[1]:
           TN += 1
        elif y_predicted[i]==gv.labels[0] and y_actual[i]!=y_predicted[i]:
           FP += 1
        elif y_predicted[i]==gv.labels[1] and y_actual[i]!=y_predicted[i]:
           FN += 1
    T_N=TP + FP + TN + FN
    print("Accuracy =", (((TP+TN)/T_N)*100),"%")
    print("Sensitivity =", ((TP)/(TP + FN)))
    print("Specificity =", ((TN)/(TN + FP)))
#***PRIORS Calculation***
def priorCalculate():
    global prior_P, prior_N
    classValueCount=gv.dfTrain['BUY'].value_counts()
    prior_P=classValueCount[gv.labels[0]]/(classValueCount[gv.labels[0]]+classValueCount[gv.labels[1]])
    prior_N=classValueCount[gv.labels[1]]/(classValueCount[gv.labels[0]]+classValueCount[gv.labels[1]])

#***Likelyhood calculation***
def likelyhoodCalculate(typeVal):
    for key, value in gv.valuesFullDS.items():
        likelyhood=list()
        frequencyList=list()
        for eachValue in value:
            if(typeVal=='N'):
                count=(X_Train_Negative[key] == eachValue).sum()
                trainLen=trainLengthNegative
            else:
                count=(X_Train_Positive[key] == eachValue).sum()
                trainLen=trainLengthPositive
            frequencyList.append(count)
            m,p=findExtra(value)
            likelyhood.append((count+(m*p))/(trainLen+m))     
        data = {'Value':value, 'Frequency':frequencyList, 'Likelyhood':likelyhood}    
        df_Objects = pd.DataFrame(data) 
        if(typeVal=='N'):
            gv.likelyhood_N[key]=df_Objects 
        else:
            gv.likelyhood_P[key]=df_Objects   
#**** Find posterior ****            
def calculatePosterior(posterior, likelyhood, items):
        df_Found=likelyhood[items[0]]
        foundRow=df_Found.loc[df_Found['Value'] == items[1]]
        individualLikelyhood=foundRow['Likelyhood']
        individualLikelyhood=individualLikelyhood.values[0]
        posterior=posterior*individualLikelyhood 
        return posterior
#***    Find probability to belong in each classes***
def calculateTestProbability(testRow):
    global individualLikelyhood
    global posterior_N
    global posterior_P
    individualLikelyhood=0
    posterior_P=prior_P
    posterior_N=prior_N
    for items in testRow.iteritems():
        posterior_N=calculatePosterior(posterior_N, gv.likelyhood_N, items)
        posterior_P=calculatePosterior(posterior_P, gv.likelyhood_P, items)
#*** TESTING ***
predictions=list()
def testPrediction():
    for index, row in gv.X_test.iterrows():
        calculateTestProbability(row)
        posterior_N_normalize=(posterior_N)/(posterior_N+posterior_P)
        posterior_P_normalize=(posterior_P)/(posterior_N+posterior_P)
        if(posterior_N_normalize>posterior_P_normalize):
            predictions.append(gv.labels[1])
        else:
            predictions.append(gv.labels[0])                
findUniqueValueDict()     
priorCalculate()   
#*** NEGATIVE LIKELYHOOD CALCULATION ***
likelyhoodCalculate('N')
#*** POSITIVE LIKELYHOOD CALCULATION ***
likelyhoodCalculate('P')
testPrediction()
#*** PERFORMANCE Measure ***
performanceCalculation(list(gv.Y_test), predictions)
gv.dfTest['predictions']=predictions
#*** Save predictions file ***
gv.dfTest.to_csv("Predictions.csv", sep=',',index = None)