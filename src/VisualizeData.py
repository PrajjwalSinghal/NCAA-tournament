import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def main():
    VisualizePCA()
    VisualizeTrainingTime()
    VisualizeTestingTime()
    VisualizeTestingScore()
    Visualize2019PredictionScore()

def Visualize2019PredictionScore():
    TestSet = pd.read_csv('../Results/PredictionsResult.csv')
    TestSet = TestSet.drop(TestSet.columns[0], axis = 1)
    TestSet = TestSet.drop(TestSet.columns[0], axis = 1)
    TestSet = TestSet.drop('TimeTaken', axis = 1)
    MixedPredictionTime = TestSet[TestSet['TestData'] == 'Mixed 2019']
    X = list(MixedPredictionTime['ModelName'])
    Y = list(MixedPredictionTime['PredictionScore'])
    Y = [x * 100 for x in Y]
    fig = figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    for i in range(0, 7):
        string = str(Y[i])
        string = string[0:6]
        if(string == '100.0'):
            ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),color='red')
            continue
        if(X[i] == 'DecisionTrees'):
            ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),color='g')
            continue
        ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),)
    ax.set_ylim(40,120)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy in Percentage')
    ax.set_title('2019 Prediction Score Mixed Models')
    plt.savefig('../Plots/Mixed2019Predicitons.png')
    
    NCAAPredictionTime = TestSet[TestSet['TestData'] == 'NCAA 2019']
    X = list(NCAAPredictionTime['ModelName'])
    Y = list(NCAAPredictionTime['PredictionScore'])
    Y = [x * 100 for x in Y]
    fig = figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    for i in range(0, 7):
        string = str(Y[i])
        string = string[0:6]
        if(string == '100.0'):
            ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),color='red')
            continue
        if(X[i] == 'RandomForest'):
            ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),color='g')
            continue
        ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),)
    ax.set_ylim(40,120)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy in Percentage')
    ax.set_title('2019 Prediction Score NCAAa Models')
    plt.savefig('../Plots/NCAA2019Predicitons.png')
    
def VisualizeTestingScore():
    TestSet = pd.read_csv('../Results/PredictionsResult.csv')
    TestSet = TestSet.drop(TestSet.columns[0], axis = 1)
    TestSet = TestSet.drop(TestSet.columns[0], axis = 1)
    TestSet = TestSet.drop('TimeTaken', axis = 1)
    
    MixedPredictionTime = TestSet[TestSet['TestData'] == 'Mixed']
    X = list(MixedPredictionTime['ModelName'])
    Y = list(MixedPredictionTime['PredictionScore'])
    Y = [x * 100 for x in Y]
    fig = figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    for i in range(0, 7):
        string = str(Y[i])
        string = string[0:6]
        if((X[i] == 'SVM') or (X[i] == 'LogisticRegression')):
            ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),color='g')
            continue
        
        ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),)
        
    ax.set_ylim(40,100)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy in Percentage')
    ax.set_title('Testing Score for Mixed Testing Set')
    plt.savefig('../Plots/MixedTestingResult.png')
    
    NCAAPredictionTime = TestSet[TestSet['TestData'] == 'NCAA']
    X = list(NCAAPredictionTime['ModelName'])
    Y = list(NCAAPredictionTime['PredictionScore'])
    Y = [x * 100 for x in Y]
    fig = figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    for i in range(0, 7):
        string = str(Y[i])
        string = string[0:6]
        if((X[i] == 'DecisionTrees') or (X[i] == 'LogisticRegression') or (X[i] == 'SVM')):
            ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),color='g')
            continue
        ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], Y[i] + 10),
                    arrowprops=dict(facecolor='black', shrink=0.05),)
    ax.set_ylim(20,120)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy in Percentage')
    ax.set_title('Testing Score for NCAA Testing Set')
    plt.savefig('../Plots/NCAATestingResult.png')
    
def VisualizeTestingTime():
    # MixedSet = 83089 x 13
    # NCAASet = 1084 x 31
    TestSet = pd.read_csv('../Results/PredictionsResult.csv')
    TestSet = TestSet.drop(TestSet.columns[0], axis = 1)
    TestSet = TestSet.drop(TestSet.columns[0], axis = 1)
    TestSet = TestSet.drop('PredictionScore', axis = 1)
    MixedPredictionTime = TestSet[TestSet['TestData'] == 'Mixed']
    X = list(MixedPredictionTime['ModelName'])
    Y = list(MixedPredictionTime['TimeTaken'])
    
    fig = figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    for i in range(0, 7):
        string = str(Y[i])
        string = string[0:6]
        ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], float(Y[i]) + 3),
                    arrowprops=dict(facecolor='black', shrink=0.05),)
    ax.set_ylim(-0.5,32)
    ax.set_xlabel('Models')
    ax.set_ylabel('Time in Seconds')
    ax.set_title('Testing Times for Mixed Models')
    plt.savefig('../Plots/MixedTestingTime.png')
    
    NCAAPredictionTime = TestSet[TestSet['TestData'] == 'NCAA']
    X = list(NCAAPredictionTime['ModelName'])
    Y = list(NCAAPredictionTime['TimeTaken'])
    
    fig = figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    for i in range(0, 7):
        string = str(Y[i])
        string = string[0:6]
        ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], float(Y[i]) + 0.005),
                    arrowprops=dict(facecolor='black', shrink=0.05),)
    ax.set_ylim(-0.001,0.02)
    ax.set_xlabel('Models')
    ax.set_ylabel('Time in Seconds')
    ax.set_title('Testing Times for NCAA Models')
    plt.savefig('../Plots/NCAATestingTime.png')
def VisualizePCA():
    Mixed_Test = pd.read_csv('../PreProcessedData/Mixed_Test.csv')
    Mixed_Train = pd.read_csv('../PreProcessedData/Mixed_Train.csv')
    
    MixedDataset = Mixed_Train.append(Mixed_Test)
    MixedDataset = MixedDataset.drop(MixedDataset.columns[0], axis = 1)
    
    Mixed_X = MixedDataset.drop('DiffResult', axis = 1)
    
    pca = PCA()
    pca.fit(Mixed_X)
    pca.n_components_
    VarianceList = pca.explained_variance_ratio_
    x = ['1', '2', '3', '4', '5',
               '6', '7', '8', '9', '10',
               '11', '12', '13', '14', '15', '16', '17', '18',
               '19', '20', '21', '22', '23',
               '24', '25', '26', '27', '28', '29', '30', '31']
    Sum = 0
    SumVariance = []
    for Variance in VarianceList:
        Sum = Sum + Variance
        SumVariance.append(Sum)
        
    fig = figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(x, SumVariance)
    ax.plot(x, SumVariance)
    #plt.text(x[12], SumVariance[12], '95', fontsize=12)
    ax.annotate('95%', xy=(x[12], SumVariance[12]), xytext=(x[13], 1.05),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

    ax.set_ylim(0.2,1.1)
    ax.set_title('Variance Vs. No. of Componenets')
    plt.savefig('../Plots/PCA.png')
    
def VisualizeTrainingTime():
    TrainingTimes = pd.read_csv('../Models/TrainingTime.csv')
    NCAATimes = TrainingTimes.drop('Mixed', axis = 1)
    X = (NCAATimes['ModelName'])
    Y = (NCAATimes['NCAA'])
    
    fig = figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    for i in range(0, 7):
        string = str(Y[i])
        string = string[0:6]
        ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], float(Y[i]) + 0.003),
                    arrowprops=dict(facecolor='black', shrink=0.05),)
    ax.set_xlabel('Models')
    ax.set_ylabel('Time in Seconds')
    ax.set_title('Training Time on NCAA Dataset')
    plt.savefig('../Plots/NCAATrainingTimes.png')
    
    MixedTimes = TrainingTimes.drop('NCAA', axis = 1)
    X = (MixedTimes['ModelName'])
    Y = (MixedTimes['Mixed'])
    
    fig = figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.scatter(X, Y)
    for i in range(0, 7):
        string = str(Y[i])
        string = string[0:6]
        ax.annotate(string, xy=(X[i], Y[i]), xytext = (X[i], float(Y[i]) + 50),
                    arrowprops=dict(facecolor='black', shrink=0.05),)
    ax.set_xlabel('Models')
    ax.set_ylabel('Time in Seconds')
    ax.set_ylim(-10,340)
    ax.set_title('Training Time on Mixed Dataset')
    plt.savefig('../Plots/MixedTrainingTimes.png')
    
main()