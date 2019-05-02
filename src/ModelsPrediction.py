import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import numpy as np
import time
from threading import Thread
import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
##################################################################################
# Thread Class from Stack Overflow
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
# ************** #
        
def main():
    NCAAData = pd.read_csv('../PreProcessedData/NCAA_Test.csv')
#    RegularData = pd.read_csv('../PreProcessedData/Regular_Test.csv')
    MixedData = pd.read_csv('../PreProcessedData/ReducedMixedDataTest.csv')
    
    MixedData = MixedData.drop(MixedData.columns[0], axis = 1)
#    RegularData = RegularData.drop(RegularData.columns[0], axis = 1)
    NCAAData = NCAAData.drop(NCAAData.columns[0], axis = 1)
    
    DecisionTrees_Mixed, DecisionTrees_NCAA, DecisionTrees_Regular = load_models('DecisionTrees')
    KernelSVM_Mixed, KernelSVM_NCAA, KernelSVM_Regular = load_models('KernelSVM')
    SVM_Mixed, SVM_NCAA, SVM_Regular = load_models('SVM')
    KNN_Mixed, KNN_NCAA, KNN_Regular = load_models('KNN')
    LogisticRegression_Mixed, LogisticRegression_NCAA, LogisticRegression_Regular = load_models('LogisticRegression')
    NaiveBayes_Mixed, NaiveBayes_NCAA, NaiveBayes_Regular = load_models('NaiveBayes')
    RandomForest_Mixed, RandomForest_NCAA, RandomForest_Regular = load_models('RandomForest')
    
    #############################################################################################################
    
    PredictionResults = pd.DataFrame()
    columns = ['Index','ModelName', 'TestData', 'PredictionScore', 'TimeTaken']
    for column in columns:
        PredictionResults[column] = np.nan
    
    t1 = ThreadWithReturnValue(target=GetPredictionResult, args=(DecisionTrees_Mixed, MixedData, 'DecisionTrees', 'Mixed'))
    t2 = ThreadWithReturnValue(target=GetPredictionResult, args=(DecisionTrees_NCAA, NCAAData, 'DecisionTrees', 'NCAA'))
#    t3 = ThreadWithReturnValue(target=GetPredictionResult, args=(DecisionTrees_Regular, RegularData, 'DecisionTrees', 'Regular'))
    
    t1.start()
    t2.start()
#    t3.start()
    
    PredictionResults = PredictionResults.append(t1.join())
    PredictionResults = PredictionResults.append(t2.join())
#    PredictionResults = PredictionResults.append(t3.join())
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(DecisionTrees_Mixed, MixedData, 'DecisionTrees', 'Mixed'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(DecisionTrees_NCAA, NCAAData, 'DecisionTrees', 'NCAA'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(DecisionTrees_Regular, RegularData, 'DecisionTrees', 'Regular'))
    
    t1 = ThreadWithReturnValue(target=GetPredictionResult, args=(KernelSVM_Mixed, MixedData, 'KernelSVM', 'Mixed'))
    t2 = ThreadWithReturnValue(target=GetPredictionResult, args=(KernelSVM_NCAA, NCAAData, 'KernelSVM', 'NCAA'))
#    t3 = ThreadWithReturnValue(target=GetPredictionResult, args=(KernelSVM_Regular, RegularData, 'KernelSVM', 'Regular'))
    
    t1.start()
    t2.start()
#    t3.start()
    
    PredictionResults = PredictionResults.append(t1.join())
    PredictionResults = PredictionResults.append(t2.join())
#    PredictionResults = PredictionResults.append(t3.join())
    
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(KernelSVM_Mixed, MixedData, 'KernelSVM', 'Mixed'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(KernelSVM_NCAA, NCAAData, 'KernelSVM', 'NCAA'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(KernelSVM_Regular, RegularData, 'KernelSVM', 'Regular'))
    
    t1 = ThreadWithReturnValue(target=GetPredictionResult, args=(SVM_Mixed, MixedData, 'SVM', 'Mixed'))
    t2 = ThreadWithReturnValue(target=GetPredictionResult, args=(SVM_NCAA, NCAAData, 'SVM', 'NCAA'))
#    t3 = ThreadWithReturnValue(target=GetPredictionResult, args=(SVM_Regular, RegularData, 'SVM', 'Regular'))
    
    t1.start()
    t2.start()
#    t3.start()
    
    PredictionResults = PredictionResults.append(t1.join())
    PredictionResults = PredictionResults.append(t2.join())
#    PredictionResults = PredictionResults.append(t3.join())
    
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(SVM_Mixed, MixedData, 'SVM', 'Mixed'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(SVM_NCAA, NCAAData, 'SVM', 'NCAA'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(SVM_Regular, RegularData, 'SVM', 'Regular'))
    
    t1 = ThreadWithReturnValue(target=GetPredictionResult, args=(KNN_Mixed, MixedData, 'KNN', 'Mixed'))
    t2 = ThreadWithReturnValue(target=GetPredictionResult, args=(KNN_NCAA, NCAAData, 'KNN', 'NCAA'))
#    t3 = ThreadWithReturnValue(target=GetPredictionResult, args=(KNN_Regular, RegularData, 'KNN', 'Regular'))
    
    t1.start()
    t2.start()
#    t3.start()
    
    PredictionResults = PredictionResults.append(t1.join())
    PredictionResults = PredictionResults.append(t2.join())
#    PredictionResults = PredictionResults.append(t3.join())
    
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(KNN_Mixed, MixedData, 'KNN', 'Mixed'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(KNN_NCAA, NCAAData, 'KNN', 'NCAA'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(KNN_Regular, RegularData, 'KNN', 'Regular'))
    
    t1 = ThreadWithReturnValue(target=GetPredictionResult, args=(LogisticRegression_Mixed, MixedData, 'LogisticRegression', 'Mixed'))
    t2 = ThreadWithReturnValue(target=GetPredictionResult, args=(LogisticRegression_NCAA, NCAAData, 'LogisticRegression', 'NCAA'))
#    t3 = ThreadWithReturnValue(target=GetPredictionResult, args=(LogisticRegression_Regular, RegularData, 'LogisticRegression', 'Regular'))
    
    t1.start()
    t2.start()
#    t3.start()
    
    PredictionResults = PredictionResults.append(t1.join())
    PredictionResults = PredictionResults.append(t2.join())
#    PredictionResults = PredictionResults.append(t3.join())
    
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(LogisticRegression_Mixed, MixedData, 'LogisticRegression', 'Mixed'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(LogisticRegression_NCAA, NCAAData, 'LogisticRegression', 'NCAA'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(LogisticRegression_Regular, RegularData, 'LogisticRegression', 'Regular'))
    
    t1 = ThreadWithReturnValue(target=GetPredictionResult, args=(NaiveBayes_Mixed, MixedData, 'NaiveBayes', 'Mixed'))
    t2 = ThreadWithReturnValue(target=GetPredictionResult, args=(NaiveBayes_NCAA, NCAAData, 'NaiveBayes', 'NCAA'))
#    t3 = ThreadWithReturnValue(target=GetPredictionResult, args=(NaiveBayes_Regular, RegularData, 'NaiveBayes', 'Regular'))
    
    t1.start()
    t2.start()
#    t3.start()
    
    PredictionResults = PredictionResults.append(t1.join())
    PredictionResults = PredictionResults.append(t2.join())
#    PredictionResults = PredictionResults.append(t3.join())
    
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(NaiveBayes_Mixed, MixedData, 'NaiveBayes', 'Mixed'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(NaiveBayes_NCAA, NCAAData, 'NaiveBayes', 'NCAA'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(NaiveBayes_Regular, RegularData, 'NaiveBayes', 'Regular'))
    
    t1 = ThreadWithReturnValue(target=GetPredictionResult, args=(RandomForest_Mixed, MixedData, 'RandomForest', 'Mixed'))
    t2 = ThreadWithReturnValue(target=GetPredictionResult, args=(RandomForest_NCAA, NCAAData, 'RandomForest', 'NCAA'))
#    t3 = ThreadWithReturnValue(target=GetPredictionResult, args=(RandomForest_Regular, RegularData, 'RandomForest', 'Regular'))
    
    t1.start()
    t2.start()
#    t3.start()
    
    PredictionResults = PredictionResults.append(t1.join())
    PredictionResults = PredictionResults.append(t2.join())
#    PredictionResults = PredictionResults.append(t3.join())
    
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(RandomForest_Mixed, MixedData, 'RandomForest', 'Mixed'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(RandomForest_NCAA, NCAAData, 'RandomForest', 'NCAA'))
    #PredcitionResutls = PredictionResults.append(GetPredictionResult(RandomForest_Regular, RegularData, 'RandomForest', 'Regular'))
    
   
        #############################################################################################################
        
    NCAARound1Dataframe, NCAARound2Dataframe, NCAARound3Dataframe, NCAARound4Dataframe, NCAARound5Dataframe, NCAAFinalDataframe, MixedRound1Dataframe, MixedRound2Dataframe, MixedRound3Dataframe, MixedRound4Dataframe, MixedRound5Dataframe, MixedFinalDataframe = generate2019TestDataset()
    
    chunks = [MixedRound1Dataframe, MixedRound2Dataframe, MixedRound3Dataframe, MixedRound4Dataframe, MixedRound5Dataframe, MixedFinalDataframe]
    MixedDataframe = pd.concat(chunks, ignore_index = True)
    MixedDataframe = getReducedDf(MixedDataframe)
    
    chunks = [NCAARound1Dataframe, NCAARound2Dataframe, NCAARound3Dataframe, NCAARound4Dataframe, NCAARound5Dataframe, NCAAFinalDataframe]
    NCAADataframe = pd.concat(chunks, ignore_index = True)
    
    ModelsListMixed = [(DecisionTrees_Mixed, 'DecisionTrees'), (KernelSVM_Mixed, 'KernelSVM'), (SVM_Mixed, 'SVM'),  (KNN_Mixed, 'KNN'),  (LogisticRegression_Mixed, 'LogisticRegression'), (NaiveBayes_Mixed, 'NaiveBayes'), (RandomForest_Mixed, 'RandomForest')]
    
    ModelsListNCAA = [ (DecisionTrees_NCAA, 'DecisionTrees'), (KernelSVM_NCAA, 'KernelSVM'), (SVM_NCAA, 'SVM'), (KNN_NCAA, 'KNN'), (LogisticRegression_NCAA, 'LogisticRegression'), (NaiveBayes_NCAA, 'NaiveBayes'), (RandomForest_NCAA , 'RandomForest')]
    
    PredictionResultsDetailed = pd.DataFrame()
    columns = ['Match No.','ModelName', 'DataType' ,'Expected Result', 'Predicted Result']
    for column in columns:
        PredictionResultsDetailed[column] = np.nan
    
    for model in ModelsListMixed:
        compactResults, DetailedResults = get2019CompactPredictions(model[0],model[1], MixedDataframe, 'Mixed 2019')
        PredictionResults = PredictionResults.append(compactResults)
        PredictionResultsDetailed = PredictionResultsDetailed.append(DetailedResults, ignore_index = True)
    for model in ModelsListNCAA:
        compactResults, DetailedResults = get2019CompactPredictions(model[0],model[1], NCAADataframe, 'NCAA 2019')
        PredictionResults = PredictionResults.append(compactResults)
        PredictionResultsDetailed = PredictionResultsDetailed.append(DetailedResults,ignore_index = True)
        
    PredictionResults.to_csv('../Results/PredictionsResult.csv')
    PredictionResultsDetailed.to_csv('../Results/PredictionsResultDetailed.csv')
    

def getReducedDf(MixedDataframe):
    Y = MixedDataframe['DiffResult']
    X = MixedDataframe.drop('DiffResult', axis = 1)
    pca = PCA(n_components = 13)
    pca.fit(X)
    X = pca.transform(X)
    MixedDataframe = pd.DataFrame(X)
    MixedDataframe['DiffResult'] = Y
    
    return MixedDataframe


def get2019CompactPredictions(model,modelName, df, datatype):
    PredictionResultsCompact = pd.DataFrame()
    columns = ['ModelName','PredictionScore', 'TestData', 'TimeTaken']
    for column in columns:
        PredictionResultsCompact[column] = np.nan
    if datatype == 'NCAA 2019':
        df = df.drop('MatchType', axis = 1)
    Y = df['DiffResult']
    X = df.drop('DiffResult', axis = 1) 
    start = time.time()
    Y_Pred = model.predict(X)
    end = time.time() - start
    Score = accuracy_score(Y, Y_Pred)
    PredictionResultsCompact.loc[len(PredictionResultsCompact)] = [modelName, Score, datatype ,end]
    PredictionResultsDetailed = pd.DataFrame()
    columns = ['Match No.','ModelName', 'DataType' ,'Expected Result', 'Predicted Result']
    for column in columns:
        PredictionResultsDetailed[column] = np.nan
    for i in range(len(Y_Pred)):
        PredictionResultsDetailed.loc[len(PredictionResultsDetailed)] = [i+1,modelName,datatype ,Y[i], Y_Pred[i]]
    return (PredictionResultsCompact, PredictionResultsDetailed)
    
def GetPredictionResult(Model, TestData, ModelName, DataName):
    Y = TestData['DiffResult']
    X = TestData.drop('DiffResult', axis = 1)
    start = time.time()
    Y_Pred = Model.predict(X)
    end = time.time() - start
    Score = accuracy_score(Y, Y_Pred)
    df = pd.DataFrame()
    df['ModelName'] = np.nan
    df['TestData'] = np.nan
    df['PredictionScore'] = np.nan
    df['TimeTaken'] = np.nan
    temp = [ ModelName, DataName, Score, end]
    df.loc[0] = temp
    return df
    

def load_models(ModelName):
    ModelName = '../Models/' + ModelName
    MixedModelName = ModelName + '_Mixed.pkl'
    NCAAModelName = ModelName + '_NCAA.pkl'
    RegularModelName = ModelName + '_Mixed.pkl'
    
    MixedModel = joblib.load(MixedModelName)
    NCAAModel = joblib.load(NCAAModelName)
    RegularModel = joblib.load(RegularModelName)
    
    return (MixedModel, NCAAModel, RegularModel)

def generate2019TestDataset():
    Teams = pd.read_csv('../DataFiles/Teams.csv')
    Teams = Teams.drop('FirstD1Season' , axis = 1)
    Teams = Teams.drop('LastD1Season' , axis = 1)
    
    Round1Matches = getRound1Matches(Teams)
    Round2Matches = getRound2Matches(Teams)
    Round3Matches = getRound3Matches(Teams)
    Round4Matches = getRound4Matches(Teams)
    Round5Matches = getRound5Matches(Teams)
    finalmatch = getFinalMatch(Teams)
    
    with open('../PreProcessedData/NCAATeamDictionary.pkl', 'rb') as pickle_file:
        NCAATeamDictionary = pickle.load(pickle_file)
    with open('../PreProcessedData/RegularTeamDictionary.pkl', 'rb') as pickle_file:
        RegualarTeamDictionary = pickle.load(pickle_file)
    
    NCAARound1Dataframe = getNCAADataframe(Round1Matches, NCAATeamDictionary)
    NCAARound2Dataframe = getNCAADataframe(Round2Matches, NCAATeamDictionary)
    NCAARound3Dataframe = getNCAADataframe(Round3Matches, NCAATeamDictionary)
    NCAARound4Dataframe = getNCAADataframe(Round4Matches, NCAATeamDictionary)
    NCAARound5Dataframe = getNCAADataframe(Round5Matches, NCAATeamDictionary)
    NCAAFinalDataframe = getNCAADataframe(finalmatch, NCAATeamDictionary)
    
    MixedRound1Dataframe = NCAARound1Dataframe
    MixedRound1Dataframe.insert(loc = 0, column = 'MatchType', value = 1)
    MixedRound2Dataframe = NCAARound2Dataframe
    MixedRound2Dataframe.insert(loc = 0, column = 'MatchType', value = 1)
    MixedRound3Dataframe = NCAARound3Dataframe
    MixedRound3Dataframe.insert(loc = 0, column = 'MatchType', value = 1)
    MixedRound4Dataframe = NCAARound4Dataframe
    MixedRound4Dataframe.insert(loc = 0, column = 'MatchType', value = 1)
    MixedRound5Dataframe = NCAARound5Dataframe
    MixedRound5Dataframe.insert(loc = 0, column = 'MatchType', value = 1)
    MixedFinalDataframe = NCAAFinalDataframe
    MixedFinalDataframe.insert(loc = 0, column = 'MatchType', value = 1)
    
    
    return (NCAARound1Dataframe, NCAARound2Dataframe, NCAARound3Dataframe, NCAARound4Dataframe, NCAARound5Dataframe, NCAAFinalDataframe,
            MixedRound1Dataframe, MixedRound2Dataframe, MixedRound3Dataframe, MixedRound4Dataframe, MixedRound5Dataframe, MixedFinalDataframe)
    

    
def getNCAADataframe(Matches, NCAATeamDictionary):
    
    Dataframe = pd.DataFrame()
    columns = ['DiffNumOfWins','DiffAvgWScore','DiffNumofLosses','DiffAvgLScore',
               'DiffAvgWFGM','DiffAvgWFGA','DiffAvgWFGM3','DiffAvgWFGA3','DiffAvgWFTM',
               'DiffAvgWFTA','DiffAvgWOR','DiffAvgWDR','DiffAvgWAst','DiffAvgWTO','DiffAvgWStl',
               'DiffAvgWBlk','DiffAvgWPF','DiffAvgLFGM','DiffAvgLFGA','DiffAvgLFGM3','DiffAvgLFGA3',
               'DiffAvgLFTM','DiffAvgLFTA','DiffAvgLOR','DiffAvgLDR','DiffAvgLAst',
               'DiffAvgLTO','DiffAvgLStl','DiffAvgLBlk','DiffAvgLPF', 'DiffResult']
    
    for column in columns:
        Dataframe[column] = np.nan
    
    #print(Matches)
    
    for index, match in Matches.iterrows():
        #print(match['WTeamID'])
        #print('STUPID \n \n')
        WDf = NCAATeamDictionary.get(match['WTeamID'])
        LDf = NCAATeamDictionary.get(match['LTeamID'])
        if LDf is None:
            WVect = WDf[WDf['Season'] == max(WDf['Season'])]
            WVect = WVect.drop('Season', axis = 1)
            WVect['Result'] = 1
            Result = WVect.values
            Dataframe.loc[len(Dataframe)] = Result[0]
            continue
        WVect = WDf[WDf['Season'] == max(WDf['Season'])]
        LVect = LDf[LDf['Season'] == max(LDf['Season'])]
        WVect = WVect.drop('Season', axis = 1)
        LVect = LVect.drop('Season', axis = 1)
        WVect['Result'] = 1
        LVect['Result'] = 0
        Result = WVect.values - LVect.values
        Dataframe.loc[len(Dataframe)] = Result[0]
    
    return Dataframe
    
def getFinalMatch(Teams):
    Round4Matches = [
            ('Virginia', 'Texas Tech'),
            ]
    
    df = pd.DataFrame(columns = ['WTeamID', 'LTeamID'])
    for match in Round4Matches:
        WTeam = match[0]
        LTeam = match[1]
        Wcheck = Teams[Teams['TeamName'] == WTeam]
        Lcheck = Teams[Teams['TeamName'] == LTeam]
        df.loc[len(df)] = [Wcheck['TeamID'].values[0], Lcheck['TeamID'].values[0]]
    df.to_csv('../NCAA2019/Final.csv')    
    return df 

def getRound5Matches(Teams):
    Round4Matches = [
            ('Texas Tech', 'Michigan St'),
            ('Virginia', 'Auburn'),
            ]
    
    df = pd.DataFrame(columns = ['WTeamID', 'LTeamID'])
    for match in Round4Matches:
        WTeam = match[0]
        LTeam = match[1]
        Wcheck = Teams[Teams['TeamName'] == WTeam]
        Lcheck = Teams[Teams['TeamName'] == LTeam]
        df.loc[len(df)] = [Wcheck['TeamID'].values[0], Lcheck['TeamID'].values[0]]
    df.to_csv('../NCAA2019/Round5.csv')
    return df 

def getRound4Matches(Teams):
    Round4Matches = [
            ('Michigan St', 'Duke'),
            ('Texas Tech', 'Gonzaga'),
            ('Virginia', 'Purdue'),
            ('Auburn', 'Kentucky'),
            ]
    
    df = pd.DataFrame(columns = ['WTeamID', 'LTeamID'])
    for match in Round4Matches:
        WTeam = match[0]
        LTeam = match[1]
        Wcheck = Teams[Teams['TeamName'] == WTeam]
        Lcheck = Teams[Teams['TeamName'] == LTeam]
        df.loc[len(df)] = [Wcheck['TeamID'].values[0], Lcheck['TeamID'].values[0]]
    
    df.to_csv('../NCAA2019/Round4.csv')
    return df 

def getRound3Matches(Teams):
    Round3Matches = [
            ('Duke', 'Virginia Tech'),
            ('Michigan St', 'LSU'),
            ('Gonzaga', 'Florida St'),
            ('Texas Tech', 'Michigan'),
            ('Virginia', 'Oregon'),
            ('Purdue', 'Tennessee'),
            ('Auburn', 'North Carolina'),
            ('Kentucky', 'Houston'),
            ]
    
    df = pd.DataFrame(columns = ['WTeamID', 'LTeamID'])
    for match in Round3Matches:
        WTeam = match[0]
        LTeam = match[1]
        Wcheck = Teams[Teams['TeamName'] == WTeam]
        Lcheck = Teams[Teams['TeamName'] == LTeam]
        df.loc[len(df)] = [Wcheck['TeamID'].values[0], Lcheck['TeamID'].values[0]]
    df.to_csv('../NCAA2019/Round3.csv') 
    return df 


def getRound2Matches(Teams):
    Round2Matches = [
            ('Duke', 'UCF'),
            ('Virginia Tech', 'Liberty'),
            ('LSU', 'Maryland'),
            ('Michigan St', 'Minnesota'),
            ('Gonzaga', 'Baylor'),
            ('Florida St', 'Murray St'),
            ('Texas Tech', 'Buffalo'),
            ('Michigan', 'Florida'),
            ('Virginia', 'Oklahoma'),
            ('Oregon', 'UC Irvine'),
            ('Purdue', 'Villanova'),
            ('Tennessee', 'Iowa'),
            ('North Carolina', 'Washington'),
            ('Auburn', 'Kansas'),
            ('Houston', 'Ohio St'),
            ('Kentucky', 'Wofford')
            ]
    
    df = pd.DataFrame(columns = ['WTeamID', 'LTeamID'])
    for match in Round2Matches:
        WTeam = match[0]
        LTeam = match[1]
        Wcheck = Teams[Teams['TeamName'] == WTeam]
        Lcheck = Teams[Teams['TeamName'] == LTeam]
        df.loc[len(df)] = [Wcheck['TeamID'].values[0], Lcheck['TeamID'].values[0]]
    df.to_csv('../NCAA2019/Round2.csv')   
    return df 

def getRound1Matches(Teams):
    Round1Matches = [
            ('Duke', 'North Dakota'),
            ('UCF', 'Virginia'),
            ('Liberty', 'Mississippi St'),
            ('Virginia Tech', 'St Louis'),
            ('Maryland', 'Belmont'),
            ('LSU', 'Yale'),
            ('Minnesota', 'Louisville'),
            ('Michigan St', 'Bradley'),
            ('Gonzaga', 'F Dickinson'),
            ('Baylor', 'Syracuse'),
            ('Murray St', 'Marquette'),
            ('Florida St', 'Vermont'),
            ('Buffalo', 'Arizona St'),
            ('Texas Tech', 'N Kentucky'),
            ('Florida', 'Nevada'),
            ('Michigan', 'Montana'),
            ('Virginia', 'Gardner Webb'),
            ('Oklahoma', 'Mississippi'),
            ('Oregon', 'Wisconsin'),
            ('UC Irvine', 'Kansas St'),
            ('Villanova', 'St Mary\'s CA'),
            ('Purdue', 'Old Dominion'),
            ('Iowa', 'Cincinnati'),
            ('Tennessee', 'Colgate'),
            ('North Carolina', 'Iona'),
            ('Washington', 'Utah St'),
            ('Auburn', 'New Mexico St'),
            ('Kansas', 'Northeastern'),
            ('Ohio St', 'Iowa St'),
            ('Houston', 'Georgia St'),
            ('Wofford', 'Seton Hall'),
            ('Kentucky', 'Abilene Chr')
            ]
    
    df = pd.DataFrame(columns = ['WTeamID', 'LTeamID'])
    for match in Round1Matches:
        WTeam = match[0]
        LTeam = match[1]
        Wcheck = Teams[Teams['TeamName'] == WTeam]
        Lcheck = Teams[Teams['TeamName'] == LTeam]
        df.loc[len(df)] = [Wcheck['TeamID'].values[0], Lcheck['TeamID'].values[0]]
    df.to_csv('../NCAA2019/Round1.csv')
    return df 
    
    
main()