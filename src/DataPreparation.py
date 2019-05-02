#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from random import seed
from random import randint
from threading import Thread
from multiprocessing.pool import ThreadPool
import time
from sklearn.model_selection import train_test_split
import pickle
# seed random number generator
seed(1)


# *************** #
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
    
    start = time.time()
    TrainingDataframeRegularSeason = getTrainingDataset('Regular')
    RegularSeason = time.time()
    print('Time Taken By Regular season:  ', RegularSeason - start)
    TrainingDataframeNCAASeason = getTrainingDataset('NCAA')
    NCAA = time.time()
    print('Time Taken By NCAA:  ', NCAA - RegularSeason)
    
    # Saving to csv for easy reading
    TrainingDataframeRegularSeason.to_csv('../PreProcessedData/RegularSeasonPreProcessedData.csv')
    TrainingDataframeNCAASeason.to_csv('../PreProcessedData/NCAAPreProcessedData.csv')
    
    NCAADataset = pd.read_csv('../PreProcessedData/NCAAPreProcessedData.csv')
    RegularDataset = pd.read_csv('../PreProcessedData/RegularSeasonPreProcessedData.csv')
    
    NCAADataset = NCAADataset.drop(NCAADataset.columns[0], axis = 1)
    RegularDataset = RegularDataset.drop(RegularDataset.columns[0], axis = 1)
    
    NCAA_Train, NCAA_Test = train_test_split(NCAADataset, test_size = 0.25, random_state = 0)
    Regular_Train, Regular_Test = train_test_split(RegularDataset, test_size = 0.25, random_state = 0)
    
    NCAA_Train.to_csv('../PreProcessedData/NCAA_Train.csv')
    NCAA_Test.to_csv('../PreProcessedData/NCAA_Test.csv')
    
    Regular_Train.to_csv('../PreProcessedData/Regular_Train.csv')
    Regular_Test.to_csv('../PreProcessedData/Regular_Test.csv')
    
    Regular_Train.insert(loc=0, column='MatchType', value=0)
    Regular_Test.insert(loc=0, column='MatchType', value=0)
    NCAA_Test.insert(loc=0, column='MatchType', value=1)
    NCAA_Train.insert(loc=0, column='MatchType', value=1)
    
    
    MixedTrainingDataset = Regular_Train.append(NCAA_Train)
    MixedTestDataset = Regular_Test.append(NCAA_Test)
    MixedTestDataset.to_csv('../PreProcessedData/Mixed_Test.csv')
    MixedTrainingDataset.to_csv('../PreProcessedData/Mixed_Train.csv')
    
def getTrainingDataset(filename):
    
    if filename == 'NCAA':
        DetailedResults = pd.read_csv('NCAATourneyDetailedResults.csv')
        CompactResults = pd.read_csv('NCAATourneyCompactResults.csv')
    else:
        DetailedResults = pd.read_csv('RegularSeasonDetailedResults.csv')
        CompactResults = pd.read_csv('RegularSeasonCompactResults.csv')
    
    
    DetailedResults = DetailedResults.drop('DayNum', axis = 1)
    Teams = pd.read_csv('Teams.csv')
    Teams = Teams.drop('FirstD1Season', axis = 1)
    Teams = Teams.drop('LastD1Season', axis = 1)
    
    TeamsDictionary = PreprocessData(DetailedResults, Teams)
    
    if filename == 'NCAA':
        f = open("../PreProcessedData/NCAATeamDictionary.pkl", "wb")
        pickle.dump(TeamsDictionary, f)
    else:
        f = open("../PreProcessedData/RegularTeamDictionary.pkl", "wb")
        pickle.dump(TeamsDictionary, f)
    
    MatchResults = DetailedResults[['Season', 'WTeamID', 'LTeamID']]
    
    # *********** Separate into 4 different threads ************//
    M1, M2, M3, M4 = np.array_split(MatchResults, 4)
    
    t1 = ThreadWithReturnValue(target=getTrainingDataframe, args=(TeamsDictionary, M1))
    t2 = ThreadWithReturnValue(target=getTrainingDataframe, args=(TeamsDictionary, M2))
    t3 = ThreadWithReturnValue(target=getTrainingDataframe, args=(TeamsDictionary, M3))
    t4 = ThreadWithReturnValue(target=getTrainingDataframe, args=(TeamsDictionary, M4))
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    
    T1 = t1.join()
    T2 = t2.join()
    T3 = t3.join()
    T4 = t4.join()
    #TrainingDataframeNCAASeason = getTrainingDataframe(TeamsDictionaryNCAASeason, MatchResultsNCAASeason)
    TrainingDataframe = T1
    TrainingDataframe = TrainingDataframe.append(T2)
    TrainingDataframe = TrainingDataframe.append(T3)
    TrainingDataframe = TrainingDataframe.append(T4)
    return TrainingDataframe

    
def getDifference(TeamsDictionary,Year,WTeamID,LTeamID,flag):
    Wdataframe = TeamsDictionary.get(WTeamID, "Not Found")
    Wvector = Wdataframe[Wdataframe['Season'] == Year]
    Ldataframe = TeamsDictionary.get(LTeamID, "Not Found")
    Lvector = Ldataframe[Ldataframe['Season'] == Year]
    if flag == 0:
        Diff = Wvector.values - Lvector.values
        Diff = np.delete(Diff, 0, 1)
        Diff = np.append(Diff, 1)
    else:
        Diff =  Lvector.values - Wvector.values
        Diff = np.delete(Diff, 0, 1)
        Diff = np.append(Diff, 0)
    return Diff

def getTrainingDataframe(TeamsDictionary, MatchResults):
    TrainingDataset = pd.DataFrame()
    columns = ['DiffNumOfWins','DiffAvgWScore','DiffNumofLosses','DiffAvgLScore',
               'DiffAvgWFGM','DiffAvgWFGA','DiffAvgWFGM3','DiffAvgWFGA3','DiffAvgWFTM',
               'DiffAvgWFTA','DiffAvgWOR','DiffAvgWDR','DiffAvgWAst','DiffAvgWTO','DiffAvgWStl',
               'DiffAvgWBlk','DiffAvgWPF','DiffAvgLFGM','DiffAvgLFGA','DiffAvgLFGM3','DiffAvgLFGA3',
               'DiffAvgLFTM','DiffAvgLFTA','DiffAvgLOR','DiffAvgLDR','DiffAvgLAst',
               'DiffAvgLTO','DiffAvgLStl','DiffAvgLBlk','DiffAvgLPF', 'DiffResult']
    
    for column in columns:
        TrainingDataset[column] = np.nan
    for index, row in MatchResults.iterrows():
        flag = randint(0,1)
        TrainingDataset.loc[len(TrainingDataset)] = getDifference(TeamsDictionary,
                                                               row['Season'],
                                                               row['WTeamID'],
                                                               row['LTeamID'],
                                                               flag)
    return TrainingDataset

def PreprocessData(DetailedResults, Teams):
    TeamDictionary = {}
    for TeamID in Teams['TeamID']:
        TeamDataFrame = separateByTeam(TeamID, DetailedResults)
        if len(TeamDataFrame.index) != 0:
            TeamDictionary[TeamID] = CreateTeamDataset(TeamID, TeamDataFrame)
        
    return TeamDictionary

def separateByTeam(TeamID, DetailedResults):
    TeamDataFrame = DetailedResults[DetailedResults['WTeamID'] == TeamID]
    TeamDataFrame = TeamDataFrame.append(DetailedResults[DetailedResults['LTeamID'] == TeamID])
    return TeamDataFrame

def separateByYears(year, TeamDataFrame):
    df = TeamDataFrame[TeamDataFrame['Season'] == year]
    return df

def CreateTeamDataset(TeamID, TeamDataFrame):
    minYear = (TeamDataFrame['Season'].min())
    maxYear = (TeamDataFrame['Season'].max())
    columns = ['Season','NumOfWins','AvgWScore','NumofLosses','AvgLScore',
               'AvgWFGM','AvgWFGA','AvgWFGM3','AvgWFGA3','AvgWFTM',
               'AvgWFTA','AvgWOR','AvgWDR','AvgWAst','AvgWTO','AvgWStl',
               'AvgWBlk','AvgWPF','AvgLFGM','AvgLFGA','AvgLFGM3','AvgLFGA3',
               'AvgLFTM','AvgLFTA','AvgLOR','AvgLDR','AvgLAst',
               'AvgLTO','AvgLStl','AvgLBlk','AvgLPF']
    
    TeamDataset = pd.DataFrame()
    for column in columns:
        TeamDataset[column] = np.nan
    for year in range(minYear, maxYear+1):
        YearDataset = separateByYears(year, TeamDataFrame)
        TeamDataset.loc[len(TeamDataset)] = getFeatureVector(YearDataset, TeamID, year)
    
    return TeamDataset

def getFeatureVector(YearDataset, TeamID, year):
    
    WinningMatches = YearDataset[YearDataset['WTeamID'] == TeamID]
    LosingMatches = YearDataset[YearDataset['LTeamID'] == TeamID]
    # ************************************************************ #
    Season = year
    if len(WinningMatches.index) != 0:
        NumOfWins = len(WinningMatches.Season)
        AvgWScore = sum(WinningMatches.WScore) / len(WinningMatches.WScore)
        AvgWFGM = sum(WinningMatches.WFGM) / len(WinningMatches.WFGM)
        AvgWFGA = sum(WinningMatches.WFGA) / len(WinningMatches.WFGA)
        AvgWFGM3 = sum(WinningMatches.WFGM3) / len(WinningMatches.WFGM3)
        AvgWFGA3 = sum(WinningMatches.WFGA3) / len(WinningMatches.WFGA3)
        AvgWFTM = sum(WinningMatches.WFTM) / len(WinningMatches.WFTM)
        AvgWFTA = sum(WinningMatches.WFGA3) / len(WinningMatches.WFGA3)
        AvgWOR = sum(WinningMatches.WOR) / len(WinningMatches.WOR)
        AvgWDR = sum(WinningMatches.WDR) / len(WinningMatches.WDR)
        AvgWAst = sum(WinningMatches.WAst) / len(WinningMatches.WAst)
        AvgWTO = sum(WinningMatches.WTO) / len(WinningMatches.WTO)
        AvgWStl = sum(WinningMatches.WStl) / len(WinningMatches.WStl)
        AvgWBlk = sum(WinningMatches.WBlk) / len(WinningMatches.WBlk)
        AvgWPF = sum(WinningMatches.WPF) / len(WinningMatches.WPF)
    else:
        NumOfWins = 0
        AvgWScore = 0
        AvgWFGM = 0
        AvgWFGA = 0
        AvgWFGM3 = 0
        AvgWFGA3 = 0
        AvgWFTM = 0
        AvgWFTA = 0
        AvgWOR = 0
        AvgWDR = 0
        AvgWAst = 0
        AvgWTO = 0
        AvgWStl = 0
        AvgWBlk = 0
        AvgWPF = 0
    
    if len(LosingMatches.index) != 0:
        NumOfLosses = len(LosingMatches.Season)
        AvgLScore = sum(LosingMatches.LScore) / len(LosingMatches.LScore)
        AvgLFGM = sum(LosingMatches.LFGM) / len(LosingMatches.LFGM)
        AvgLFGA = sum(LosingMatches.LFGA) / len(LosingMatches.LFGA)
        AvgLFGM3 = sum(LosingMatches.LFGM3) / len(LosingMatches.LFGM3)
        AvgLFGA3 = sum(LosingMatches.LFGA3) / len(LosingMatches.LFGA3)
        AvgLFTM = sum(LosingMatches.LFTM) / len(LosingMatches.LFTM)
        AvgLFTA = sum(LosingMatches.LFTA) / len(LosingMatches.LFTA)
        AvgLOR = sum(LosingMatches.LOR) / len(LosingMatches.LOR)
        AvgLDR = sum(LosingMatches.LDR) / len(LosingMatches.LDR)
        AvgLAst = sum(LosingMatches.LAst) / len(LosingMatches.LAst)
        AvgLTO = sum(LosingMatches.LTO) / len(LosingMatches.LTO)
        AvgLStl = sum(LosingMatches.LStl) / len(LosingMatches.LStl)
        AvgLBlk = sum(LosingMatches.LBlk) / len(LosingMatches.LBlk)
        AvgLPF = sum(LosingMatches.LPF) / len(LosingMatches.LPF)
    
    else:
        NumOfLosses = 0
        AvgLScore = 0
        AvgLFGM = 0
        AvgLFGA = 0
        AvgLFGM3 = 0
        AvgLFGA3 = 0
        AvgLFTM = 0
        AvgLFTA = 0
        AvgLOR = 0
        AvgLDR = 0
        AvgLAst = 0
        AvgLTO = 0
        AvgLStl = 0
        AvgLBlk = 0
        AvgLPF = 0
    # ************************************************************ #
    
    vector = [Season,NumOfWins, AvgWScore, NumOfLosses,AvgLScore,AvgWFGM,
              AvgWFGA,AvgWFGM3,AvgWFGA3,AvgWFTM,AvgWFTA,AvgWOR,AvgWDR,
              AvgWAst,AvgWTO,AvgWStl,AvgWBlk,AvgWPF,AvgLFGM,AvgLFGA,
              AvgLFGM3,AvgLFGA3,AvgLFTM,AvgLFTA,AvgLOR,AvgLDR,AvgLAst,
              AvgLTO,AvgLStl,AvgLBlk,AvgLPF]
    return vector
        

main()


'''
Time Taken By Regular and NCAA to Prepare

Time Taken By Regular season ((With 4 threads)):   196.87859892845154

Time Taken By NCAA (With 4 threads):   9.151640176773071

'''