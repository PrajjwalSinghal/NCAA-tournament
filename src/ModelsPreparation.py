import pandas as pd
from threading import Thread
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import time
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
##################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
        
# t1 = ThreadWithReturnValue(target=functionname, args=(parameter 1, parameter 2))
# t1.start()
# T1 = t1.join()

def main():
    
    createMixedTrainAndTestDataset()
    file = open("../Models/TimeTakenByModel.txt", "w")
    
    #################################################################################
    #########################   Coompleting Mixed Dataset First #####################
    t1 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/ReducedMixedDataTrain.csv', 'LogisticRegression'))
    t2 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/ReducedMixedDataTrain.csv', 'KNN'))
    t3 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/ReducedMixedDataTrain.csv', 'SVM'))
    t4 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/ReducedMixedDataTrain.csv', 'KernelSVM'))
    t5 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/ReducedMixedDataTrain.csv', 'NaiveBayes'))
    t6 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/ReducedMixedDataTrain.csv', 'RandomForest'))
    t7 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/ReducedMixedDataTrain.csv', 'DecisionTrees'))
        
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    
    LogisticRegression_Mixed, TimeTaken = t1.join()
    file.write('LogisticRegression_Mixed Time = ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(LogisticRegression_Mixed, '../Models/LogisticRegression_Mixed.pkl')
    
    KNN_Mixed, TimeTaken = t2.join()
    file.write('KNN_Mixed Time = ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(KNN_Mixed, '../Models/KNN_Mixed.pkl')
    
    SVM_Mixed, TimeTaken = t3.join()
    file.write('SVM_Mixed Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(SVM_Mixed, '../Models/SVM_Mixed.pkl')
    
    KernelSVM_Mixed, TimeTaken = t4.join()
    file.write('KernelSVM_Mixed Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(KernelSVM_Mixed, '../Models/KernelSVM_Mixed.pkl')
    
    NaiveBayes_Mixed, TimeTaken = t5.join()
    file.write('NaiveBayes_Mixed Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(NaiveBayes_Mixed, '../Models/NaiveBayes_Mixed.pkl')
    
    RandomForest_Mixed, TimeTaken = t6.join()
    file.write('RandomForest_Mixed Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(RandomForest_Mixed, '../Models/RandomForest_Mixed.pkl')
    
    DecisionTrees_Mixed, TimeTaken = t7.join()
    file.write('DecisionTrees_Mixed Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(DecisionTrees_Mixed, '../Models/DecisionTrees_Mixed.pkl')
    #################################################################################
    
    t1 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/NCAA_Train.csv', 'LogisticRegression'))
#    t2 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/Regular_Train.csv', 'LogisticRegression'))
    t3 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/NCAA_Train.csv', 'KNN'))
#    t4 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/Regular_Train.csv', 'KNN'))
    
    t1.start()
#    t2.start()
    t3.start()
#    t4.start()
    
    
    LogisticRegression_NCAA, TimeTaken = t1.join()
    file.write('LogisticRegression_NCAA Time = ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(LogisticRegression_NCAA, '../Models/LogisticRegression_NCAA.pkl')
    
#    LogisticRegression_Regular, TimeTaken = t2.join()
#    file.write('LogisticRegression_Regular Time = ')
#    file.write(TimeTaken)
#    file.write('\n')
#    file.flush()
#    joblib.dump(LogisticRegression_Regular, '../Models/LogisticRegression_Regular.pkl')
    
    KNN_NCAA, TimeTaken = t3.join()
    file.write('KNN_NCAA Time = ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(KNN_NCAA, '../Models/KNN_NCAA.pkl')
    
#    KNN_Regular, TimeTaken = t4.join()
#    file.write('KNN_Regular Time = ')
#    file.write(TimeTaken)
#    file.write('\n')
#    file.flush()
#    joblib.dump(KNN_Regular, '../Models/KNN_Regular.pkl')
    
    ##################################################################################
 
    t1 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/NCAA_Train.csv', 'SVM'))
#    t2 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/Regular_Train.csv', 'SVM'))
    t3 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/NCAA_Train.csv', 'KernelSVM'))
#    t4 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/Regular_Train.csv', 'KernelSVM'))
    
    t1.start()
#    t2.start()
    t3.start()
#    t4.start()
    
    SVM_NCAA, TimeTaken = t1.join()
    file.write('SVM_NCAA Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(SVM_NCAA, '../Models/SVM_NCAA.pkl')
    
#    SVM_Regular, TimeTaken = t2.join()
#    file.write('SVM_Regular Time: ')
#    file.write(TimeTaken)
#    file.write('\n')
#    file.flush()
#    joblib.dump(SVM_Regular, '../Models/SVM_Regular.pkl')
    
    KernelSVM_NCAA, TimeTaken = t3.join()
    file.write('KernelSVM_NCAA Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(KernelSVM_NCAA, '../Models/KernelSVM_NCAA.pkl')
    
#    KernelSVM_Regular, TimeTaken = t4.join()
#    file.write('KernelSVM_Regular Time: ')
#    file.write(TimeTaken)
#    file.write('\n')
#    file.flush()
#    joblib.dump(KernelSVM_Regular, '../Models/KernelSVM_Regular.pkl')
    
    ##################################################################################

    t1 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/NCAA_Train.csv', 'NaiveBayes'))
#    t2 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/Regular_Train.csv', 'NaiveBayes'))
    t3 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/NCAA_Train.csv', 'RandomForest'))
#    t4 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/Regular_Train.csv', 'RandomForest'))
    
    t1.start()
#    t2.start()
    t3.start()
#    t4.start()
    
    NaiveBayes_NCAA, TimeTaken = t1.join()
    file.write('NaiveBayes_NCAA Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(NaiveBayes_NCAA, '../Models/NaiveBayes_NCAA.pkl')
    
#    NaiveBayes_Regular, TimeTaken = t2.join()
#    file.write('NaiveBayes_Regular Time: ')
#    file.write(TimeTaken)
#    file.write('\n')
#    file.flush()
#    joblib.dump(NaiveBayes_Regular, '../Models/NaiveBayes_Regular.pkl')
    
    RandomForest_NCAA, TimeTaken = t3.join()
    file.write('RandomForest_NCAA Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(RandomForest_NCAA, '../Models/RandomForest_NCAA.pkl')
    
#    RandomForest_Regular, TimeTaken = t4.join()
#    file.write('RandomForest_Regular Time: ')
#    file.write(TimeTaken)
#    file.write('\n')
#    file.flush()
#    joblib.dump(RandomForest_Regular, '../Models/RandomForest_Regular.pkl')
    
    ##################################################################################

    t1 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/NCAA_Train.csv', 'DecisionTrees'))
#    t2 = ThreadWithReturnValue(target=TrainModel, args=('../PreProcessedData/Regular_Train.csv', 'DecisionTrees'))
    
    
    t1.start()
#   t2.start()
    
    DecisionTrees_NCAA, TimeTaken = t1.join()
    file.write('DecisionTrees_NCAA Time: ')
    file.write(TimeTaken)
    file.write('\n')
    file.flush()
    joblib.dump(DecisionTrees_NCAA, '../Models/DecisionTrees_NCAA.pkl')
    
#    DecisionTrees_Regular, TimeTaken = t2.join()
#    file.write('DecisionTrees_Regular Time: ')
#    file.write(TimeTaken)
#    file.write('\n')
#    file.flush()
#    joblib.dump(DecisionTrees_Regular, '../Models/DecisionTrees_Regular.pkl')
    
    
    file.close()
    ##################################################################################
    ###########  SAVING THE MODELS #############   
    ##################################################################################

def TrainModel(filename, modelname):
    Dataset = pd.read_csv(filename)
    Dataset = Dataset.drop(Dataset.columns[0], axis = 1)
    Y = Dataset['DiffResult']
    X = Dataset.drop('DiffResult', axis = 1)
    X = StandardScaler().fit_transform(X)
    if modelname == 'LogisticRegression':
        start = time.time()
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X, Y)
        TimeTaken = str(time.time() - start)
    elif modelname == 'KNN':
        start = time.time()
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X, Y)
        TimeTaken = str(time.time() - start)
    elif modelname == 'SVM':
        start = time.time()
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X, Y)
        TimeTaken = str(time.time() - start)
    elif modelname == 'KernelSVM':
        start = time.time()
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X, Y)
        TimeTaken = str(time.time() - start)
    elif modelname == 'NaiveBayes':
        start = time.time()
        classifier = GaussianNB()
        classifier.fit(X, Y)
        TimeTaken = str(time.time() - start)
    elif modelname == 'DecisionTrees':
        start = time.time()
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X, Y)
        TimeTaken = str(time.time() - start)
    elif modelname == 'RandomForest':
        start = time.time()
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X, Y)
        TimeTaken = str(time.time() - start)
    return (classifier, TimeTaken)

def ApplyPCA():     # Only on Mixed Training and Test
    Mixed_Test = pd.read_csv('../PreProcessedData/Mixed_Test.csv')
    Mixed_Train = pd.read_csv('../PreProcessedData/Mixed_Train.csv')
    
    MixedDataset = Mixed_Train.append(Mixed_Test)
    MixedDataset = MixedDataset.drop(MixedDataset.columns[0], axis = 1)
    
    Mixed_Y = MixedDataset['DiffResult']
    Mixed_X = MixedDataset.drop('DiffResult', axis = 1)
    
    pca = PCA(.95)
    pca.fit(Mixed_X)
    pca.n_components_
    pca.explained_variance_ratio_
    MixedDataset = pd.DataFrame()
    Mixed_X = pca.transform(Mixed_X)
    MixedDataset = pd.DataFrame(Mixed_X)
    MixedDataset.columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5',
               'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10',
               'PCA11', 'PCA12', 'PCA13']
    
    Mixed_Y = np.array(Mixed_Y)
    MixedDataset['DiffResult'] = Mixed_Y
    
    return MixedDataset


def createMixedTrainAndTestDataset():
    MixedDataset = ApplyPCA()
    MixedDatasetTrain, MixedDatasetTest = train_test_split(MixedDataset, test_size = 0.25, random_state = 0)
    MixedDatasetTrain.to_csv('../PreProcessedData/ReducedMixedDataTrain.csv')
    MixedDatasetTest.to_csv('../PreProcessedData/ReducedMixedDataTest.csv')
    
main()



''' 
Models List

1. Logistic Regression
2. K-Nearest Neighbors (K-NN)
3. Support Vector Machine (SVM)
4. Kernel SVM
5. Naive Bayes
6. Decision Tree Classification
7. Random Forest Classifications

'''