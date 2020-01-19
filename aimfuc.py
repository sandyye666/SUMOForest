import numpy as np
# ZDT1
from gcforest.gcforest import GCForest
from sklearn.metrics import confusion_matrix
from numpy import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, \
            classification_report, recall_score, precision_recall_curve,accuracy_score,matthews_corrcoef
import numpy as np

# 传入种群基因表现型矩阵Phen以及种群个体的可行性列向量LegV
from sklearn.model_selection import train_test_split


def WRF(Phen, LegV, proba, result):
    cost_matrix = [[0, 1],
                   [3, 0]]
    probaF = np.zeros((len(proba), len(proba[0])))
    for i in range(len(proba)):
        for j in range(len(proba[0])):
            if(proba[i][j] == 0):
                probaF[i][j] = 1
    predT= np.dot(Phen, proba) * cost_matrix[1][0]
    predF= np.dot(Phen, probaF) * cost_matrix[0][1]
    pred = np.zeros((len(predT), len(predT[0])))
    for i in range(len(predT)):
        for j in range(len(predT[0])):
            if (predT[i][j] >predF[i][j]):
                pred[i][j] = 1
            else:
                pred[i][j] = 0
    funclist= []
    for i in range(len(pred)):
        confmat = confusion_matrix(result, pred[i])
        print(confmat)
        sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        print('1. The acc score of the model {}\n'.format(accuracy_score(result, pred[i])))
        print('2. The sp score of the model {}\n'.format(sp))
        print('3. The sn score of the model {}\n'.format(sn))
        print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result, pred[i])))
        print('9. The auc score of the model {}\n'.format(roc_auc_score(result, pred[i], average='macro')))
        # print(confmat[0, 1])
        # print(cost_matrix[33][1])
        # print(cost_matrix[1][0])
        temper = confmat[0, 1] + confmat[1, 0]
        # print(temper)
        funclist.append(temper)
    # confmat = confusion_matrix(result,pred)
    # func = confmat[0, 1] * cost_matrix[0, 1]+confmat[1, 0] * cost_matrix[1, 0]
    func = array(funclist)
    refunc= func.reshape(-1, 1)
    return [refunc, LegV]

# 传入种群基因表现型矩阵Phen以及种群个体的可行性列向量LegV
def gcforestF12(Phen, LegV, proba, result):
    w1 = Phen[:, [0]]
    w2 = Phen[:, [1]]
    w3 = 1-w1-w2
    w = np.column_stack((Phen, w3))
    probaF = proba[:, ::2].T
    probaT = proba[:, 1::2].T
    predT= np.dot(w, probaT)
    predF= np.dot(w, probaF)
    pred = np.zeros((len(predT), len(predT[0])))
    for i in range(len(predT)):
        for j in range(len(predT[0])):
            if (predT[i][j] >predF[i][j]):
                pred[i][j] = 1
            else:
                pred[i][j] = 0
    funclist= []
    for i in range(len(pred)):
        confmat = confusion_matrix(result, pred[i])
        # print(confmat)
        # sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        # sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        # print('1. The acc score of the model {}\n'.format(accuracy_score(result, pred[i])))
        # print('2. The sp score of the model {}\n'.format(sp))
        # print('3. The sn score of the model {}\n'.format(sn))
        # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result, pred[i])))
        # print('5. The F-1 score of the model {}\n'.format(f1_score(result, pred[i], average='macro')))
        # print('9. The auc score of the model {}\n'.format(roc_auc_score(result, pred[i], average='macro')))
        temper = f1_score(result, pred[i], average='macro')
        funclist.append(temper)
    func = array(funclist)
    refunc= func.reshape(-1, 1)
    idx1 = np.where(w1 + w2 > 1)[0]
    # refunc[idx1] = 0
    exIdx = np.unique(np.hstack([idx1]))
    LegV[exIdx] = 0
    return [refunc, LegV]

# 传入种群基因表现型矩阵Phen以及种群个体的可行性列向量LegV
def gcforestCM(Phen, LegV, proba, result):
    probaF = proba[:, ::2].T
    probaT = proba[:, 1::2].T
    predT= np.dot(Phen, probaT)
    predF= np.dot(Phen, probaF)

    pred = np.zeros((len(predT), len(predT[0])))
    for i in range(len(predT)):
        for j in range(len(predT[0])):
            if (predT[i][j] >predF[i][j]):
                pred[i][j] = 1
            else:
                pred[i][j] = 0
    funclist= []
    for i in range(len(pred)):
        confmat = confusion_matrix(result, pred[i])
        cost_matrix = [[0, 1],
                       [2, 0]]
        # print(confmat)
        # sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        # sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        # print('1. The acc score of the model {}\n'.format(accuracy_score(result, pred[i])))
        # print('2. The sp score of the model {}\n'.format(sp))
        # print('3. The sn score of the model {}\n'.format(sn))
        # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result, pred[i])))
        # print('5. The F-1 score of the model {}\n'.format(f1_score(result, pred[i], average='macro')))
        # print('9. The auc score of the model {}\n'.format(roc_auc_score(result, pred[i], average='macro')))
        temper = confmat[0, 1] * cost_matrix[0][1] + confmat[1, 0] * cost_matrix[1][0]
        funclist.append(temper)
    func = array(funclist)
    refunc= func.reshape(-1, 1)
    # idx1 = np.where(w1 + w2 > 1)[0]
    # refunc[idx1] = 0
    # exIdx = np.unique(np.hstack([idx1]))
    # LegV[exIdx] = 0
    return [refunc, LegV]

# 传入种群基因表现型矩阵Phen以及种群个体的可行性列向量LegV
def gcforestF13(Phen, LegV, proba, result):
    # w1 = Phen[:, [0]]
    # w2 = Phen[:, [1]]
    # w3 = 1-w1-w2
    # w = np.column_stack((Phen, w3))
    probaF = proba[:, ::2].T
    probaT = proba[:, 1::2].T
    Phen = Phen[:, 0:3]
    predT= np.dot(Phen, probaT)
    predF= np.dot(Phen, probaF)
    pred = np.zeros((len(predT), len(predT[0])))
    for i in range(len(predT)):
        for j in range(len(predT[0])):
            if (predT[i][j] >predF[i][j]):
                pred[i][j] = 1
            else:
                pred[i][j] = 0
    funclist= []
    for i in range(len(pred)):
        confmat = confusion_matrix(result, pred[i])
        # print(confmat)
        # sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        # sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        # print('1. The acc score of the model {}\n'.format(accuracy_score(result, pred[i])))
        # print('2. The sp score of the model {}\n'.format(sp))
        # print('3. The sn score of the model {}\n'.format(sn))
        # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result, pred[i])))
        # print('5. The F-1 score of the model {}\n'.format(f1_score(result, pred[i], average='macro')))
        # print('9. The auc score of the model {}\n'.format(roc_auc_score(result, pred[i], average='macro')))
        temper = f1_score(result, pred[i], average='macro')
        funclist.append(temper)
    func = array(funclist)
    refunc= func.reshape(-1, 1)
    # idx1 = np.where(w1 + w2 > 1)[0]
    # refunc[idx1] = 0
    # exIdx = np.unique(np.hstack([idx1]))
    # LegV[exIdx] = 0
    return [refunc, LegV]
