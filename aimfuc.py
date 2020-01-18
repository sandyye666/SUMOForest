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


def DTLZ1(Chrom, LegV):  # M is the dimension
    M = 3
    x = Chrom.T
    XM = x[M - 1:]
    k = x.shape[0] - M + 1
    gx = 100 * (k + np.sum((XM - 0.5) ** 2 - np.cos(20 * np.pi * (XM - 0.5)), 0))

    ObjV = (np.array([[]]).T) * np.zeros((1, Chrom.shape[0]))
    ObjV = np.vstack([ObjV, 0.5 * np.cumprod(x[:M - 1], 0)[-1] * (1 + gx)])
    for i in range(2, M):
        ObjV = np.vstack([ObjV, 0.5 * np.cumprod(x[: M - i], 0)[-1] * (1 - x[M - i]) * (1 + gx)])
    ObjV = np.vstack([ObjV, 0.5 * (1 - x[0]) * (1 + gx)])
    return [ObjV.T, LegV]

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

def test(Phen, LegV):
    x1 = Phen[:, [0]]
    x2 = Phen[:, [1]]
    x3 = Phen[:, [2]]
    x4 = Phen[:, [3]]
    f = 18 * x1 + 10 * x2 + 12 * x3 + 8 * x4
    # 约束条件
    idx1 = np.where(12 * x1 + 6 * x2 + 10 * x3 + 4 * x4 > 20)[0]
    idx2 = np.where(x3 + x4 > 1)[0]
    idx3 = np.where(x3 - x1 > 0)[0]
    idx4 = np.where(x4 - x2 > 0)[0]
    # 采用惩罚方法1
    f[idx1] = -1
    f[idx2] = -1
    f[idx3] = -1
    f[idx4] = -1
    return [f, LegV]

random_state =2019
def allga(Phen, LegV, feature_data, result_data):
    print(shape(Phen[:, 0:3]))
    wR = Phen[:, [3]]
    wE = Phen[:, [4]]
    wL = Phen[:, [5]]
    class_weightR = {0: 1, 1: 1}
    class_weightE = {0: 1, 1: 1}
    class_weightL = {0: 1, 1: 1}
    X_train, X_testv, Y_train, Y_testv = train_test_split(feature_data, result_data, test_size=0.4, random_state=random_state)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_testv, Y_testv, test_size=0.5, random_state=random_state)
    Phen3 = Phen[:, 0:3]
    V_pred = np.zeros((len(Phen), len(Y_validation)))
    T_pred = np.zeros((len(Phen), len(Y_test)))
    for i in range(len(wR)):
        class_weightR[0] = wR[i]
        class_weightE[0] = wE[i]
        class_weightL[0] = wL[i]
        print(wR[i])
        print(class_weightR)
        print(class_weightE)
        print(class_weightL)
        config = {}
        ca_config = {}
        ca_config["random_state"] = 0  # 0 or 1
        ca_config["max_layers"] = 10  # 最大的层数，layer对应论文中的level
        ca_config["early_stopping_rounds"] = 3  # 如果出现某层的三层以内的准确率都没有提升，层中止
        ca_config["n_classes"] = 2  # 判别的类别数量
        ca_config["estimators"] = []
        # ca_config["estimators"].append(
        #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
        #          "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
        ca_config["estimators"].append(
            {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 70, "min_samples_leaf": 4,
             "min_samples_split": 10, "max_depth": 8, "class_weight": class_weightR, "n_jobs": -1})
        ca_config["estimators"].append(
            {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 100, "min_samples_leaf": 4,
             "min_samples_split": 10, "max_depth": 8, "class_weight": class_weightE, "n_jobs": -1})
        ca_config["estimators"].append(
            {"n_folds": 5, "type": "LogisticRegression", "class_weight": class_weightL, "penalty": "l2",
             "solver": "lbfgs"})
        config["cascade"] = ca_config  # 共使用了四个基学习器
        gc = GCForest(config)  # should be a dict
        X_train_enc = gc.fit_transform(X_train, Y_train)
        y_pred = gc.predict(X_validation)
        X_validation_enc = gc.transform(X_validation)
        X_test_enc = gc.transform(X_test)
        V_probaF = X_validation_enc[:, ::2].T
        V_probaT = X_validation_enc[:, 1::2].T

        V_predT = np.dot(Phen3[i], V_probaT)
        V_predF = np.dot(Phen3[i], V_probaF)
        for j in range(len(V_predT[0])):
            if (V_predT[i][j] >V_predF[i][j]):
                V_pred[i][j] = 1
            else:
                V_pred[i][j] = 0
        print(shape(V_probaT))
        print(shape(V_probaF))
    Phen = Phen[:, 0:3]
    print(shape(Phen))
    predT= np.dot(Phen, V_probaT)
    predF= np.dot(Phen, V_probaF)
    pred = np.zeros((len(predT), len(predT[0])))
    for i in range(len(predT)):
        for j in range(len(predT[0])):
            if (predT[i][j] >predF[i][j]):
                pred[i][j] = 1
            else:
                pred[i][j] = 0
    funclist= []
    for i in range(len(pred)):
        confmat = confusion_matrix(Y_validation, pred[i])
        # print(confmat)
        # sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        # sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        # print('1. The acc score of the model {}\n'.format(accuracy_score(result, pred[i])))
        # print('2. The sp score of the model {}\n'.format(sp))
        # print('3. The sn score of the model {}\n'.format(sn))
        # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result, pred[i])))
        # print('5. The F-1 score of the model {}\n'.format(f1_score(result, pred[i], average='macro')))
        # print('9. The auc score of the model {}\n'.format(roc_auc_score(result, pred[i], average='macro')))
        temper = f1_score(Y_validation, pred[i], average='macro')
        funclist.append(temper)
    func = array(funclist)
    refunc= func.reshape(-1, 1)
    # idx1 = np.where(w1 + w2 > 1)[0]
    # refunc[idx1] = 0
    # exIdx = np.unique(np.hstack([idx1]))
    # LegV[exIdx] = 0
    return [refunc, LegV, X_test_enc, Y_test]
