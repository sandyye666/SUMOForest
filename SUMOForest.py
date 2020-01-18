from __future__ import division
import docx
from numpy import *
import numpy as np
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from  sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# import tensorflow as tf
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, \
            classification_report, recall_score, precision_recall_curve,accuracy_score,matthews_corrcoef
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import geatpy as ga
from gcforest.gcforest import GCForest

# from code_templet import code_templet
from new_code_templet import new_code_templet
from new_code_templet import new_code_templetX


def getSite(filename):
    doc = docx.Document(filename)
    siteList = []
    for site in doc.paragraphs:
        siteList.append(site.text)
    #print(siteList)
    return siteList

# 判断是否为天然氨基酸，用X代表其他所有氨基酸，所有氨基酸用数字来替代，构造单个频率矩阵以及n_gram频率矩阵
def replace_no_native_amino_acid(lists, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    # num_list = []
    # print(len(lists))
    # print(len(lists[0]))
    # print(native_amino_acid[0])
    numlists = zeros((len(lists), 22))
    frequency_array = zeros((21, 21))
    n_gram_frequency_array = zeros((400, 20))
    flag = 1
    for site in range(len(lists)):
        for i in range(len(lists[0])):# 位点位置
            for j in range(len(native_amino_acid)):# 氨基酸种类
                # print(site, i, j)
                if lists[site][i] == native_amino_acid[j]:
                    numlists[site][i] = j+1
                    frequency_array[j][i] = frequency_array[j][i] + 1
                    flag = 0
                    if i > 0 & i < (len(lists[0])-2):
                        a = (numlists[site][i-1]-1) * 20
                        b = numlists[site][i] - 1
                        n_gram_index = int(a + b)
                        # print(a)
                        n_gram_frequency_array[n_gram_index][i-1] = n_gram_frequency_array[n_gram_index][i-1] + 1
                    break
            if flag != 0:
                # site = site[:i] + 'X' +site[i+1:]
                numlists[site][i] = 21
                frequency_array[20][i] = frequency_array[20][i]+1
            flag = 1
            if datatype == 1:
                numlists[site][21] = 1
            # print(i)
                # print(site)
        # replaced_list.append(site)
        # print(site)
    length = len(lists)
    for i in range(len(frequency_array)):
        for j in range(len(frequency_array[0])):
            frequency_array[i][j] =frequency_array[i][j]/length
    for i in range(len(n_gram_frequency_array)):
        for j in range(len(n_gram_frequency_array[0])):
            n_gram_frequency_array[i][j] = n_gram_frequency_array[i][j]/length
    # for r in n_gram_frequency_array:
    #     print(r)
    # print(frequency_array)
    return numlists, frequency_array, n_gram_frequency_array

# 构造skip_gram的频率矩阵
def get_skip_gram_frequency_array(lists, datatype, k):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    # print(len(lists))
    numlists = zeros((len(lists), 22))
    skip_gram_frequency_array = zeros((400, 20-k))
    flag = 1
    for site in range(len(lists)):
        for i in range(len(lists[0])):  # 位点位置
            for j in range(len(native_amino_acid)):  # 氨基酸种类
                # print(site, i, j)
                if lists[site][i] == native_amino_acid[j]:
                    numlists[site][i] = j + 1
                    flag = 0
                    if i > k:
                        a = (numlists[site][i - k - 1] - 1) * 20
                        b = numlists[site][i] - 1
                        skip_gram_index = int(a + b)
                        # print(a)
                        skip_gram_frequency_array[skip_gram_index][i - k - 1] = skip_gram_frequency_array[skip_gram_index][i-k-1] + 1
                    break
            if flag != 0:
                # site = site[:i] + 'X' +site[i+1:]
                numlists[site][i] = 21
            flag = 1
            if datatype == 1:
                numlists[site][21] = 1
                # print(i)
                # print(site)
                # replaced_list.append(site)
                # print(site)
    length = len(lists)
    for i in range(len(skip_gram_frequency_array)):
        for j in range(len(skip_gram_frequency_array[0])):
            skip_gram_frequency_array[i][j] = skip_gram_frequency_array[i][j] / length
    # for r in skip_gram_frequency_array:
    #     print(r)
    # print(frequency_array)
    return skip_gram_frequency_array


# exmplelist = ['AAAAAAAAAAKAAAAAAAAAA', 'CCCCCCCCCCKCCCCCCCCCC', 'AAAAAAAAAAKAAAAAAAAAA',
#               'AAAAAAAAAAKAAAAAAAAAA', 'AAAAAAAAAAKAAAAAAAAAA']
# skip_gram_frequency_array(exmplelist, 1, 1)

# 正样本频率减去负样本频率得到整体样本频率
def result_frequency_site(positive_site, negative_site,datatype):
    result_site = zeros((len(positive_site), len(positive_site[0])))
    for i in range(len(positive_site)):
        for j in range(len(positive_site[0])):
            if datatype == "frequency":
                result_site[i][j] = positive_site[i][j] - negative_site[i][j]
            elif datatype == "entropy":
                result_site[i][j] = negative_site[i][j] - positive_site[i][j]
    # print(result_site)
    return result_site

# 把序号表示矩阵转换到频率表示或熵表示
def to_site(lists, datatype, frequency_array):
    full_frequency_array = zeros((len(lists), 22))
    # print(len(lists))
    j = 0
    for site in range(len(lists)):
        # j = j+1
        for i in range(len(lists[0])-1):#位点位置
            # print(full_frequency_array[j][0])
            position = int(lists[site][i])
            # print(position)
            # print(shape(position))
            full_frequency_array[site][i] =frequency_array[position-1][i]
        if datatype == 1:#标记正负样本
            full_frequency_array[site][21] = 1
    # for r in full_frequency_array:
    #     if datatype ==1:
    #         print(r)
    #     # print(r)
    return full_frequency_array

# 把序号表示矩阵转换到n_gram频率表示
def to_n_gram_site(lists, datatype, n_gram_frequency_array):
    full_n_gram_frequency_array = zeros((len(lists), 21))
    # print(len(lists))
    for site in range(len(lists)):
        for i in range(len(lists[0])-1):#位点位置
            if i > 0:
                position_a = int(lists[site][i-1]) - 1
                position_b = int(lists[site][i]) - 1
                position_index = int(position_a * 20 + position_b)
                full_n_gram_frequency_array[site][i-1] = n_gram_frequency_array[position_index][i-1]
        if datatype == 1:#标记正负样本
            full_n_gram_frequency_array[site][20] = 1
    # for r in full_frequency_array:
    #     if datatype ==1:
    #         print(r)
    #     # print(r)
    return full_n_gram_frequency_array

# 把序号表示矩阵转换到skip_gram频率表示
def to_skip_gram_site(lists, datatype,skip_gram_frequency_array, k):
    full_skip_gram_frequency_array = zeros((len(lists), 21-k))
    # print(len(lists))
    for site in range(len(lists)):
        for i in range(len(lists[0])-1):#位点位置
            if i > k:
                position_a = int(lists[site][i-1-k]) - 1
                position_b = int(lists[site][i]) - 1
                position_index = int(position_a * 20 + position_b)
                full_skip_gram_frequency_array[site][i-1-k] = skip_gram_frequency_array[position_index][i-1-k]
        if datatype == 1:#标记正负样本
            arraylen = len(full_skip_gram_frequency_array[0])-1
            full_skip_gram_frequency_array[site][arraylen] = 1
    # for r in full_skip_gram_frequency_array:
    #     print(r)
    return full_skip_gram_frequency_array

# 算出每一列的熵
def entropy_of_site(arrays):
    entropy_arrays = zeros((len(arrays), 2))
    for site in range(len(arrays)):
        for i in range(len(arrays[0])-1):
            if arrays[site][i] == 0:
                break
            else:
                entropy_arrays[site][0] = entropy_arrays[site][0] - arrays[site][i] * np.math.log(arrays[site][i])
        entropy_arrays[site][1] = arrays[site][len(arrays[0])-1]
    # for r in arrays:
    #     print(r)
    return entropy_arrays

# # 算出每一列的序列特异性评分sequence specificity score,传入的是序号矩阵，正负频率分数，
# def specificity_score_of_site(arrays,positive_site,negative_site, datatype):
#     specificity_score_arrays = zeros((len(arrays), 2))
#     for i in range(len(arrays)):
#         for j in range(len(arrays[0])-1):
#             if positive_site[i][j] == 0 & negative_site[i][j] == 0:
#                 break
#             else:
#                 score_p = score_p + 0.5 * positive_array[i][j] * np.math.log(positive_array[i][j]/(positive_array[i][j]+negative_array[i][j]))
#                 score_n = score_n + 0.5 * negative_array[i][j] * np.math.log(negative_array[i][j]/(positive_array[i][j]+negative_array[i][j]))
#         specificity_score_arrays[i][0] = 0 - score_p - score_n
#         specificity_score_arrays[i][1] = datatype
#     return specificity_score_arrays

# 条件频率矩阵,传入的是整体的频率矩阵
def conditional_frequency(arrays):
    conditional_frequency_array = zeros((len(arrays), len(arrays[0])))
    for site in range(len(arrays)):
        for i in range(len(arrays[0])):
            # print(len(arrays[0]))
            if arrays[site][i] ==0:
                conditional_frequency_array[site][i] = 0
            else:
                if (i < 9) & (i >= 0):
                    conditional_frequency_array[site][i] = arrays[site][i] / (arrays[site][i + 1])
                elif (i > 11):
                    conditional_frequency_array[site][i] = arrays[site][i] / (arrays[site][i - 1])
        conditional_frequency_array[site][9] = arrays[site][9]
        conditional_frequency_array[site][10] = arrays[site][10]
        conditional_frequency_array[site][11] =arrays[site][11]
    # for r in conditional_frequency_array:
    #     print(r)
    return conditional_frequency_array


# #用（条件，信息）熵来代替频率
# def to_entropy(arrays):
#     entropy_array = zeros((len(arrays), 21))
#     for site in range(len(arrays)):
#         for i in range(len(arrays[0])):  # 位点位置
#             if arrays[site][i] == 0:
#                 entropy_array[site][i] = 0
#             else:
#                 entropy_array[site][i] = 0 - arrays[site][i] * np.math.log(arrays[site][i], 2)
#         # entropy_array[site][21] = arrays[site][21]仅仅只传入了特征矩阵的分布21*21
#     # for r in entropy_array:
#     #     print(r)
#     return entropy_array

# 根据氨基酸在—1和+2的出现的氨基酸的特性构造两组特征
def hydrophobic_position_array(allarrary, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    type1 = ('I', 'L', 'V',)
    # 0.4388
    type2 = ('A', 'F', 'M', 'P', 'W',)
    # -0.031
    type3 = ('G', 'Y',)
    # -0.0644
    # other：-0.3725
    type4 = ('D', 'E',)
    # 0.6287
    # -0.6299
    position1_array = zeros((len(allarrary), 3))
    for i in range(len(allarrary)-1):

        # print(allarrary[i][9])
        if allarrary[i][9] in type1:
            position1_array[i][0] = 0.4388
        elif allarrary[i][9] in type2:
            position1_array[i][0] = -0.031
        elif allarrary[i][9] in type3:
            position1_array[i][0] = -0.064
        else:
            position1_array[i][0] = -0.3725

        # print(allarrary[i][12])
        if allarrary[i][12] in type4:
            position1_array[i][1] = 0.6287
        else:
            position1_array[i][1] = -0.6299
        position1_array[i][2] = datatype
    # for r in position1_array:
    #     print(r)
    return position1_array

# 根据氨基酸在—1和+2的出现的氨基酸的特性构造两组特征（0，1表示法）
def hydrophobic_position_array0(allarrary, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    type1 = ('I', 'L', 'V',)
    # 0.4388
    type2 = ('A', 'F', 'M', 'P', 'W',)
    # -0.031
    type3 = ('G', 'Y',)
    # -0.0644
    # other：-0.3725
    type4 = ('D', 'E',)
    # 0.6287
    # -0.6299
    position1_array = zeros((len(allarrary), 6))
    for i in range(len(allarrary)-1):

        # print(allarrary[i][9])
        if allarrary[i][9] in type1:
            for j in range(4):
                if j == 3:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        elif allarrary[i][9] in type2:
            for j in range(4):
                if j == 2:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        elif allarrary[i][9] in type3:
            for j in range(4):
                if j == 1:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        else:
            for j in range(4):
                if j == 0:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0

        # print(allarrary[i][12])
        if allarrary[i][12] in type4:
            position1_array[i][4] = 0
        else:
            position1_array[i][4] = 1
        position1_array[i][5] = datatype
    # for r in position1_array:
    #     print(r)
    return position1_array

# 特征矩阵拼接
def splice_feature_array(feature_array_x, feature_array_y):
    sum_feature_array = zeros((len(feature_array_x), len(feature_array_x[0])+len(feature_array_y[0])-1))
    for site in range(len(sum_feature_array)):
        for i in range(len(feature_array_x[0])-1):
            sum_feature_array[site][i] = feature_array_x[site][i]
        for i in range(len(feature_array_x[0])-1, len(sum_feature_array[0])):
            sum_feature_array[site][i] = feature_array_y[site][i+1-len(feature_array_x[0])]
    return sum_feature_array

# 保证生成的随机状态一致
random_state = 2018
np.random.seed(random_state)
#获取平均数
def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)

def Stacking(feature_data, result_data):
    feature_train, feature_test, result_train, result_test = train_test_split(feature_data, result_data, test_size=0.1, random_state=random_state)
    '''模型融合中使用到的各个单模型'''
    class_weight = dict({0: 1.5, 1: 20.5})
    clfs = [
            RandomForestClassifier(bootstrap=True, class_weight=class_weight, criterion='entropy', max_depth=15,
                                   max_features=40, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                   min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=174, n_jobs=-1, oob_score=False,
                                   random_state=random_state, verbose=0, warm_start=False),
            # tree.DecisionTreeClassifier(random_state=random_state, max_depth=4, min_samples_leaf=6, min_samples_split=18,
            #                             max_features=85, criterion='gini'),
            # KNeighborsClassifier(n_neighbors=50),
            # GaussianNB(),
            # MultinomialNB(), # 必须要求样本值为负，不符合
            # SVC(gamma='auto', C=0.001, kernel="linear",probability=True),
            # LogisticRegression(random_state=random_state, C=0.9, solver='newton-cg', class_weight=class_weight),
            # GradientBoostingClassifier(random_state=random_state, learning_rate=0.2, n_estimators=261,
            #                            max_depth=10, min_samples_split=2, min_samples_leaf=95,
            #                            max_features=21, subsample=0.75)
    ]

    '''切分一部分数据作为测试集'''
    X, X_predict, y, y_predict = train_test_split(feature_data, result_data, test_size=0.2, random_state=random_state)
    skfall = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state).split(feature_data, result_data))
    acc = np.zeros(5).reshape(5,-1)
    sp = np.zeros(5).reshape(5,-1)
    sn = np.zeros(5).reshape(5,-1)
    mcc = np.zeros(5).reshape(5,-1)
    auc = np.zeros(5).reshape(5,-1)
    for k, (train, predict) in enumerate(skfall):
        X, y, X_predict, y_predict = feature_data[train], result_data[train], feature_data[predict], result_data[predict]
        dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
        dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))
        '''5折stacking'''
        n_splits = 5
        skf = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X, y))
        for j, clf in enumerate(clfs):
            '''依次训练各个单模型'''
            # print(j, clf)
            dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
            for i, (train, test) in enumerate(skf):
                '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
                # print("Fold", i)
                X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
                clf.fit(X_train, y_train)
                y_submission = clf.predict_proba(X_test)[:, 1]
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
            '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
            dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
            print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
        class_weightLR = dict({0: 0.8, 1: 1.8})
        # class_weightLR= dict({0: 0.3, 1: 1.8})
        # clf = LogisticRegression(random_state=random_state, C=0.9, solver='newton-cg', class_weight=class_weight)
        clf = LogisticRegression(class_weight=class_weightLR)
        # clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
        clf.fit(dataset_blend_train, y)
        y_submission = clf.predict(dataset_blend_test)

        # print("Linear stretch of predictions to [0,1]")
        # y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
        print("blend result")
        print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))
        confmat = confusion_matrix(y_predict, y_submission)
        acc[k] = accuracy_score(y_predict, y_submission)
        sn[k] = (confmat[1, 1] / (confmat[1, 0] + confmat[1, 1]))
        sp[k] = (confmat[0, 0] / (confmat[0, 0] + confmat[0, 1]))
        mcc[k] = matthews_corrcoef(y_predict, y_submission)
        auc[k] = roc_auc_score(y_predict, y_submission, average='macro')
        # print('1. The acc score of the model {}\n'.format(accuracy_score(y_predict, y_submission)))
        # print('2. The sp score of the model {}\n'.format(sp[k]))
        # print('3. The sn score of the model {}\n'.format(sn[k]))
        # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(y_predict, y_submission)))
        # print('9. The auc score of the model {}\n'.format(roc_auc_score(y_predict, y_submission, average='macro')))
        # print('6. The recall score of the model {}\n'.format(recall_score(y_predict, y_submission, average='macro')))
        # print('5. The F-1 score of the model {}\n'.format(f1_score(y_predict, y_submission, average='macro')))
        # print('7. Classification report \n {} \n'.format(classification_report(y_predict, y_submission)))
        # print('8. Confusion matrix \n {} \n'.format(confusion_matrix(y_predict, y_submission)))

        # class_weight = dict({0: 20.5, 1: 1.5})
        def re_predict(data, threshods):
            argmax = np.argmax(data)
            if argmax == 1:
                return argmax
            else:
                if data[argmax] >= threshods[argmax]:
                    return argmax
                else:
                    return (argmax + 1)

        y_submission_proba = clf.predict_proba(dataset_blend_test)

        threshold = [0.92, 0.30]
        new_pred = []
        print(y_submission.shape[0])
        for i in range(y_submission.shape[0]):
            new_pred.append(re_predict(y_submission_proba[i, :], threshold))
        print(k)
        confmat = confusion_matrix(y_predict, new_pred)
        # sp = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        # # sn = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        acc[k] = accuracy_score(y_predict, new_pred)
        sn[k] = (confmat[1, 1] / (confmat[1, 0] + confmat[1, 1]))
        sp[k] = (confmat[0, 0] / (confmat[0, 0] + confmat[0, 1]))
        mcc[k] = matthews_corrcoef(y_predict, new_pred)
        auc[k] = roc_auc_score(y_predict, new_pred, average='macro')

        # print('1. The acc score of the model {}\n'.format(accuracy_score(y_predict, new_pred)))
        # print('2. The sp score of the model {}\n'.format(sn[k]))
        # print('3. The sn score of the model {}\n'.format(sp[k]))
        # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(y_predict, new_pred)))
        # print('4. The auc score of the model {}\n'.format(roc_auc_score(y_predict, new_pred, average='macro')))
        # print('5. The F-1 score of the model {}\n'.format(f1_score(y_predict, new_pred, average='macro')))
        # print('6. The recall score of the model {}\n'.format(recall_score(y_predict, new_pred, average='macro')))
        # print('7. Classification report \n {} \n'.format(classification_report(y_predict, new_pred)))
        # print('8. Confusion matrix \n {} \n'.format(confusion_matrix(y_predict, new_pred)))
    aveacc = acc.mean()
    avesn = sn.mean()
    avesp = sp.mean()
    avemcc = mcc.mean()
    aveauc = auc.mean()
    print("--------------------------------------------------------------------------")
    print('1. The acc score of the model {}\n'.format(aveacc))
    print('2. The sp score of the model {}\n'.format(avesp))
    print('3. The sn score of the model {}\n'.format(avesn))
    print('4. The mcc score of the model {}\n'.format(avemcc))
    print('5. The auc score of the model {}\n'.format(aveauc))
    # print('5. The F-1 score of the model {}\n'.format(f1_score(y_predict, new_pred, average='macro')))
    # print('6. The recall score of the model {}\n'.format(recall_score(y_predict, new_pred, average='macro')))
    # print('7. Classification report \n {} \n'.format(classification_report(y_predict, new_pred)))
    # print('8. Confusion matrix \n {} \n'.format(confusion_matrix(y_predict, new_pred)))

def RandomForest_prediction(feature_data, result_data):
    kf = StratifiedKFold(n_splits=5) # 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同，需要目标数据
    all_pred = np.zeros(feature_data.shape[0])
    all_proba = np.zeros(feature_data.shape[0])
    random_state = 2019
    for train_index, test_index in kf.split(feature_data, result_data):
        feature_train, feature_test, result_train, result_test= \
            feature_data[train_index], feature_data[test_index], result_data[train_index], result_data[test_index]
        # clf = RandomForestClassifier()
        # clf =RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=8,
        #                        max_features=None, max_leaf_nodes=4, min_impurity_decrease=0.0,
        #                        min_impurity_split=None, min_samples_leaf=1, min_samples_split=10,
        #                        min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=-1, oob_score=False,
        #                        random_state=random_state, verbose=0, warm_start=False)
        class_weight = {0: 1, 1: 10}
        class_weight[1] = 20
        print(class_weight)
        clf= RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=15,
                               max_features=40, max_leaf_nodes=None, min_impurity_decrease=0.0,
                               min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=174, n_jobs=-1, oob_score=False,
                               random_state=random_state, verbose=0, warm_start=False, class_weight=class_weight)
        #  "n_estimators": 70, "min_samples_leaf": 4, "min_samples_split": 10, "max_depth": 8, "class_weight": "balanced", "n_jobs": -1
        # result_train.ravel()
        clf.fit(feature_train, result_train.ravel())
        # print("系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大")
        # print(clf.feature_importances_)
        # feature_importances = np.zeros((4))
        # featurearray = clf.feature_importances_
        #
        # for i in range(len(featurearray)):
        #     if (i < 21):
        #         feature_importances[0] += featurearray[i]
        #         # 频率权重矩阵 21
        #     elif (i < 42):
        #         feature_importances[1] += featurearray[i]
        #         # 条件频率权重矩阵 21
        #     elif (i < 44):
        #         feature_importances[2] += featurearray[i]
        #         # 位置特性 2
        #     else:
        #         feature_importances[3] += featurearray[i]
        #         # bigram 74
        #
        # for r in feature_importances:
        #     print(r)
        test_pred = clf.predict(feature_test)
        test_proba = clf.predict_proba(feature_test)
        all_pred[test_index] = test_pred
        all_proba[test_index] = test_proba[:, 1]
    confmat = confusion_matrix(result_data, all_pred)
    sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
    sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
    print('1. The acc score of the model {}\n'.format(accuracy_score(result_data, all_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_data, all_pred)))
    print('9. The auc score of the model {}\n'.format(roc_auc_score(result_data, all_proba, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(result_data, all_pred, average='macro')))

def ExtraTree_prediction(feature_data, result_data):
    n_splits=5
    kf = StratifiedKFold(n_splits=n_splits) # 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同，需要目标数据
    all_pred = np.zeros(feature_data.shape[0])
    all_proba = np.zeros(feature_data.shape[0])
    for train_index, test_index in kf.split(feature_data, result_data):
        feature_train, feature_test, result_train, result_test= \
            feature_data[train_index], feature_data[test_index], result_data[train_index], result_data[test_index]
        class_weight = {0: 1, 1: 1}
        clf = ExtraTreesClassifier(random_state=random_state, class_weight=class_weight)
        clf.fit(feature_train, result_train.ravel())
        test_pred = clf.predict(feature_test)
        test_proba = clf.predict_proba(feature_test)
        all_pred[test_index] = test_pred
        all_proba[test_index] = test_proba[:, 1]
    confmat = confusion_matrix(result_data, all_pred)
    sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
    sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
    print('1. The acc score of the model {}\n'.format(accuracy_score(result_data, all_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_data, all_pred)))
    print('9. The auc score of the model {}\n'.format(roc_auc_score(result_data, all_proba, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(result_data, all_pred, average='macro')))

def LogisticRegression_prediction(feature_data, result_data):
    n_splits=5
    kf = StratifiedKFold(n_splits=n_splits) # 分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同，需要目标数据
    all_pred = np.zeros(feature_data.shape[0])
    all_proba = np.zeros(feature_data.shape[0])
    for train_index, test_index in kf.split(feature_data, result_data):
        feature_train, feature_test, result_train, result_test= \
            feature_data[train_index], feature_data[test_index], result_data[train_index], result_data[test_index]
        class_weight = {0: 1, 1: 13}
        clf = LogisticRegression(class_weight=class_weight)
        clf.fit(feature_train, result_train.ravel())
        test_pred = clf.predict(feature_test)
        test_proba = clf.predict_proba(feature_test)
        all_pred[test_index] = test_pred
        all_proba[test_index] = test_proba[:, 1]
    confmat = confusion_matrix(result_data, all_pred)
    sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
    sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
    print('1. The acc score of the model {}\n'.format(accuracy_score(result_data, all_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_data, all_pred)))
    print('9. The auc score of the model {}\n'.format(roc_auc_score(result_data, all_proba, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(result_data, all_pred, average='macro')))
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0  # 0 or 1
    ca_config["max_layers"] = 10  #最大的层数，layer对应论文中的level
    ca_config["early_stopping_rounds"] = 3  #如果出现某层的三层以内的准确率都没有提升，层中止
    ca_config["n_classes"] = 2      #判别的类别数量
    ca_config["estimators"] = []
    # ca_config["estimators"].append(
    #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
    #          "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 70, "min_samples_leaf": 4, "min_samples_split": 10, "max_depth": 8, "class_weight": "balanced", "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 100, "min_samples_leaf": 4, "min_samples_split": 10, "max_depth": 8, "class_weight": dict({0: 13, 1: 1}), "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression", "class_weight": "balanced", "penalty" : "l2", "solver": "lbfgs"})
    config["cascade"] = ca_config    #共使用了四个基学习器
    return config
def get_toy_config0(r, e, l):
    config = {}
    ca_config = {}
    class_weightR={0:1, 1:1}
    class_weightE={0:1, 1:1}
    class_weightL={0:1, 1:1}
    class_weightR[0] = r
    class_weightE[0] = e
    class_weightL[0] = l
    ca_config["random_state"] = 0  # 0 or 1
    ca_config["max_layers"] = 10  #最大的层数，layer对应论文中的level
    ca_config["early_stopping_rounds"] = 3  #如果出现某层的三层以内的准确率都没有提升，层中止
    ca_config["n_classes"] = 2      #判别的类别数量
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 70, "min_samples_leaf": 4, "min_samples_split": 10, "max_depth": 8, "class_weight": class_weightR, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 100, "min_samples_leaf": 4, "min_samples_split": 10, "max_depth": 8, "class_weight": class_weightE, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression", "class_weight": class_weightL, "penalty" : "l2", "solver": "lbfgs"})
    config["cascade"] = ca_config    #共使用了四个基学习器
    return config
def get_toy_config1():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0  # 0 or 1
    ca_config["max_layers"] = 10  #最大的层数，layer对应论文中的level
    ca_config["early_stopping_rounds"] = 3  #如果出现某层的三层以内的准确率都没有提升，层中止
    ca_config["n_classes"] = 2      #判别的类别数量
    ca_config["estimators"] = []
    # ca_config["estimators"].append(
    #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
    #          "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config    #共使用了四个基学习器
    return config
# get_toy_config()生成的结构，如下所示：

'''
{
"cascade": {
    "random_state": 0,
    "max_layers": 100,000000000000000000000000000000
    "early_stopping_rounds": 3,
    "n_classes": 2,
    "estimators": [
        {"n_folds":5,"type":"ExtraTreesClassifier","n_estimators":10,
		"max_depth":null,"n_jobs":-1},
       {"n_folds":5,"type":"RandomForestClassifier","n_estimators":10,
		"max_depth":null,"n_jobs":-1},
        {"n_folds":5,"type":"XGBClassifier","n_estimators":10,"max_depth":5,
		"objective":"multi:softprob", "silent":true,
		"nthread":-1, "learning_rate":0.1},
        { "n_folds": 5, "type": "LogisticRegression"}
    ]
}
}
'''
        # {"n_folds":5,"type":"RandomForestClassifier","n_estimators":10,
		# "max_depth":null,"n_jobs":-1},
     # {"n_folds":5,"type":"XGBClassifier","n_estimators":10,"max_depth":5,
		# "objective":"multi:softprob", "silent":true,
		# "nthread":-1, "learning_rate":0.1},
# {"n_folds": 5, "type": "LogisticRegression"}
def GAGCForest_prediction0(feature_data, result_data):
    # 获取函数接口地址
    AIM_M = __import__('aimfuc')
    AIM_F = 'allga'
    """============================变量设置============================"""
    wR = [0.01, 1]
    wE = [0.01, 1]
    wL = [0.01, 1]
    bR = [1, 1]
    bE = [1, 1]
    bL = [1, 1]
    w1 = [0, 1]
    w2 = [0, 1]
    w3 = [0, 1]
    b1 = [1, 1]
    b2 = [1, 1]
    b3 = [1, 1]
    ranges = np.vstack([w1, w2, w3, wR, wE, wL]).T  # 生成自变量的范围矩阵
    borders = np.vstack([b1, b2, b3, bR, bE, bL]).T  # 生成自变量的边界矩阵
    # ranges = np.vstack([np.zeros((1, 3)), np.ones((1, 3))])  # 生成自变量的范围矩阵
    # print(shape(ranges))
    # borders = np.vstack([.ones((1, 3)), np.ones((1, 3))])  # 生成自变量的边界矩阵
    precisions = [6] * 6  # 自变量的编码精度
    scales = [0] * 6
    codes = [1] * 6
    # print(np.ones((1, 300)))
    # scales = list(np.zeros((1, 300)))  # 采用算术刻度
    # codes = np.vstack([np.ones((1, 300)), np.ones((1, 300))])  # 变量的编码方式，2个变量均使用格雷编码
    # print(shape(codes))
    """========================遗传算法参数设置========================="""
    # NIND = 50  # 种群规模
    # MAXGEN = 100  # 最大遗传代数
    # GGAP = 0.8  # 代沟：子代与父代个体不相同的概率为0.8
    # selectStyle = 'sus';  # 遗传算法的选择方式设为"sus"——随机抽样选择
    # recombinStyle = 'xovdp'  # 遗传算法的重组方式，设为两点交叉
    # recopt = 0.9  # 交叉概率
    # pm = 0.1  # 变异概率
    # SUBPOP = 1  # 设置种群数为1
    # maxormin = 1  #
    # 设置最大最小化目标标记为1，表示是最小化目标，-1则表示最大化目标

    FieldD = ga.crtfld(ranges, borders, precisions, codes, scales)  #

    # 调用编程模板
    [weightarray, pop_trace, var_trace, times, X_test_enc, Y_test] = new_code_templetX(AIM_M, AIM_F, None, None, FieldD, problem='R',
                                                             maxormin=-1,
                                                             MAXGEN=10, NIND=50, SUBPOP=1, GGAP=0.8,
                                                             selectStyle='sus',
                                                             recombinStyle='xovsp', recopt=0.9, pm=0.7,
                                                             distribute=True,
                                                             feature_data=feature_data, result_data=result_data,
                                                             drawing=0)
    print('用时：', times, '秒')
    # w3 = 1 - weight[0] - weight[1]
    # print(weight)

    # weightarray = np.concatenate((weight, [w3]), axis=0)
    weightarray = weightarray[:3]
    for element in weightarray:
        print(element)
    test_probaF = X_test_enc[:, ::2].T
    test_probaT = X_test_enc[:, 1::2].T
    test_predT = np.dot(weightarray, test_probaT)
    test_predF = np.dot(weightarray, test_probaF)
    test_pred = np.zeros(len(test_predT))
    test_proba = np.zeros(len(test_predT))
    for i in range(len(test_predT)):
        temper = test_predT[i] + test_predF[i]
        test_proba = test_predT/temper
        if (test_predT[i] > test_predF[i]):
            test_pred[i] = 1
        else:
            test_pred[i] = 0
    confmat = confusion_matrix(Y_test, test_pred)
    sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
    sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
    print('1. The acc score of the model {}\n'.format(accuracy_score(Y_test, test_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(Y_test, test_pred)))

    print('9. The auc score of the model {}\n'.format(roc_auc_score(Y_test, test_proba, average='macro')))
    print('6. The recall score of the model {}\n'.format(recall_score(Y_test, test_pred, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(Y_test, test_pred, average='macro')))
    print('7. Classification report \n {} \n'.format(classification_report(Y_test, test_pred)))
    print('8. Confusion matrix \n {} \n'.format(confusion_matrix(Y_test, test_pred)))

def GAGCForest_prediction(feature_data, result_data):
    n_splits = 5
    acc_scores = np.zeros(n_splits)
    recall_scores = np.zeros(n_splits)
    mcc_scores = np.zeros(n_splits)
    f1_scores = np.zeros(n_splits)
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(feature_data,
                                                                                                result_data)
    new_test_pred = np.zeros(feature_data.shape[0])
    new_test_proba = np.zeros(feature_data.shape[0])
    for j, (train_idx, test_idx) in enumerate(skfolds):
        X_train = feature_data[train_idx]
        Y_train = result_data[train_idx]
        X_test = feature_data[test_idx]
        Y_test = result_data[test_idx]
        config = get_toy_config()
        gc = GCForest(config)  # should be a dict
        X_train_enc = gc.fit_transform(X_train, Y_train)
        y_pred = gc.predict(X_test)
        X_test_enc = gc.transform(X_test)
        # 获取函数接口地址
        AIM_M = __import__('aimfuc')
        AIM_F = 'gcforestF13'
        """============================变量设置============================"""
        w1 = [0, 1]
        w2 = [0, 1]
        w3 = [0, 1]
        b1 = [1, 1]
        b2 = [1, 1]
        b3 = [1, 1]
        ranges = np.vstack([w1, w2, w3]).T  # 生成自变量的范围矩阵
        borders = np.vstack([b1, b2, b3]).T  # 生成自变量的边界矩阵
        # ranges = np.vstack([np.zeros((1, 3)), np.ones((1, 3))])  # 生成自变量的范围矩阵
        # print(shape(ranges))
        # borders = np.vstack([.ones((1, 3)), np.ones((1, 3))])  # 生成自变量的边界矩阵
        precisions = [6] * 3  # 自变量的编码精度
        scales = [0] * 3
        codes = [1] * 3
        # print(np.ones((1, 300)))
        # scales = list(np.zeros((1, 300)))  # 采用算术刻度
        # codes = np.vstack([np.ones((1, 300)), np.ones((1, 300))])  # 变量的编码方式，2个变量均使用格雷编码
        # print(shape(codes))
        """========================遗传算法参数设置========================="""
        # NIND = 50  # 种群规模
        # MAXGEN = 100  # 最大遗传代数
        # GGAP = 0.8  # 代沟：子代与父代个体不相同的概率为0.8
        # selectStyle = 'sus';  # 遗传算法的选择方式设为"sus"——随机抽样选择
        # recombinStyle = 'xovdp'  # 遗传算法的重组方式，设为两点交叉
        # recopt = 0.9  # 交叉概率
        # pm = 0.1  # 变异概率
        # SUBPOP = 1  # 设置种群数为1
        # maxormin = 1  #
        # 设置最大最小化目标标记为1，表示是最小化目标，-1则表示最大化目标

        FieldD = ga.crtfld(ranges, borders, precisions, codes, scales)  #

        # 调用编程模板
        [weightarray, pop_trace, var_trace, times] = new_code_templet(AIM_M, AIM_F, None, None, FieldD, problem='R',
                                                                 maxormin=-1,
                                                                 MAXGEN=10, NIND=50, SUBPOP=1, GGAP=0.8,
                                                                 selectStyle='sus',
                                                                 recombinStyle='xovsp', recopt=0.9, pm=0.7,
                                                                 distribute=True,
                                                                 proba=X_train_enc, result=Y_train,
                                                                 drawing=0)
        print('用时：', times, '秒')
        # w3 = 1 - weight[0] - weight[1]
        # print(weight)

        # weightarray = np.concatenate((weight, [w3]), axis=0)
        for element in weightarray:
            print(element)
        test_probaF = X_test_enc[:, ::2].T
        test_probaT = X_test_enc[:, 1::2].T
        test_predT = np.dot(weightarray, test_probaT)
        test_predF = np.dot(weightarray, test_probaF)
        test_pred = np.zeros(len(test_predT))
        test_proba = np.zeros(len(test_predT))
        for i in range(len(test_predT)):
            temper = test_predT[i] + test_predF[i]
            test_proba = test_predT/temper
            if (test_predT[i] > test_predF[i]):
                test_pred[i] = 1
            else:
                test_pred[i] = 0
        confmat = confusion_matrix(Y_test, test_pred)
        sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        print('1. The acc score of the model {}\n'.format(accuracy_score(Y_test, test_pred)))
        print('2. The sp score of the model {}\n'.format(sp))
        print('3. The sn score of the model {}\n'.format(sn))
        print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(Y_test, test_pred)))

        print('9. The auc score of the model {}\n'.format(roc_auc_score(Y_test, test_proba, average='macro')))
        print('6. The recall score of the model {}\n'.format(recall_score(Y_test, test_pred, average='macro')))
        print('5. The F-1 score of the model {}\n'.format(f1_score(Y_test, test_pred, average='macro')))
        print('7. Classification report \n {} \n'.format(classification_report(Y_test, test_pred)))
        print('8. Confusion matrix \n {} \n'.format(confusion_matrix(Y_test, test_pred)))

        recall = recall_score(Y_test, test_pred, average='macro')
        f1 = f1_score(Y_test, test_pred, average='macro')
        acc = accuracy_score(Y_test, test_pred)
        mcc = matthews_corrcoef(Y_test, test_pred)
        recall_scores[j] = recall
        f1_scores[j] = f1
        acc_scores[j] = acc
        mcc_scores[j] = mcc
        new_test_pred[test_idx] = test_pred
        new_test_proba[test_idx] = test_proba
        print(
            "CV- {} recall: {}, acc_score: {} , mcc_score: {}, f1_score: {}".format(j, recall, acc, mcc, f1))
    new_confmat = confusion_matrix(result_data, new_test_pred)
    sn = new_confmat[1, 1] / (new_confmat[1, 0] + new_confmat[1, 1])
    sp = new_confmat[0, 0] / (new_confmat[0, 0] + new_confmat[0, 1])
    print("---------------------------------遗传算法-----------------------------------------")
    print('1. The acc score of the model {}\n'.format(accuracy_score(result_data, new_test_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_data, new_test_pred)))
    print('9. The auc score of the model {}\n'.format(roc_auc_score(result_data, new_test_proba, average='macro')))
    print('6. The recall score of the model {}\n'.format(recall_score(result_data, new_test_pred, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(result_data, new_test_pred, average='macro')))
    print('7. Classification report \n {} \n'.format(classification_report(result_data, new_test_pred)))
    print('8. Confusion matrix \n {} \n'.format(confusion_matrix(result_data, new_test_pred)))

def GCForest_prediction(feature_data, result_data):
    random_state=2019
    n_splits = 5
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(feature_data, result_data)
    test_pred = np.zeros(feature_data.shape[0])
    test_proba = np.zeros(feature_data.shape[0])
    acc_scores = np.zeros(n_splits)
    recall_scores = np.zeros(n_splits)
    mcc_scores = np.zeros(n_splits)
    f1_scores = np.zeros(n_splits)
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = feature_data[train_idx]
        Y_train = result_data[train_idx]
        X_test = feature_data[test_idx]
        Y_test = result_data[test_idx]
        config = get_toy_config()
        gc = GCForest(config)  # should be a dict
        X_train_enc = gc.fit_transform(X_train, Y_train)
        part_X_train_enc = X_train_enc[:, ::2]
        y_pred = gc.predict(X_test)
        X_test_enc = gc.transform(X_test)
        part_X_test_enc = X_test_enc[:, ::2]
        y_proba = gc.predict_proba(X_test)[:, 1]
        acc = accuracy_score(Y_test, y_pred)
        print("Test Accuracy of GcForest (save and load) = {:.2f} %".format(acc * 100))
        confmat = confusion_matrix(Y_test, y_pred)
        sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
        sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
        print('1. The acc score of the model {}\n'.format(accuracy_score(Y_test, y_pred)))
        print('2. The sp score of the model {}\n'.format(sp))
        print('3. The sn score of the model {}\n'.format(sn))
        print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(Y_test, y_pred)))
        print('9. The auc score of the model {}\n'.format(roc_auc_score(Y_test, y_proba, average='macro')))
        print('6. The recall score of the model {}\n'.format(recall_score(Y_test, y_pred, average='macro')))
        print('5. The F-1 score of the model {}\n'.format(f1_score(Y_test, y_pred, average='macro')))
        print('7. Classification report \n {} \n'.format(classification_report(Y_test, y_pred)))
        print('8. Confusion matrix \n {} \n'.format(confusion_matrix(Y_test, y_pred)))

        recall = recall_score(Y_test, y_pred, average='macro')
        f1 = f1_score(Y_test, y_pred, average='macro')
        acc = accuracy_score(Y_test, y_pred)
        mcc = matthews_corrcoef(Y_test, y_pred)

        recall_scores[j] = recall
        f1_scores[j] = f1
        acc_scores[j] = acc
        mcc_scores[j] = mcc

        test_pred[test_idx] = y_pred
        test_proba[test_idx] = y_proba
        print(
            "CV- {} recall: {}, acc_score: {} , mcc_score: {}, f1_score: {}".format(j, recall, acc, mcc, f1))
    confmat = confusion_matrix(result_data, test_pred)
    sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
    sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
    print("--------------------------------------深度森林------------------------------------")
    print('1. The acc score of the model {}\n'.format(accuracy_score(result_data, test_pred)))
    print('2. The sp score of the model {}\n'.format(sp))
    print('3. The sn score of the model {}\n'.format(sn))
    print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_data, test_pred)))
    print('9. The auc score of the model {}\n'.format(roc_auc_score(result_data, test_proba, average='macro')))
    print('6. The recall score of the model {}\n'.format(recall_score(result_data, test_pred, average='macro')))
    print('5. The F-1 score of the model {}\n'.format(f1_score(result_data, test_pred, average='macro')))
    print('7. Classification report \n {} \n'.format(classification_report(result_data, test_pred)))
    print('8. Confusion matrix \n {} \n'.format(confusion_matrix(result_data, test_pred)))



    # # class_weightLR = dict({0: 1, 1: 5})
    # # clf = LogisticRegression(C=1.0, class_weight=class_weightLR, dual=False, fit_intercept=True,
    # #       intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
    # #       penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
    # #       verbose=0, warm_start=False)
    # # clf = LogisticRegression()
    # clf = GaussianNB()
    # # clf = Perceptron()
    # # clf = QuadraticDiscriminantAnalysis()
    # clf.fit(part_X_train_enc, result_train)
    # # new_y_proba = clf.predict_proba(part_X_test_enc)[:, 1]
    # new_y_pred = clf.predict(part_X_test_enc)
    # confmat = confusion_matrix(result_test, new_y_pred)
    # sn = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])
    # sp = confmat[0, 0] / (confmat[0, 0] + confmat[0, 1])
    # print('1. The acc score of the model {}\n'.format(accuracy_score(result_test, new_y_pred)))
    # print('2. The sp score of the model {}\n'.format(sp))
    # print('3. The sn score of the model {}\n'.format(sn))
    # print('4. The mcc score of the model {}\n'.format(matthews_corrcoef(result_test, new_y_pred)))
    # # print('9. The auc score of the model {}\n'.format(roc_auc_score(result_test, new_y_proba, average='macro')))
    # print('6. The recall score of the model {}\n'.format(recall_score(result_test, new_y_pred, average='macro')))
    # print('5. The F-1 score of the model {}\n'.format(f1_score(result_test, new_y_pred, average='macro')))
    # print('7. Classification report \n {} \n'.format(classification_report(result_test, new_y_pred)))
    # print('8. Confusion matrix \n {} \n'.format(confusion_matrix(result_test, new_y_pred)))


if __name__ == '__main__':
    psiteList = getSite('PositiveData.docx')
    nsiteList = getSite('NegativeData.docx')
    positive = 1
    negative = 0
    positive_array, positive_frequency_site, positive_n_gram_frequency_site = replace_no_native_amino_acid(psiteList, positive)
    negative_array, negative_frequency_site, negative_n_gram_frequency_site = replace_no_native_amino_acid(nsiteList, negative)

    # print(positive_frequency_site)
    # print(negative_frequency_site)
    # print('wao')

    # 只利用序列编号进行预测
    allarray = np.concatenate((positive_array, negative_array), axis=0)
    # print(allarray)
    # # print(shape(allarray))
    # x, y = np.split(allarray, (21,), axis=1)

    # 频率矩阵
    sum_frequency_site = result_frequency_site(positive_frequency_site, negative_frequency_site, "frequency")
    positive_frequency_array = to_site(positive_array, positive, sum_frequency_site)
    negative_frequency_array = to_site(negative_array, negative, sum_frequency_site)
    frequency_allarray = np.concatenate((positive_frequency_array, negative_frequency_array), axis=0)
    print("frequency_allarray:"+str((len(frequency_allarray[0])-1)))
    # x, y = np.split(frequency_allarray, (21,), axis=1)

    # n_gram频率矩阵
    sum_n_gram_frequency_site = result_frequency_site(positive_n_gram_frequency_site, negative_n_gram_frequency_site, "frequency")
    positive_n_gram_frequency_array = to_n_gram_site(positive_array, positive, sum_n_gram_frequency_site)
    negative_n_gram_frequency_array = to_n_gram_site(negative_array, negative, sum_n_gram_frequency_site)
    n_gram_frequency_allarray = np.concatenate((positive_n_gram_frequency_array, negative_n_gram_frequency_array), axis=0)
    # x, y =np.split(n_gram_frequency_allarray, (20,), axis=1)

    # skip_gram频率矩阵 1，2，3
    positive_skip_gram_frequency_site1 = get_skip_gram_frequency_array(psiteList, positive, 1)
    negative_skip_gram_frequency_site1 = get_skip_gram_frequency_array(nsiteList, negative, 1)
    sum_skip_gram_frequency_site1 = result_frequency_site(positive_skip_gram_frequency_site1,
                                                         negative_skip_gram_frequency_site1, "frequency")
    positive_skip_gram_frequency_array1 = to_skip_gram_site(positive_array, positive, sum_skip_gram_frequency_site1, 1)
    negative_skip_gram_frequency_array1 = to_skip_gram_site(negative_array, negative, sum_skip_gram_frequency_site1, 1)
    skip_gram_frequency_allarray1 = np.concatenate(
        (positive_skip_gram_frequency_array1, negative_skip_gram_frequency_array1), axis=0)

    positive_skip_gram_frequency_site2 = get_skip_gram_frequency_array(psiteList, positive, 2)
    negative_skip_gram_frequency_site2 = get_skip_gram_frequency_array(nsiteList, negative, 2)
    sum_skip_gram_frequency_site2 = result_frequency_site(positive_skip_gram_frequency_site2,
                                                         negative_skip_gram_frequency_site2, "frequency")
    positive_skip_gram_frequency_array2 = to_skip_gram_site(positive_array, positive, sum_skip_gram_frequency_site2, 2)
    negative_skip_gram_frequency_array2 = to_skip_gram_site(negative_array, negative, sum_skip_gram_frequency_site2, 2)
    skip_gram_frequency_allarray2 = np.concatenate(
        (positive_skip_gram_frequency_array2, negative_skip_gram_frequency_array2), axis=0)

    positive_skip_gram_frequency_site3 = get_skip_gram_frequency_array(psiteList, positive, 3)
    negative_skip_gram_frequency_site3 = get_skip_gram_frequency_array(nsiteList, negative, 3)
    sum_skip_gram_frequency_site3 = result_frequency_site(positive_skip_gram_frequency_site3,
                                                         negative_skip_gram_frequency_site3, "frequency")
    positive_skip_gram_frequency_array3 = to_skip_gram_site(positive_array, positive, sum_skip_gram_frequency_site3, 3)
    negative_skip_gram_frequency_array3 = to_skip_gram_site(negative_array, negative, sum_skip_gram_frequency_site3, 3)
    skip_gram_frequency_allarray3 = np.concatenate(
        (positive_skip_gram_frequency_array3, negative_skip_gram_frequency_array3), axis=0)

    min_skip_gram_frequency_allarray=splice_feature_array(skip_gram_frequency_allarray1,skip_gram_frequency_allarray2)
    skip_gram_frequency_allarray=splice_feature_array(min_skip_gram_frequency_allarray,skip_gram_frequency_allarray3)
    bigram_feature_array = splice_feature_array(n_gram_frequency_allarray, min_skip_gram_frequency_allarray)
    print("bigram_feature_array:"+str(len(bigram_feature_array[0])-1))
    # x, y = np.split(bigram_feature_array, (len(bigram_feature_array[0])-1,), axis=1)

    # 条件频率矩阵
    positive_conditional_frequency_array = conditional_frequency(positive_frequency_array)
    negative_conditional_frequency_array = conditional_frequency(negative_frequency_array)
    conditional_frequency_array = np.concatenate((positive_conditional_frequency_array, negative_conditional_frequency_array), axis=0)
    print("conditional_frequency_array:"+str(len(conditional_frequency_array[0])-1))
    # x, y = np.split(conditional_frequency_array, (len(conditional_frequency_array[0])-1,), axis=1)
    # for r in conditional_frequency_array:
    #     print(r)

    # 
    only_positive_frequency_array = to_site(positive_array, 1, positive_frequency_site)
    only_negative_frequency_array = to_site(negative_array, 0, negative_frequency_site)
    positive_entropy_array = entropy_of_site(only_positive_frequency_array)
    negative_entropy_array = entropy_of_site(only_negative_frequency_array)
    entropy_allarray = np.concatenate((positive_entropy_array, negative_entropy_array), axis=0)
    two_feature_array = splice_feature_array(conditional_frequency_array, entropy_allarray)
    # x, y = np.split(two_feature_array, (len(two_feature_array[0])-1, ), axis=1)

    # 再加上位置特征集进行预测
    positive_position_array = hydrophobic_position_array0(psiteList, 1)
    negative_position_array = hydrophobic_position_array0(nsiteList, 0)
    position_array = np.concatenate((positive_position_array, negative_position_array), axis=0)
    print("position_array:"+str(len(position_array[0])-1))
    # x, y = np.split(position_array, (len(position_array[0])-1, ), axis=1)

    # 加上n_gram进行预测

    three_feature_array =splice_feature_array(conditional_frequency_array, frequency_allarray)
    # for r in union_frequency_entropy_array:
    #     print(r)
    # x, y = np.split(three_feature_array, (len(three_feature_array[0])-1, ), axis=1)
    four_feature_array = splice_feature_array(frequency_allarray, position_array)
    # x, y = np.split(four_feature_array, (len(four_feature_array[0])-1, ), axis=1)
    five_feature_array = splice_feature_array(position_array, n_gram_frequency_allarray)
    # x, y =np.split(five_feature_array, (len(five_feature_array[0])-1, ), axis=1)

    # 再加上skip_gram进行预测
    six_feature_array = splice_feature_array(five_feature_array, min_skip_gram_frequency_allarray)
    x, y = np.split(six_feature_array, (len(six_feature_array[0])-1, ), axis=1)
    # print(shape(x))a
    # y.ravel('F')
    # LogisticRegression_prediction(x, y.ravel())
    # ExtraTree_prediction(x, y.ravel())
    # RandomForest_prediction(x, y.ravel())
    # BP_prediction(x,y)
    # Stacking(x, y.ravel())
    # GBDT_prediction(x, y.ravel())
    # SVM_prediction(x, y.ravel())
    # GCForest_prediction(x, y.ravel())
    GAGCForest_prediction0(x, y.ravel())
