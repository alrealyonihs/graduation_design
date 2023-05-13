import pandas as pd
import warnings
import copy
import sys
import csv
import datetime

from itertools import cycle
from sklearn import model_selection, metrics,svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

from util.myplots import plotROCs
from util.settings import *
from util.process import *
from util.const import *

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


from util.utils import datasetname, create_userids, keeporder_split


# current_dataset: which dataset to evaluate
# dataset_amount: how many data use from the dataset (ALL, FIRST1000)
# num_actions: how many mouse actions to use for decision

def evaluate_dataset( current_dataset, dataset_amount, num_actions, num_training_actions):
    # filename ='/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/' + FEAT_DIR + '/' + datasetname(current_dataset, dataset_amount, num_training_actions)
    filename ='/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/features/master10Test_Extracted.csv'
    # filename_train = '/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/features/master10Train_Extracted.csv'
    # filename_test = '/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/features/master10Test_Extracted.csv'
    csv_filename = '/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/csv_record/'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.csv'
    csv_file = open(csv_filename, 'w', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    print(filename)
    #print(filename_test)
    dataset = pd.read_csv(filename)
    # dataset_train = pd.read_csv(filename_train)
    #dataset = dataset.drop(columns="num",axis=1)
    # dataset_test = pd.read_csv(filename_test)
    # dataset_test = dataset_test.drop(columns="num",axis=1)

    print(dataset.shape)
    #print(dataset_train.shape)
    csv_writer.writerow(["use:RandomForestClassifier"])


    # DataFrame
    df = pd.DataFrame(dataset)
    # df_train = pd.DataFrame(dataset_train)
    # df_test = pd.DataFrame(dataset_test)
    #print(df)

    num_features = int(dataset.shape[1])
    # num_features = int(dataset_train.shape[1])
    print("Num features: ", num_features)
    csv_writer.writerow(["Num features: ", num_features])
    array = dataset.values
    # array_train = dataset_train.values
    # array_test = dataset_test.values

    X = array[:, 0:num_features - 1]
    y = array[:, num_features - 1]
    # Xtrain = array_train[:, 0:num_features - 1]
    # ytrain = array_train[:, num_features - 1]
    # Xtest = array_test[:, 0:num_features - 1]
    # ytest = array_test[:, num_features - 1]

    #userids = create_userids(current_dataset)
    # userids = [15]
    userids = range(0,10)

    print(userids)
    csv_writer.writerow(userids)


    # Train user-specific classifiers and evaluate them
    items = userids

    # fpr = {} <==> fpr = dict()
    fpr = {}
    tpr = {}
    roc_auc = {}
    avg_acc=0


    for i in userids:
        # print("Training classifier for the user "+str(i))
        # Select all positive samples that belong to current user
        user_positive_data = df.loc[df.iloc[:, -1].isin([i])]
        # user_positive_train_data = df_train.loc[df_train.iloc[:, -1].isin([i])]
        # user_positive_test_data = df_test.loc[df_test.iloc[:, -1].isin([i])]
        #print(user_positive_data)

        numSamples = user_positive_data.shape[0]
        array_positive = copy.deepcopy(user_positive_data.values)
        array_positive[:, -1] = 1

        # numSamples = user_positive_train_data.shape[0]
        # array_positive = copy.deepcopy(user_positive_train_data.values)
        # array_test_positive = copy.deepcopy(user_positive_test_data.values)
        # array_positive[:, -1] = 1
        # array_test_positive[:,-1] = 1

        # negative data for the current user
        user_neagtive_data = select_negatives_from_other_users(dataset, i, numSamples)
        array_negative = copy.deepcopy(user_neagtive_data.values)
        array_negative[:, -1] = 0
        # user_neagtive_data = select_negatives_from_other_users(dataset_train, i, numSamples)
        # user_neagtive_test_data = select_negatives_from_other_users(dataset_test, i, numSamples)
        # array_negative = copy.deepcopy(user_neagtive_data.values)
        # array_test_negative = copy.deepcopy(user_neagtive_test_data.values)
        # array_negative[:, -1] = 0
        # array_test_negative[:,-1] = 0

        # concatenate negative and positive data
        dataset_user = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)]).values
        # dataset_train_user = pd.concat([pd.DataFrame(array_positive), pd.DataFrame(array_negative)]).values
        # dataset_test_user = pd.concat([pd.DataFrame(array_test_positive), pd.DataFrame(array_test_negative)]).values
        print(dataset_user)
        # dataset_train_user = dataset_train_user
        # dataset_test_user = pd.to_numpy(dataset_test_user)
        
        X = dataset_user[:, 0:-1]
        y = dataset_user[:, -1]
        # X_train = dataset_train_user[:, 0:-1]
        # y_train = dataset_train_user[:, -1]
        # X_validation = dataset_test_user[:, 0:-1]
        # y_validation = dataset_test_user[:, -1]

        
        if CURRENT_SPLIT_TYPE == SPLIT_TYPE.RANDOM:
            X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=TEST_SIZE,random_state= RANDOM_STATE)
        else:
            X_train, X_validation, y_train, y_validation = keeporder_split(X, y, test_size=TEST_SIZE)


        #model = GradientBoostingClassifier(random_state= RANDOM_STATE)
        model = RandomForestClassifier(random_state= RANDOM_STATE)
        #model = ExtraTreesClassifier(random_state= RANDOM_STATE)
        #model = BaggingClassifier(random_state= RANDOM_STATE)
        #model = DecisionTreeClassifier()
        #model = AdaBoostClassifier()
        #model = svm.SVC(probability=True)
        #model = KNeighborsClassifier(n_neighbors=20)
        model.fit(X_train, y_train)

        # scoring = ['accuracy', 'roc_auc' ]
        # scores = cross_validate(model, X_train, y_train, scoring=scoring, cv = 10, return_train_score = False)
        scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=False)
        cv_accuracy = scores['test_score']
        print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_accuracy.mean(), cv_accuracy.std() * 2))
        csv_writer.writerow(["CV Accuracy: %0.2f (+/- %0.2f)" % (cv_accuracy.mean(), cv_accuracy.std() * 2)])

        y_predicted = model.predict(X_validation)
        y_prob = model.predict_proba(X_validation)[:,1]
        test_accuracy = accuracy_score(y_validation, y_predicted)
        print("Test Accuracy: %0.2f" % test_accuracy)
        csv_writer.writerow(["Test Accuracy: %0.2f" % test_accuracy])
        avg_acc = avg_acc + test_accuracy

        #get precision
        precision = precision_score(y_validation, y_predicted)
        print("precision:%0.2f" % precision)
        csv_writer.writerow(["precision: %0.2f" % precision])

        #get recall_score
        recall = recall_score(y_validation, y_predicted)
        print("recall_score:%0.2f" % recall)
        csv_writer.writerow(["recall_score: %0.2f" % recall])

        #get f1-score
        f1 = f1_score(y_validation, y_predicted)
        print("F1_score:%0.2f" % f1)
        csv_writer.writerow(["F1_score: %0.2f" % f1])

        #get auc
        auc1 = roc_auc_score(y_validation, y_prob)
        print("auc1:%0.2f" % auc1)
        csv_writer.writerow(["auc: %0.2f" % auc1])

        fpr1, tpr1, _ = roc_curve(y_validation, y_prob)
        auc_score = auc(fpr1 , tpr1)
        print("auc:%0.2f" % auc_score)

        #get ROC curve
        # fpr,tpr,threshold = roc_curve(y_validation, y_predicted)
        # fig,ax = plt.subplots()
        # roc_display = RocCurveDisplay.from_predictions(y_validation, y_predicted, pos_label=1)
        # roc_display.plot()
        # plt.title("ROC for user15 under ExtraTreesClassifier")
        # # plt.show()
        # filename = '/home/liuyanling/Code/mouse_dynamics_balabit_chaoshen_dfl-master/ROC/ROC15 for ExtraTreesClassifier.png'
        # plt.savefig(filename)

        fpr[i], tpr[i], thr = evaluate_sequence_of_samples(model, X_validation, y_validation, num_actions)

        threshold = -1
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr[i], tpr[i])(x), 0., 1.)
            threshold = interp1d(fpr[i], thr)(eer)
        except (ZeroDivisionError, ValueError):
            print("Division by zero")

        roc_auc[i] = auc(fpr[i], tpr[i])
        print(str(i) + ": " + str(roc_auc[i])+" threshold: "+str(threshold))
        print("")
        csv_writer.writerow([str(i) + ": " + str(roc_auc[i])+" threshold: "+str(threshold)])

    plotROCs(fpr, tpr, roc_auc, items)
    avg_acc = avg_acc/10
    print("Average_accuracy:%0.4f" % avg_acc)
    csv_writer.writerow(["Average_accuracy: %0.4f" % avg_acc])

def evaluate_sequence_of_samples(model, X_validation, y_validation, num_actions):
    # print(len(X_validation))
    if num_actions == 1:
        y_scores = model.predict_proba(X_validation)
        writeCSVa(y_validation, y_scores[:, 1])
        return roc_curve(y_validation, y_scores[:, 1])

    X_val_positive = []
    X_val_negative = []
    for i in range(len(y_validation)):
        if y_validation[i] == 1:
            X_val_positive.append(X_validation[i])
        else:
            X_val_negative.append(X_validation[i])
    pos_scores = model.predict_proba(X_val_positive)
    neg_scores = model.predict_proba(X_val_negative)

    scores =[]
    labels =[]

    n_pos = len(X_val_positive)
    for i in range(n_pos-num_actions+1):
        score = 0
        for j in range(num_actions):
            score += pos_scores[i+j][1]
        score /= num_actions
        scores.append(score)
        labels.append(1)

    n_neg = len(X_val_negative)
    for i in range(n_neg - num_actions + 1):
        score = 0
        for j in range(num_actions):
            score += neg_scores[i + j][1]
        score /= num_actions
        scores.append(score)
        labels.append(0)

   # writeCSVa(labels, scores)
    return roc_curve(labels, scores)

def select_negatives_from_other_users( dataset, userid, numsamples ):
    # num_features = dataset.shape[1]
    #other_users_data =  dataset['userid'] != userid
    other_users_data =  dataset['class'] != userid
    dataset_negatives = dataset[other_users_data].sample(numsamples, random_state= RANDOM_STATE)
    return dataset_negatives

