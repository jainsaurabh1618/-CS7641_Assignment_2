import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from ml import helper
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import warnings
import time
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

import mlrose_hiive

warnings.filterwarnings("ignore")

# wine = pd.read_csv('wine_red', sep=';')
# wine.loc[wine['quality'] <= 6, 'quality'] = 0
# wine.loc[wine['quality'] > 6, 'quality'] = 1
# data_1 = wine  # pd.concat([wine_low, wine_med, wine_high])
# data_1 = shuffle(data_1, random_state=42)
# data_1_feature = data_1.drop('quality', axis=1)
# data_1_feature = data_1[['alcohol', 'sulphates', 'volatile acidity', 'citric acid', 'density']]
# data_1_label = data_1['quality']

data_1, data_1_feature, data_1_label = helper.get_heart_data()

mx_attempt = 10
mx_iter = 5000
hid_nodes = [5, 3]
learn_rate = .001

X_train, X_test, y_train, y_test = train_test_split(data_1_feature, data_1_label, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = mlrose_hiive.NeuralNetwork(hidden_nodes=hid_nodes, activation='relu',
                                 algorithm='gradient_descent', early_stopping=True,
                                 max_attempts=mx_attempt, max_iters=mx_iter,
                                 bias=True, learning_rate=learn_rate,
                                 curve=True, random_state=42)

start = time.time()
clf.fit(X_train, y_train)
end = time.time()
nn_train_time_backprop = end - start

y_train_pred = clf.predict(X_train)
acc_score_train_bakprop = accuracy_score(y_train, y_train_pred)
start = time.time()
y_pred = clf.predict(X_test)
end = time.time()
nn_test_time_backprop = end - start
acc_score_test_bakprop = accuracy_score(y_test, y_pred)

backprop_fitness_curve = clf.fitness_curve

clf_backprop_final = clf

print("Backdrop")
print('acc_score_train_bakprop: ', acc_score_train_bakprop)
print('acc_score_test_bakprop: ', acc_score_test_bakprop)
print('nn_train_time_backprop: ', nn_train_time_backprop)
print('nn_test_time_backprop: ', nn_test_time_backprop)
print('clf.fitted_weights: ', clf.fitted_weights)

plt.plot(backprop_fitness_curve, label='Backprop')
plt.title('Iterations vs. Fitness (Backprop)')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('images/nn_backprop_iter_fitness.png')
plt.clf()

plot_confusion_matrix(clf, X_test, y_test)
plt.savefig('images/backprop_confusion_matrix.png')
plt.clf()

# RHC
restart_range = [1, 10, 20, 30, 40, 50]
nn_train_time_rhc_final = 0
nn_test_time_rhc_final = 0
acc_score_train_rhc_final = 0
acc_score_test_rhc_final = 0
rhc_fitness_curve_final = None
restart_val_final = 0
clf_rhc_final = None
rhc_fit_curve = []

print("RHC")
for restart_val in restart_range:
    clf = mlrose_hiive.NeuralNetwork(hidden_nodes=hid_nodes, activation='relu',
                                     algorithm='random_hill_climb', early_stopping=True,
                                     max_attempts=mx_attempt, max_iters=mx_iter,
                                     bias=True, learning_rate=learn_rate,
                                     restarts=restart_val, curve=True, random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    nn_train_time_rhc = end - start
    y_train_pred = clf.predict(X_train)
    acc_score_train_rhc = accuracy_score(y_train, y_train_pred)

    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    nn_test_time_rhc = end - start
    acc_score_test_rhc = accuracy_score(y_test, y_pred)

    rhc_fitness_curve = clf.fitness_curve
    rhc_fit_curve.append(clf.fitness_curve)

    if acc_score_test_rhc > acc_score_test_rhc_final:
        nn_train_time_rhc_final = nn_train_time_rhc
        nn_test_time_rhc_final = nn_test_time_rhc
        acc_score_train_rhc_final = acc_score_train_rhc
        acc_score_test_rhc_final = acc_score_test_rhc
        rhc_fitness_curve_final = rhc_fitness_curve
        restart_val_final = restart_val
        clf_rhc_final = clf

print('acc_score_train_rhc_final: ', acc_score_train_rhc_final)
print('acc_score_test_rhc_final: ', acc_score_test_rhc_final)
print('nn_train_time_rhc_final: ', nn_train_time_rhc_final)
print('nn_test_time_rhc_final: ', nn_test_time_rhc_final)
print(restart_val_final)

plt.plot(rhc_fit_curve[0][:, 0], label='Restart - 1')
plt.plot(rhc_fit_curve[1][:, 0], label='Restart - 10')
plt.plot(rhc_fit_curve[2][:, 0], label='Restart - 20')
plt.plot(rhc_fit_curve[3][:, 0], label='Restart - 30')
plt.plot(rhc_fit_curve[4][:, 0], label='Restart - 40')
plt.plot(rhc_fit_curve[5][:, 0], label='Restart - 50')
plt.title('Iterations vs. Loss (RHC)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/nn_rhc_iter_loss.png')
plt.clf()

plt.plot(rhc_fit_curve[0][:, 1], label='Restart - 1')
plt.plot(rhc_fit_curve[1][:, 1], label='Restart - 10')
plt.plot(rhc_fit_curve[2][:, 1], label='Restart - 20')
plt.plot(rhc_fit_curve[3][:, 1], label='Restart - 30')
plt.plot(rhc_fit_curve[4][:, 1], label='Restart - 40')
plt.plot(rhc_fit_curve[5][:, 1], label='Restart - 50')
plt.title('Iterations vs. Function Evals - RHC')
plt.xlabel('Iterations')
plt.ylabel('Function Evaluations')
plt.legend()
plt.savefig('images/nn_rhc_iter_funceval.png')
plt.clf()

plot_confusion_matrix(clf_rhc_final, X_test, y_test)
plt.savefig('images/rhc_confusion_matrix.png')
plt.clf()
##########################################################################


# SA
sch_range = [mlrose_hiive.ExpDecay(), mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay()]
nn_train_time_sa_final = 0
nn_test_time_sa_final = 0
acc_score_train_sa_final = 0
acc_score_test_sa_final = 0
sa_fitness_curve_final = None
sch_val_final = 0
clf_sa_final = None
sa_fit_curve = []

print("SA")
for sch in sch_range:
    clf = mlrose_hiive.NeuralNetwork(hidden_nodes=hid_nodes, activation='relu',
                                     algorithm='simulated_annealing', early_stopping=True,
                                     max_attempts=mx_attempt, max_iters=mx_iter,
                                     bias=True, learning_rate=learn_rate,
                                     schedule=sch, curve=True, random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    nn_train_time_sa = end - start
    y_train_pred = clf.predict(X_train)
    acc_score_train_sa = accuracy_score(y_train, y_train_pred)

    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    nn_test_time_sa = end - start
    acc_score_test_sa = accuracy_score(y_test, y_pred)

    sa_fitness_curve = clf.fitness_curve
    sa_fit_curve.append(clf.fitness_curve)

    if acc_score_test_sa > acc_score_test_sa_final:
        nn_train_time_sa_final = nn_train_time_sa
        nn_test_time_sa_final = nn_test_time_sa
        acc_score_train_sa_final = acc_score_train_sa
        acc_score_test_sa_final = acc_score_test_sa
        sa_fitness_curve_final = sa_fitness_curve
        sch_val_final = sch
        clf_sa_final = clf

print('acc_score_train_sa_final: ', acc_score_train_sa_final)
print('acc_score_test_sa_final: ', acc_score_test_sa_final)
print('nn_train_time_sa_final: ', nn_train_time_sa_final)
print('nn_test_time_sa_final: ', nn_test_time_sa_final)
print(sch_val_final)

plt.plot(sa_fit_curve[0][:, 0], label='Schedule - Exp')
plt.plot(sa_fit_curve[1][:, 0], label='Schedule - Geom')
plt.plot(sa_fit_curve[2][:, 0], label='Schedule - Arith')
plt.title('Iterations vs. Loss (SA)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/nn_sa_iter_loss.png')
plt.clf()

plt.plot(sa_fit_curve[0][:, 1], label='Schedule - Exp')
plt.plot(sa_fit_curve[1][:, 1], label='Schedule - Geom')
plt.plot(sa_fit_curve[2][:, 1], label='Schedule - Arith')
plt.title('Iterations vs. Function Evals - SA')
plt.xlabel('Iterations')
plt.ylabel('Function Evaluations')
plt.legend()
plt.savefig('images/nn_sa_iter_funceval.png')
plt.clf()

plot_confusion_matrix(clf_sa_final, X_test, y_test)
plt.savefig('images/sa_confusion_matrix.png')
plt.clf()
##########################################################################

# GA
mut_prob_range = np.arange(0.1, 1, 0.2)
nn_train_time_ga_final = 0
nn_test_time_ga_final = 0
acc_score_train_ga_final = 0
acc_score_test_ga_final = 0
ga_fitness_curve_final = None
mut_prob_val_final = 0
clf_ga_final = None
ga_fit_curve = []

print("GA")
for mut_prob in mut_prob_range:
    clf = mlrose_hiive.NeuralNetwork(hidden_nodes=hid_nodes, activation='relu',
                                     algorithm='genetic_alg', early_stopping=True,
                                     max_attempts=mx_attempt, max_iters=mx_iter,
                                     bias=True, learning_rate=learn_rate,
                                     mutation_prob=mut_prob, curve=True, random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    nn_train_time_ga = end - start
    y_train_pred = clf.predict(X_train)
    acc_score_train_ga = accuracy_score(y_train, y_train_pred)

    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    nn_test_time_ga = end - start
    acc_score_test_ga = accuracy_score(y_test, y_pred)

    ga_fitness_curve = clf.fitness_curve
    ga_fit_curve.append(clf.fitness_curve)

    if acc_score_test_ga > acc_score_test_ga_final:
        nn_train_time_ga_final = nn_train_time_ga
        nn_test_time_ga_final = nn_test_time_ga
        acc_score_train_ga_final = acc_score_train_ga
        acc_score_test_ga_final = acc_score_test_ga
        ga_fitness_curve_final = ga_fitness_curve
        mut_prob_val_final = mut_prob
        clf_ga_final = clf

print('acc_score_train_ga_final: ', acc_score_train_ga_final)
print('acc_score_test_ga_final: ', acc_score_test_ga_final)
print('nn_train_time_ga_final: ', nn_train_time_ga_final)
print('nn_test_time_ga_final: ', nn_test_time_ga_final)
print(mut_prob_val_final)

plt.plot(ga_fit_curve[0][:, 0], label='Mut Prob - 0.1')
plt.plot(ga_fit_curve[1][:, 0], label='Mut Prob - 0.3')
plt.plot(ga_fit_curve[2][:, 0], label='Mut Prob - 0.5')
plt.plot(ga_fit_curve[3][:, 0], label='Mut Prob - 0.7')
plt.plot(ga_fit_curve[4][:, 0], label='Mut Prob - 0.9')
plt.title('Iterations vs. Loss (GA)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/nn_ga_iter_loss.png')
plt.clf()

plt.plot(ga_fit_curve[0][:, 1], label='Mut Prob - 0.1')
plt.plot(ga_fit_curve[1][:, 1], label='Mut Prob - 0.3')
plt.plot(ga_fit_curve[2][:, 1], label='Mut Prob - 0.5')
plt.plot(ga_fit_curve[3][:, 1], label='Mut Prob - 0.7')
plt.plot(ga_fit_curve[4][:, 1], label='Mut Prob - 0.9')
plt.title('Iterations vs. Function Evals - GA')
plt.xlabel('Iterations')
plt.ylabel('Function Evaluations')
plt.legend()
plt.savefig('images/nn_ga_iter_funceval.png')
plt.clf()

plot_confusion_matrix(clf_ga_final, X_test, y_test)
plt.savefig('images/ga_confusion_matrix.png')
plt.clf()
##########################################################################

train_times = [round(nn_train_time_backprop, 2), round(nn_train_time_rhc_final, 2), round(nn_train_time_sa_final, 2),
               round(nn_train_time_ga_final, 2)]
plt.barh(['GD', 'RHC', 'SA', 'GA'], train_times, height=.4)
for index, value in enumerate(train_times):
    plt.text(value, index, str(value))

plt.title('Train Time Comparison')
plt.xlabel('Time Train(sec)')
plt.ylabel('Algo Names')
plt.savefig('images/train_time_comp.png')
plt.clf()

test_times = [round(nn_test_time_backprop*1000, 2), round(nn_test_time_rhc_final*1000, 2),
              round(nn_test_time_sa_final*1000, 2), round(nn_test_time_ga_final*1000, 2)]
plt.barh(['GD', 'RHC', 'SA', 'GA'], test_times, height=.4)
for index, value in enumerate(test_times):
    plt.text(value, index, str(value))

plt.title('Test Time Comparison')
plt.xlabel('Time Test(millisec)')
plt.ylabel('Algo Names')
plt.savefig('images/test_time_comp.png')
plt.clf()

##########################################################################

accuracy_scores_train = [round(acc_score_train_bakprop, 2), round(acc_score_train_rhc_final, 2),
                         round(acc_score_train_sa_final, 2), round(acc_score_train_ga_final, 2)]
plt.barh(['GD', 'RHC', 'SA', 'GA'], accuracy_scores_train, height=.4)
for index, value in enumerate(accuracy_scores_train):
    plt.text(value, index, str(value))

plt.title('Train Accuracy Comparison')
plt.xlabel('Train Accuracy')
plt.ylabel('Algo Names')
plt.savefig('images/train_acc_comp.png')
plt.clf()

accuracy_scores_test = [round(acc_score_test_bakprop, 2), round(acc_score_test_rhc_final, 2),
                        round(acc_score_test_sa_final, 2), round(acc_score_test_ga_final, 2)]
plt.barh(['GD', 'RHC', 'SA', 'GA'], accuracy_scores_test, height=.4)
for index, value in enumerate(accuracy_scores_test):
    plt.text(value, index, str(value))

plt.title('Test Accuracy Comparison')
plt.xlabel('Test Accuracy')
plt.ylabel('Algo Names')
plt.savefig('images/test_acc_comp.png')
plt.clf()
