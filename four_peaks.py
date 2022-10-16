import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score
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

import mlrose_hiive

warnings.filterwarnings("ignore")

problem_size_range = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
fit_rhc = []
fit_sa = []
fit_ga = []
fit_mimic = []

time_rhc = []
time_sa = []
time_ga = []
time_mimic = []
fit_curve_rhc = None
fit_curve_sa = None
fit_curve_ga = None
fit_curve_mimic = None

mx_attempt = 10
mx_iter = 5000
algo_name = 'Four Peaks'
file_prefix = 'fp'

for prob_size in problem_size_range:
    print(prob_size)
    fit_fn = mlrose_hiive.FourPeaks()
    ds_ops = mlrose_hiive.DiscreteOpt(length=prob_size, fitness_fn=fit_fn)
    # RHC
    print('RHC')
    start = time.time()
    _, rhc_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(ds_ops, max_attempts=mx_attempt,
                                                                       max_iters=mx_iter, curve=True, random_state=42,
                                                                       restarts=100)
    fit_curve_rhc = rhc_fitness_curve
    end = time.time()
    rhc_time = end - start

    # SA
    print('SA')
    start = time.time()
    _, sa_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(ds_ops, schedule=mlrose_hiive.GeomDecay(),
                                                                       max_attempts=mx_attempt, max_iters=mx_iter,
                                                                       random_state=42, curve=True)
    fit_curve_sa = sa_fitness_curve
    end = time.time()
    sa_time = end - start

    # GA
    print('GA')
    start = time.time()
    _, ga_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(ds_ops, max_attempts=mx_attempt, max_iters=mx_iter,
                                                               random_state=42, curve=True, mutation_prob=0.3)
    fit_curve_ga = ga_fitness_curve
    end = time.time()
    ga_time = end - start

    # MIMIC
    print('MIMIC')
    start = time.time()
    _, mimic_fitness, mimic_fitness_curve = mlrose_hiive.mimic(ds_ops, max_attempts=mx_attempt, max_iters=mx_iter,
                                                               curve=True, random_state=42)
    fit_curve_mimic = mimic_fitness_curve
    end = time.time()
    mimic_time = end - start

    fit_rhc.append(rhc_fitness)
    fit_sa.append(sa_fitness)
    fit_ga.append(ga_fitness)
    fit_mimic.append(mimic_fitness)

    print('RHC Fitness ', rhc_fitness)
    print('SA Fitness ', sa_fitness)
    print('GA Fitness ', ga_fitness)
    print('MIMIC Fitness ', mimic_fitness)

    time_rhc.append(rhc_time)
    time_sa.append(sa_time)
    time_ga.append(ga_time)
    time_mimic.append(mimic_time)

plt.plot(problem_size_range, fit_rhc, 'o-', label='RHC')
plt.plot(problem_size_range, fit_sa, 'o-', label='SA')
plt.plot(problem_size_range, fit_ga, 'o-', label='GA')
plt.plot(problem_size_range, fit_mimic, 'o-', label='MIMIC')
plt.title('Fitness vs. Problem Size - ' + algo_name)
plt.xlabel('Problem Size')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('images/'+file_prefix+'_prob_size_fit.png')
plt.clf()

plt.plot(problem_size_range, time_rhc, 'o-', label='RHC')
plt.plot(problem_size_range, time_sa, 'o-', label='SA')
plt.plot(problem_size_range, time_ga, 'o-', label='GA')
plt.plot(problem_size_range, time_mimic, 'o-', label='MIMIC')
plt.title('Exec Time vs. Problem Size - ' + algo_name)
plt.xlabel('Problem Size')
plt.ylabel('Execution Time(s)')
plt.legend()
plt.savefig('images/'+file_prefix+'_prob_size_time.png')
plt.clf()

plt.plot(fit_curve_rhc[:, 0], 'o-', label='RHC')
plt.plot(fit_curve_sa[:, 0], 'o-', label='SA')
plt.plot(fit_curve_ga[:, 0], 'o-', label='GA')
plt.plot(fit_curve_mimic[:, 0], 'o-', label='MIMIC')
plt.title('Iterations vs. Fitness(Problem Size - ' + str(prob_size) + ') ' + algo_name)
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('images/'+file_prefix+'_iter_fit.png')
plt.clf()

plt.plot(fit_curve_rhc[:, 1], 'o-', label='RHC')
plt.plot(fit_curve_sa[:, 1], 'o-', label='SA')
plt.plot(fit_curve_ga[:, 1], 'o-', label='GA')
plt.plot(fit_curve_mimic[:, 1], 'o-', label='MIMIC')
plt.title('Iterations vs. Function Evals - ' + algo_name)
plt.xlabel('Iterations')
plt.ylabel('Function Evaluations')
plt.legend()
plt.savefig('images/'+file_prefix+'_iter_funceval.png')
plt.clf()

# Hyperparameter tuning

# RHC
restart_range = range(1, 51, 5)
fit_rhc = []
for restart_val in restart_range:
    _, rhc_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(ds_ops, max_attempts=mx_attempt,
                                                                       max_iters=mx_iter, curve=True, random_state=42,
                                                                       restarts=restart_val)
    fit_rhc.append(rhc_fitness)

plt.plot(restart_range, fit_rhc, 'o-')
plt.title('Restarts vs. Fitness - RHC hyperparameter tuning - ' + algo_name)
plt.xlabel('Restarts')
plt.ylabel('Fitness')
plt.savefig('images/'+file_prefix+'_rhc_restart_fit.png')
plt.clf()

# SA
sch_range = [mlrose_hiive.ExpDecay(), mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay()]
fit_sa = []
for sch in sch_range:
    _, sa_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(ds_ops, schedule=sch,
                                                                       max_attempts=mx_attempt, max_iters=mx_iter,
                                                                       random_state=42, curve=True)
    fit_sa.append(sa_fitness)

plt.plot(['Exponential', 'Geometric', 'Arithmetic'], fit_sa, 'o-')
plt.title('Schedule vs. Fitness - SA hyperparameter tuning - ' + algo_name)
plt.xlabel('Schedule')
plt.ylabel('Fitness')
plt.savefig('images/'+file_prefix+'_sa_sch_fit.png')
plt.clf()

# GA
#pop_range = range(100, 500, 100)
mut_prob_range = np.arange(0.1, 1, 0.2)
fit_ga = []
#for pop_val in pop_range:
for mut_prob in mut_prob_range:
    _, ga_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(ds_ops, max_attempts=mx_attempt, max_iters=mx_iter,
                                                               random_state=42, curve=True, mutation_prob=mut_prob)
    fit_ga.append(ga_fitness)

plt.plot(mut_prob_range, fit_ga, 'o-')
plt.title('Mutation probability vs. Fitness - GA hyperparameter tuning - ' + algo_name)
plt.xlabel('Mutation probability')
plt.ylabel('Fitness')
plt.savefig('images/'+file_prefix+'_ga_mut_prob_fit.png')
plt.clf()

# MIMIC
pct_range = np.arange(0.1, 1, 0.2)
fit_mimic = []
for pct_val in pct_range:
    _, mimic_fitness, mimic_fitness_curve = mlrose_hiive.mimic(ds_ops, max_attempts=mx_attempt, max_iters=mx_iter, curve=True,
                                                               random_state=42, keep_pct=pct_val)
    fit_mimic.append(mimic_fitness)

plt.plot(np.arange(0.1, 1, 0.2), fit_mimic, 'o-')
plt.title('keep_pct vs. Fitness - MIMIC hyperparameter tuning - ' + algo_name)
plt.xlabel('Keep Percentage')
plt.ylabel('Fitness')
plt.savefig('images/'+file_prefix+'_mimic_keeppct_fit.png')
plt.clf()