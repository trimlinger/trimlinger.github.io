#!/usr/bin/env python3

# Importing all the libraries we need
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
#from scipy.optimize import curve_fit

model_cols = ['minority_portion', 'recall', 'precision', 'Accuracy']
zeros = pd.Series([0]*3)
a_str = 'Overall accuracy is {:.1%}'
r_str = 'Recall: {} positive (default) cases, {} ({:.1%}) correct predictions'
p_str = 'Precision: {} positive (default) predictions, {} ({:.1%}) true positives'
cbar_num_format = '%.1f'

###############################################################################
# This function runs the neural network to evaluate all accuracy metrics
###############################################################################
def run_model(X_train, X_test, y_train, y_test, verbose=False):

    #if verbose:
    #    print('Run number: {}'.format(index))
    #    print('______________')
    #    print(y_train.value_counts())

    # Scale data with StandardScaler
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # This model uses the MLPClassifier neural network
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic',
                        max_iter=2000, random_state=42)

    # Fit model using training data
    mlp.fit(X_train, y_train)

    # Get model predictions
    predictions = mlp.predict(X_test)

    # Calculate number of true/false positives, true/false negatives, and
    # other related info
    N_all = y_test.size
    N_actualPos = np.count_nonzero(y_test)
    N_actualNeg = N_all - N_actualPos

    tp_sum_counts = (y_test + predictions).value_counts().sort_index()

    N_trueNeg, N_false, N_truePos = tp_sum_counts.add(zeros, fill_value=0)
    N_falseNeg = N_actualPos - N_truePos
    N_true = N_all - N_false
    N_falsePos = N_false - N_falseNeg
    N_Pos = N_truePos + N_falsePos

    # Calculate model accuracy
    accuracy = N_true / N_all
    if verbose:
        print(a_str.format(accuracy))

    # Calculate model recall
    recall = N_truePos / N_actualPos
    if verbose:
        print(r_str.format(N_actualPos, N_truePos, recall))

    # Calculate model precision, and exit if undefined
    try:
        precision = N_truePos / N_Pos
        if verbose:
            print(p_str.format(N_Pos, N_truePos, precision))
    except:
        print('No default predictions'); exit()

    return accuracy, recall, precision

###############################################################################
# 
###############################################################################
def interactive_graph(data, algo):
    title = 'Precision Recall Curves for {} Algorithm'.format(algo)
    hover_data = dict(list(zip(model_cols, [':.3f']*len(model_cols))))
    fig = px.scatter(data, x='recall', y='precision', height=600, width=600,
                     title=title, color='minority_portion',
                     hover_data=hover_data)

    fig.show()
    
###############################################################################
# Sampling algorithms for oversample, undersample, and smote
###############################################################################
def sample(model_data, add_n=None, remove_n=None, ratio=None):
    X_train, X_test, y_train, y_test = model_data
    if remove_n:
        i_drop = np.random.choice(false_train_index, remove_n, replace=False)
        X_train = X_train.drop(i_drop)
        y_train = y_train.drop(i_drop)
        
    elif add_n:
        i_add = true_train.sample(n=add_n, replace=True, random_state=42).index
        X_train = X_train.append(X_train.loc[i_add], ignore_index=True)
        y_train = y_train.append(y_train.loc[i_add], ignore_index=True)

    elif ratio:
        sm = SMOTE(sampling_strategy=ratio, random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        
    minority_portion = y_train[y_train==1].size / y_train.size
    acc, rec, prec = run_model(X_train, X_test, y_train, y_test)

    return [minority_portion, rec, prec, acc]

###############################################################################
# Script begins here
###############################################################################
# Loading data into dataframe
drugs = pd.read_csv('drug_shortage_prediction.csv', header=1, low_memory=False)

# ETL to clean up the data file
drugs_formatted = drugs.copy()

# drop monograph_name column
drugs_formatted = drugs_formatted.drop('monograph_name', axis=1)

# nonbinary columns
nb_col = ['price_scaled_dosage_form', 'total_inspections', 'age_of_drug', 'mw']
for col in drugs_formatted:
    if col not in nb_col:
        drugs_formatted[col] = drugs_formatted[col].astype(bool).astype(int)

# Define target column
target = 'rolling12_shortage' #'loan_status'

# Define the standard, unsampled training and testing datasets
df_copy = drugs_formatted.copy()
X = df_copy.drop(target, axis=1)
y = df_copy[target]
model_data = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = model_data

false_train = y_train[y_train==0]
false_train_index = false_train.index
false_train_size = false_train.size
true_train = y_train[y_train==1]
true_train_size = true_train.size

true_test_size = y_test[y_test==1].size
# Base Case
#index = 'Base Case'
#accuracy, recall, precision = run_model(*model_data)
#exit()

# Over sampling -- need to play around with start and stop
start = 0
stop = y_train.size / 2.
add_n_values = np.linspace(start, stop, num=50, dtype=int)
oversample_data = [sample(model_data, add_n=add_n) for add_n in add_n_values]
over_sampling = pd.DataFrame(oversample_data, columns=model_cols)

interactive_graph(data=over_sampling, algo='Oversampling')
over_sampling.to_csv('over_sampling.csv')
exit()

"""
# Under sampling -- need to play around with start and stop
start = true_train_size/2
stop = true_train_size
remove_n_values = np.linspace(start, stop, num=5, dtype=int)
undersample_data = [sample(model_data, remove_n=remove_n)
                    for remove_n in remove_n_values]
under_sampling = pd.DataFrame(undersample_data, columns=model_cols)

interactive_graph(data=under_sampling, algo='Undersampling')
under_sampling.to_csv('under_sampling.csv')
"""

# SMOTE sampling
#ratios = np.linspace(0.15, 1., num=5)
#smote_data = [sample(model_data, ratio=ratio) for ratio in ratios]
#smote_sampling = pd.DataFrame(smote_data, columns=model_cols)

#interactive_graph(data=smote_sampling, algo='SMOTE')
#smote_sampling.to_csv('smote_sampling.csv')

exit()


































under_sampling = pd.read_csv('under_sampling.csv')
over_sampling = pd.read_csv('over_sampling.csv')
smote_sampling = pd.read_csv('smote_sampling.csv')

temp += 1

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)

ax1.scatter(under_sampling['recall'], under_sampling['precision'], c=under_sampling['minority_portion'], marker='o')
ax2.scatter(over_sampling['recall'], over_sampling['precision'], c=over_sampling['minority_portion'], marker='o')
ax3.scatter(smote_sampling['recall'], smote_sampling['precision'], c=smote_sampling['minority_portion'], marker='o')

ax1.set_title('Undersampling')
ax2.set_title('Oversampling')
ax3.set_title('SMOTE')

ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')

im = plt.gca().get_children()[0]
cbar_num_format = '%.1f'

cbar = fig.colorbar(im, format=cbar_num_format)
cbar.ax.set_yticklabels(["{:.0%}".format(i) for i in cbar.get_ticks()])
cbar.ax.set_ylabel('Minority Class Portion in Train Cases', rotation=270, labelpad=15)

file_name = str(temp)+'.jpg'
plt.savefig(file_name, dpi=300)

plt.show()

plt.figure(figsize=(8, 4))
graph_data = pd.DataFrame(columns=['Recall', 'Precision', 'Loss'])
precision = np.arange(0.44, 0.9, 0.001)

for loss in range(3400000, 4000000, 75000):
    temp_data = pd.DataFrame()
    temp_data['Precision'] = precision
    temp_data['Recall'] = (-loss/pos_base_case + expected_loss_bad_loan + expected_profit_good_loan/precision
                          )/(expected_profit_good_loan + expected_loss_bad_loan)
    temp_data['Loss'] = loss   
    graph_data = graph_data.append(temp_data)
        
plot_subset = under_sampling[under_sampling['recall'].between(0, 1)].copy() 
plt.scatter(plot_subset['recall'], plot_subset['precision'], marker='o', zorder=1)
scatter=plt.scatter(graph_data['Recall'], graph_data['Precision'], c=graph_data['Loss'], marker='o', cmap='Wistia', zorder=0)
cb = plt.colorbar(scatter)
cb.ax.set_yticklabels(["$ {:,.0f}".format(i) for i in cb.get_ticks()])
cb.ax.set_ylabel('Loss Due to Model Misclassifications', rotation=270, labelpad=15)
plt.title('Isoprofit Curves for Undersampling Algorithm')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0.44, 0.60)
plt.ylim(0.6, 0.9)
plt.savefig('indifference.jpg', dpi=300)
plt.show()



###############################################################################
# Matt's original plotting routine
###############################################################################
def show_graph(data, algo):

    fig = plt.figure(figsize=(8, 4))
    px.scatter(data['recall'], data['precision'], 
             c=data['minority_portion'], marker='o')
    im = plt.gca().get_children()[0]
    cbar = fig.colorbar(im, format=cbar_num_format)
    cbar.ax.set_yticklabels(["{:.0%}".format(i) for i in cbar.get_ticks()])
    cbar.ax.set_ylabel('Minority Class Portion in Train Cases', rotation=270, labelpad=15)
    plt.title('Precision Recall Curves for {} Algorithm'.format(algo))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    file_name = algo+'.jpg'
    plt.savefig(file_name, dpi=300)

    plt.show()

#loans_formatted['loan_status'] = loans_formatted['loan_status'].astype('float64')
#loans_formatted = loans_formatted.dropna(axis=1, how='all')
#columns_to_drop = ['sub_grade', 'emp_title', 'pymnt_plan', 'url', 'zip_code', 'purpose', 'addr_state', 
#                   'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d']
#loans_formatted = loans_formatted.drop(columns_to_drop, axis=1)
#loans_formatted['id'] = loans_formatted['id'].astype(float)
#loans_formatted['term'] = loans_formatted['term'].str.extract('(\d+)').astype(float)
#loans_formatted['int_rate'] = loans_formatted['int_rate'].str.extract('(\d+.\d+)').astype(float)/100
#loans_formatted['emp_length'] = loans_formatted['emp_length'].str.extract('(\d+)').astype(float).fillna(0)
#loans_formatted['issue_d'] = loans_formatted['issue_d'].str[:3]
#loans_formatted['earliest_cr_line'] = loans_formatted['earliest_cr_line'].str[-2:].astype(float)
#loans_formatted['revol_util'] = loans_formatted['revol_util'].str.extract('(\d+.\d+)').astype(float)/100

#num_records = loans_formatted.shape[0]
#for col in loans_formatted.columns:
#    if (loans_formatted[col].notna().sum() < num_records/2):
#        loans_formatted = loans_formatted.drop(col, axis=1)
#    else:      
#        if (loans_formatted[col].dtype != 'float64'):
#            loans_formatted = pd.get_dummies(loans_formatted, columns=[col])
#        elif (loans_formatted[col].isna().sum() != loans_formatted.shape[0]):
#            loans_formatted[col] = loans_formatted[col].fillna(loans_formatted[col].mean())
#loans_formatted

#loans_formatted.info(max_cols=200)




#false_positives = run_results[(run_results[col_name]==0) & (run_results['Predictions']==1)].index
    #false_negatives = run_results[(run_results[col_name]==1) & (run_results['Predictions']==0)].index
    
    #loss_opp_good_loan = x_test.loc[false_positives,'funded_amnt_inv'].multiply(
    #    x_test.loc[false_positives,'int_rate'].multiply(
    #        x_test.loc[false_positives,'term'])).sum()/12

    #loss_bad_loan = x_test.loc[false_negatives,'funded_amnt_inv'].sum()    

