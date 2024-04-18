
'''
Internal validation using bootstrapping (for CI calculation)

- boostrapping: resample with replacement (n_samples = X.shape[0])
- include hyperparameter tuning 
- n_iter ~ 200 [acc to this](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html#method-22-bootstrap-confidence-intervals-using-the-percentile-method)

(c) Sonja Katz, 2024
'''


PATH = "/home/WUR/katz001/PROJECTS/permit-nsti-gas"

import os
import numpy as np
import json
import pandas as pd
import sys 
import errno  
sys.path.append(f"{PATH}/scripts")
import seaborn as sns 
import matplotlib.pyplot as plt

from func_preprocess import pipe_imputation_scaling, pipe_supervisedSelector
from func_clf import classify_boostrap


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
import pickle


def get_input():
    try:
        datasetTimepoint = sys.argv[1]
        n_bootstrap = sys.argv[2]
    except IndexError:
        print("ERROR\tPlease enter a valid dataset name (ENTRY, PRESURGERY,POSTSURGERY, BL)")
        sys.exit()
    return datasetTimepoint, int(n_bootstrap)


''' 
Prepare data --> change here for different setups!
'''
datasetTimepoint, n_bootstrap = get_input()
#datasetTimepoint = "PRESURGERY"
#n_bootstrap = 1

target = "Conclusion_micro"
percentBoruta = 100
saveFig_quickCheck = True

''' 
Select features
'''
vars = f"{target}_bootstrapped_iterativeBoruta_{percentBoruta}perc"   
varPath = f"{PATH}/results/20_featureSelection/{datasetTimepoint}/CV/{vars}.txt"

''' 
Define paths
'''
folderFigures = f"{PATH}/figures/{datasetTimepoint}/40_internalValidation"
os.makedirs(folderFigures, exist_ok=True)
resultsPath = f"{PATH}/results/40_internalValidation/{datasetTimepoint}"
os.makedirs(resultsPath, exist_ok=True)
dataPath = f"{PATH}/results/10_preprocessed/"
dataset = f"{datasetTimepoint}_{target}_preprocessed.csv"


models = {
          'rfc': RandomForestClassifier(), 
         }               


grids = {'rfc':{
               'classifier__n_estimators': [100, 300, 700],     
               'classifier__max_depth': [2,4,6],         
               'classifier__max_features': [2,4,6],  
               }
         }   

''' 
Read data
'''
data = pd.read_csv(f"{dataPath}/{dataset}", index_col=0)
X = data.drop(target, axis=1)
y = data[target]

# ### FOR DEVELOPMENT PURPOSES: smaller dataset
# X = X.iloc[:20,:]
# y = y[:20]

''' 
Read in variables
'''
sel_variables = pd.read_csv(varPath, header=None)[0].tolist()

''' 
Prepare imputation and scaling
'''
num_columns = X.loc[:,sel_variables].select_dtypes(include=["float64"]).columns
bin_columns = X.loc[:,sel_variables].select_dtypes(include=["int64"]).columns
cat_columns = X.loc[:,sel_variables].select_dtypes(include=["object"]).columns
preprocessor = pipe_imputation_scaling(num_columns, bin_columns, cat_columns)  

''' 
Run Pipeline
'''
model = 'rfc'
dic_summary = dict()

print(X.shape, y.value_counts())

for i in range(n_bootstrap):

    ''' 
    Assemble pipeline
    '''
    pipe = Pipeline([("selector", pipe_supervisedSelector(sel_variables)),
                        ("imputation", preprocessor),
                        ("classifier", models[model])])


    dic_bootstrap_results = classify_boostrap(X, 
                                              y, 
                                              pipe, 
                                              hp_grid=grids[model],
                                              perc_samples_per_boostrap=1)
    dic_summary[i] = dic_bootstrap_results


with open(f'{resultsPath}/bootstrap_{model}_n{n_bootstrap}.pickle', 'wb') as f:
    pickle.dump(dic_summary, f, protocol=pickle.HIGHEST_PROTOCOL)


### Plot KDE for instant checkign if iter=200 is enough (should be normally distributed!!; https://stats.stackexchange.com/questions/86040/rule-of-thumb-for-number-of-bootstrap-samples)
if saveFig_quickCheck:
    sns.set_theme(style="white", font_scale=1.2)
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,8))
    df = pd.DataFrame.from_dict(dic_summary).T
    sns.histplot(data=df["auc"], ax=ax1, alpha=0.6, kde=True, kde_kws={"bw_adjust":0.8}, color=sns.husl_palette()[0])
    sns.histplot(data=df["average_prec"], ax=ax2, alpha=0.6, kde=True, kde_kws={"bw_adjust":0.8}, color=sns.husl_palette()[1])
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax1.set_title("ROC - AUC")
    ax2.set_title("PR")
    plt.tight_layout()
    fig.savefig(f"{folderFigures}/bootstrapping_n{n_bootstrap}.png", dpi=300)



