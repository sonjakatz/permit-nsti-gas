
'''
Investigate how stable predictions are wrt random model initialisations

- fix the CV seed to compare same data splits between iterations
- include hyperparameter tuning
- n_iter ~ 100

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


from func_preprocess import pipe_imputation_scaling, pipe_supervisedSelector
from func_clf import classify_dcv


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
import pickle


def get_input():
    try:
        datasetTimepoint = sys.argv[1]
    except IndexError:
        print("ERROR\tPlease enter a valid dataset name (ENTRY, PRESURGERY,POSTSURGERY, BL)")
        sys.exit()
    return datasetTimepoint


''' 
Prepare data --> change here for different setups!
'''
datasetTimepoint = get_input()
#datasetTimepoint = "PRESURGERY"
target = "Conclusion_micro"
percentBoruta = 100

''' 
Select features
'''
vars = f"{target}_bootstrapped_iterativeBoruta_{percentBoruta}perc"   
varPath = f"{PATH}/results/20_featureSelection/{datasetTimepoint}/CV/{vars}.txt"

''' 
Define paths
'''
resultsPath = f"{PATH}/results/41_seedStability/{datasetTimepoint}"
os.makedirs(resultsPath, exist_ok=True)
dataPath = f"{PATH}/results/10_preprocessed/"
dataset = f"{datasetTimepoint}_{target}_preprocessed.csv"


models = {
          'log': LogisticRegression(max_iter=5000),
          'rfc': RandomForestClassifier(), 
         }               


grids = {'rfc':{
               'classifier__n_estimators': [100, 300, 700],     
               'classifier__max_depth': [2,4, 6],         
               'classifier__max_features': [2,4,6],  
               },
         'log':{'classifier__penalty': ['none','l2'], 
                'classifier__C': [0.001,0.01,0.1,1,10,100,1000]},     
         }   

''' 
Read data
'''
with open(f"{PATH}/data/data_dtypes.json", "r") as f:
    dtypes = json.load(f)
data = pd.read_csv(f"{dataPath}/{dataset}", index_col=0, dtype=dtypes)
tmp = data.select_dtypes(include=["float32"]).columns 
data[tmp] = data[tmp].astype(pd.Int64Dtype())

''' 
Split
'''
X = data.drop(target, axis=1)
y = data[target]

#### FOR DEVELOPMENT PURPOSES: smaller dataset
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
model = 'log'
dic_summary = dict()
for i in range(100):
    ''' 
    Assemble pipeline
    '''
    pipe = Pipeline([("selector", pipe_supervisedSelector(sel_variables)),
                        ("imputation", preprocessor),
                        ("classifier", models[model])])

    result = classify_dcv(X, y, pipe, 
                          hp_grid=grids[model], 
                          randomState_cvOuter=13,
                          n_split_outer=5)
    dic_summary[i] = result


with open(f'{resultsPath}/seedStability_{model}.pickle', 'wb') as f:
    pickle.dump(dic_summary, f, protocol=pickle.HIGHEST_PROTOCOL)

