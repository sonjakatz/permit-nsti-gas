
'''
Performance comparison of multiple different ML models

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
from func_clf import classify_leave_one_out_cv


from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel

import warnings
warnings.filterwarnings('ignore')

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
resultsPath = f"{PATH}/results/30_modelComparison/{datasetTimepoint}/iterativeBoruta_{percentBoruta}perc/modelComparison"
os.makedirs(resultsPath, exist_ok=True)
dataPath = f"{PATH}/results/10_preprocessed/"
dataset = f"{datasetTimepoint}_{target}_preprocessed.csv"


models = {
          'gpc': GaussianProcessClassifier(max_iter_predict=5000),
          #'log': LogisticRegression(max_iter=5000),
          #'rfc': RandomForestClassifier(), 
          #'xgb': xgb.XGBClassifier(objective="binary:logistic")
         }               


grids = {'rfc':{
               'classifier__n_estimators': [100, 300, 700],     
               'classifier__max_depth': [2,4, 6],         
               'classifier__max_features': [2,4,6],  
               },
         'gpc':{'classifier__kernel':[1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]},
         'log':{'classifier__penalty': ['none','l2'], 
                'classifier__C': [0.001,0.01,0.1,1,10,100,1000]},     
         'xgb': {'classifier__n_estimators': [100,300,700],
                 'classifier__max_depth': [2,4, 6], 
                 'classifier__max_leaves': [0, 2,4]}
         }   

''' 
Read data
'''
data = pd.read_csv(f"{dataPath}/{dataset}", index_col=0)
X = data.drop(target, axis=1)
y = data[target]

##### FOR DEVELOPMENT PURPOSES: smaller dataset
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
try: 
      for model in models.keys():
            print(model)

            ''' 
            Assemble pipeline
            '''
            pipe = Pipeline([("selector", pipe_supervisedSelector(sel_variables)),
                            ("imputation", preprocessor),
                            ("classifier", models[model])])

            df_score = pd.DataFrame()    
            df_features = pd.DataFrame()
            df_importances = pd.DataFrame() 

            saveIndivdualPred = True
            clf = GridSearchCV(pipe, grids[model], scoring='balanced_accuracy', verbose=1, cv=5, n_jobs=-1) 
            result = classify_leave_one_out_cv(clf, 
                                                X, 
                                                y,
                                                model=model, 
                                                saveIndivdualPred=saveIndivdualPred)
            #print(result)
            ''' Prepare results '''

            df_score = pd.concat([df_score, result["df_results"]], ignore_index=True)       
            if saveIndivdualPred:
                  df_indPred = pd.DataFrame()
                  df_indPred = pd.concat([df_indPred, result["df_indPred"]], ignore_index=True)
            

            ''' Save to file '''
            df_score.to_csv((resultsPath+f"/prediction_modelComparison_{model}.csv"), index=False)          
            df_indPred.to_csv((resultsPath+f"/individualPredictions_modelComparison_{model}.csv"), index=False)   

except IOError as e: 
      if e.errno == errno.EPIPE:
          print(e)
          pass