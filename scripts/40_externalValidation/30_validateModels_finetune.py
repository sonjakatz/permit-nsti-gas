
'''

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
import shap

from func_preprocess import pipe_imputation_scaling, pipe_supervisedSelector
from func_clf import externalValidate_finetune_boostrap_inclSHAP

import joblib

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
# datasetTimepoint = "PRESURGERY"
# n_bootstrap = 1

target = "Conclusion_micro"
percentBoruta = 100


''' 
Define paths
'''
folderFigures = f"{PATH}/figures/{datasetTimepoint}/50_externalValidation"
os.makedirs(folderFigures, exist_ok=True)
resultsPath = f"{PATH}/results/50_externalValidation/{datasetTimepoint}/finetune"
os.makedirs(resultsPath, exist_ok=True)

dataPath = f"{PATH}/results/10_preprocessed/validation"

modelPath = f"{PATH}/results/40_internalValidation/{datasetTimepoint}"

''' 
Variables
'''
with open(f"{PATH}/data/validation/variables_translation.json", "r") as f: varTranslation = json.load(f)
rkz_varTranslation = {v: k for k, v in varTranslation.items()}


''' 
Read validation data
'''
with open(f"{PATH}/data/validation/validation_dtypes.json", "r") as f:
    dtypes = json.load(f)

data = pd.read_csv(f"{dataPath}/{datasetTimepoint}_{target}_validationData.csv", index_col=None, dtype=dtypes)
tmp = data.select_dtypes(include=["float32"]).columns 
data[tmp] = data[tmp].astype(pd.Int64Dtype())

### translate varnames to PerMIT to fit models!
data.columns = [rkz_varTranslation[ele] for ele in data.columns]


''' 
Split
'''
X_val = data.drop(target, axis=1)
y_val = data[target].values


## FOR DEVELOPMENT PURPOSES: smaller dataset
# X_val = X_val.iloc[:70,:]
# y_val = y_val[:70]


''' 
Prepare imputation and scaling
'''
num_columns = X_val.select_dtypes(include=["float64"]).columns
bin_columns = X_val.select_dtypes(include=["int64"]).columns
cat_columns = X_val.select_dtypes(include=["object"]).columns
preprocessor = pipe_imputation_scaling(num_columns, bin_columns, cat_columns)  

### Impute test data set on its own to avoid information leakage between internal and external validation!
X_val_imp = pd.DataFrame(preprocessor.fit_transform(X_val))
X_val_imp.index = X_val.index


'''
Load fitted model 
'''

pipe_fit = joblib.load(f"{modelPath}/model_fitted_wholeDataset.sav")
clf_fit = pipe_fit["classifier"]


''' 
Run Pipeline
'''



for perc_finetune in np.arange(0.1, 1.1, .1):

    dic_summary = dict()
    dic_summary_shap = dict()
    dic_summary_predProba = dict()

    for i in range(n_bootstrap):

        dic_bootstrap_results, dic_shap_values, dic_proba = externalValidate_finetune_boostrap_inclSHAP(X_val_imp, 
                                                                                    y_val, 
                                                                                    clf_fit,
                                                                                    perc_samples_per_boostrap=perc_finetune)

        ### Save AUC and ROC for quick check
        dic_summary[i] = dic_bootstrap_results

        ### Save SHAP values
        for k,v in dic_shap_values.items():
            if k in dic_summary_shap.keys():
                dic_summary_shap[k].append(v)
            else:
                dic_summary_shap[k] = [v]

        ### Save individual predictions for further analyses
        dic_summary_predProba[i] = dic_proba

        
    with open(f'{resultsPath}/bootstrap_{round(perc_finetune,1)}_n{n_bootstrap}_qc.pickle', 'wb') as f:
        pickle.dump(dic_summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{resultsPath}/bootstrap_{round(perc_finetune,1)}_n{n_bootstrap}_shap.pickle', 'wb') as f:
        pickle.dump(dic_summary_shap, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{resultsPath}/bootstrap_{round(perc_finetune,1)}_n{n_bootstrap}_predProba.pickle', 'wb') as f:
        pickle.dump(dic_summary_predProba, f, protocol=pickle.HIGHEST_PROTOCOL)




