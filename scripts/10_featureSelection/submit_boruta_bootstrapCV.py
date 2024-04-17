
'''
Feature selection using BorutaPy

(c) Sonja Katz, 2024
'''

PATH = "/home/WUR/katz001/PROJECTS/permit-nsti-gas"

from boruta import BorutaPy
import os
import numpy as np
import json
import pandas as pd
import sys 
sys.path.append(f"{PATH}/scripts")

from sklearn.ensemble import RandomForestClassifier
from func import pipe_imputation_scaling, sample_w_replacement


def run_iterativeBoruta(X, y, cols, perc=100, n_iter=100, max_iter=100):

    dict_boruta = {}

    for i in range(n_iter):
        print(f"Round {i+1} of {n_iter}")

        ''' 
        Setup and run Boruta
        '''
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=None, perc=perc, max_iter=max_iter)
        feat_selector.fit(X, y)

        ''' 
        Get selected variables and save in dict
        '''
        selVars = np.array(cols)[feat_selector.support_]
        for var in selVars: 
            if var in dict_boruta.keys():
                dict_boruta[var] += 1
            else: 
                dict_boruta[var] = 1

    ### Normalise regarding number of iterations
    dict_boruta.update((x, y/n_iter) for x, y in dict_boruta.items())
    
    return dict_boruta


def get_input():
    try:
        dataset = sys.argv[1]
    except IndexError:
        print("ERROR\tPlease enter a valid dataset name (ENTRY, PRESURGERY,POSTSURGERY, BL)")
        sys.exit()
    return dataset


dataset = get_input()
print(dataset)
target = "Conclusion_micro"
PATH_out = f"{PATH}/results/20_featureSelection/{dataset}/bootstrap"               
os.makedirs(PATH_out, exist_ok=True)

### LOAD DATA ###
data0 = pd.read_csv(f"{PATH}/results/10_preprocessed/{dataset}_{target}_preprocessed.csv", low_memory=False)     

### Remove biasing/unnecessary labels ###
removeLabels = ["PATIENT_ID"]
data = data0.drop(removeLabels, axis=1)

### Split
X = data.drop(target, axis=1)
y = data[target]


### Prepare Preprocessing ###
### get columns to apply transformation to ###
num_columns = X.select_dtypes(include=["float64"]).columns
bin_columns = X.select_dtypes(include=["int64"]).columns
cat_columns = X.select_dtypes(include=["object"]).columns
preprocessor = pipe_imputation_scaling(num_columns, bin_columns, cat_columns).fit(X)
columnOrderAfterPreprocessing = [ele[5:] for ele in preprocessor.get_feature_names_out()]


for perc in [100]:

    for i in range(1,50):  

        outname_json=f"{i}__{target}_iterativeBoruta.json"

        ''' Bootstrap '''
        train_idx, test_idx = sample_w_replacement(X, 
                                                   n_size=np.ceil(X.shape[0]*.8).astype("int"),
                                                   stratify=y,
                                                   random_state=None)   
              
        X_train = X.iloc[train_idx,:]
        y_train = y[train_idx]

        ''' Impute '''
        X_ = preprocessor.fit_transform(X_train)
        y_ = y_train.values.ravel()
        print(X_train.shape)
        
        ''' Iterative Boruta'''
        n_iter = 30
        dict_iterBoruta = run_iterativeBoruta(X=X_,
                                            y=y_, 
                                            cols=columnOrderAfterPreprocessing, 
                                            perc=perc,
                                            n_iter=n_iter,
                                            max_iter=100)

        with open(f"{PATH_out}/{outname_json}", "w") as f: json.dump(dict_iterBoruta, f, indent=4)
