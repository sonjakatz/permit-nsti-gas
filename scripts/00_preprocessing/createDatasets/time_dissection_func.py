import pandas as pd
import re
import numpy as np
import os

def get_events_prior_to(orig_data, time0, time_prior_event, sel_columns, hard_impute=False):

    '''
    Calculates the time difference between "time0" and "time_prior_event" (delta = time0 - time_prior_event)

    Generates a binary matrix (mask) 
        if: delta > 0 
                ==> event will be included in dataset (eg. surgery occured prior to ICU admission)
                
        if: delta < 0
                ==> event available but has not yet occured (include in dataset as 0 (former: -1))
                
        if: delta == NaN
                ==> event has not happened (include in dataset as 0)          
        
    -- sel_columns: Mask is applied to all columns in "sel_columns"
    
    -- hard_impute=True: all NAs will be replaced by 0 (to avoid that column gets dropped during data cleaning!)

    RETURN modified_dataset (copy of original dataset)

    '''

    data = orig_data.copy()

    # Get dates when samples were taken
    indx_dates = data.columns[data.columns.str.contains(time_prior_event, case = False)].to_list()
    indx_dates = [time0] + indx_dates

    # Comvert to datetime objects
    df_dates = data.loc[:,indx_dates].astype("datetime64[ns]")

    # Calculate difference between ICU admission and sample dates
    df_mask = pd.DataFrame()
    for i in range(1,df_dates.shape[1]):
        df_mask["preICU_sample_"+str(i)] = (df_dates.iloc[:,0] - df_dates.iloc[:,i]).dt.total_seconds()/3600

    ### NEW: hard-impute NAN to 0!
    # Apply mask to relevant columns:
    for col in sel_columns:
        i = [col+str(x) for x in range(1,df_dates.shape[1])]

        # Hard-impute to 0
        if hard_impute is True:
            data.loc[:,i] = data.loc[:,i].fillna(0)         ### Impute with 0
            #data.loc[:,i] = data.loc[:,i].fillna(data.mode(numeric_only=True).iloc[0])            ### Impute numerical with 0 - leave out strings
            

        # prepare mask
        df_mask.columns = i

        # if delta is < 0 --> include in dataset as 0 (former: -1) (event available but has not yet occured)
        ### pd.where: Replace values where the condition is False. ### --> this is why df_mask > 0
        data.loc[:,i] = data.loc[:,i].where(df_mask > 0, 0) #-1

        # replace value by 0 (former: -1) if timestamp of variable is NA (= surgery has never taken place)
        ### pd.where: Replace values where the condition is False. ###
        data.loc[:,i] = data.loc[:,i].where(df_mask.notnull(), 0) #-1

    # return [data, df_mask]
    return data
