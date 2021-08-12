# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# reseach question: i want to predict the current BGL, every 5 min, without using any past bgl informations.
# 
# variables: only the main ones 'glucose','basal', 'CHO', 'insulin'
# %% [markdown]
# 

# %%
#loading imports ---------------
import GlucoNet_Loading
import numpy as np
import pandas as pd
import os
#processing imports ------------
from Proc_func import col_to_check, checkCarb, dummyCarbs, create_samples_V2, extract_data, get_valid_df, col_to_check 
from datetime import timedelta, datetime
#modelling imports --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score,max_error,mean_absolute_error,mean_squared_error,r2_score

# %% [markdown]
# Loading Data.
# In the cell below are created 4 dictionaries. Each key value pair is constituted by 
# key = id number of patient 
# value = associated pandas dataframe
# There are 12 patients in total.
# The 2 dictionaries all_df_train_stage1 and all_df_test_stage1 contains 6 training and 6 testing dataframes respectively for six patients.
# The 2 dictionaries all_df_train_stage2 and all_df_test_stage2 contains 6 training and 6 testing dataframes respectively for the ramaining six patients.
# 
# path = path to the folders containing the data to load. The loading process takes care of the correct transformation from xml to pandas df.

# %%

path = str(os.getcwd()) #+ '/datasets/' ATTENTION: currently the datasets folder must be in the samen path of the GlucoNet_Loading.py file
print(path)
all_df_train_stage1 = GlucoNet_Loading.parse_directory(path,'OhioT1DM-training', sys = 'mac')
all_df_test_stage1 = GlucoNet_Loading.parse_directory(path,'OhioT1DM-testing', sys = 'mac')
all_df_train_stage2 = GlucoNet_Loading.parse_directory(path,'OhioT1DM-2-training', sys = 'mac')
all_df_test_stage2 = GlucoNet_Loading.parse_directory(path,'OhioT1DM-2-testing', sys = 'mac')

# %% [markdown]
# Variable validation.
# 

# %%
data = all_df_train_stage1.get('559') #pid number must be string
data.columns


# %%
var_used_4_current_hyp = ["datetime",'glucose','basal',
       'CHO', 'insulin']
pid_to_remove1, pid_to_remove2 = col_to_check(var_used_4_current_hyp, dict1 = all_df_train_stage1, dict2 = all_df_test_stage1, dict3 = all_df_train_stage2, dict4 = all_df_test_stage2)
valid_pid_stage1, valid_pid_stage2 = get_valid_df(pid_to_remove1 = pid_to_remove1, pid_to_remove2 = pid_to_remove2)


# %%
data = data[var_used_4_current_hyp]
data


# %%
def resample(data, freq):
        """
        :param data: dataframe
        :param freq: sampling frequency
        :return: resampled data between the the first day at 00:00:00 and the last day at 23:60-freq:00 at freq sample frequency
        """
        start = data.datetime.iloc[0].strftime('%Y-%m-%d') + " 00:00:00"
        end = datetime.strptime(data.datetime.iloc[-1].strftime('%Y-%m-%d'), "%Y-%m-%d") + timedelta(days=1) - timedelta(
            minutes=freq)
        index = pd.period_range(start=start,
                                end=end,
                                freq=str(freq) + 'min').to_timestamp()
        data = data.resample(str(freq) + 'min', on="datetime").agg({'glucose': np.mean,'basal': np.sum, 'CHO': np.sum,'insulin': np.sum})
        data = data.reindex(index=index)
        data = data.reset_index()
        data = data.rename(columns={"index": "datetime"})
        return data
    
data_resampled = resample(data, 5)


# %%
def fill_na(df):
    df = df.copy(deep=True)
    return df

data_filled = fill_na(data_resampled)


# %%
def data_interpolation(df,method,order,limit):
    """
    limit value must be present in order to make all value of glucose of positive sign
    
    """
    df = df.copy(deep=True)
    df["glucose"].interpolate(method = "polynomial", order = 1, inplace = True, limit = 4)
    
    return df

data_iterpolated = data_interpolation(data_filled, method = "polynomial", order = 1, limit = 4)

# %% [markdown]
# Feature Engineering. 
# This is a substantial part. As an example, i'll create a new feature using a function, mealZone.
# mealZone, and eventually all the other feature eng. steps, must be called by the feature_eng function, that enables to apply the same operations to all datasets automatically. 

# %%

def mealZone2(df, before, after ):
    """
    create a new column mealZone with 1 if the observations falls 50 min before or 30 min after a meal(assuming that the resampling is every 5 minutes). 
    this numbers can be generalized for a mor flezible function:
    in the np.linspace line, i-n(8 in this case) indicate the periods before a meal; i+q(6 in this case) indicate the number of periods after a meal
    it is interesting to try n=0 to explicit the fact that for a window after a meal the glucose is being processed
    df = pandas dataframe object
    before = int = how many periods before cho timestamp to consider mealzone
    after = int = how many periods after cho timestamp to consider mealzone
    """
    df = df.copy(deep=True)
    mealZone = dummyCarbs(df).values
    mealIndex = np.nonzero(mealZone)[0]
    extendedMealIndex = []
    for i in mealIndex:
        to_append = np.linspace(i-before,i+after,after+before+1,dtype = int)
        extendedMealIndex.append(to_append)
    okExtendedIndex = []
    for sublist in extendedMealIndex:
        for element in sublist:
            okExtendedIndex.append(element)
    mealZone[okExtendedIndex] = 1
    df["mealZone" + str(before) + '-' +str(after)] = mealZone
    
    return df



def feature_eng(df, mealzone = False):
    df = df.copy(deep=True) 
    df = mealZone2(df, 8, 8)
    df = mealZone2(df, 0, 16)
    df = mealZone2(df, -8, 24)# TODO: fix bug 

    return df

data_feature_added = feature_eng(data_iterpolated, mealzone = True)

# %% [markdown]
# Additional manipulation can be tested here and added in the processing function. One additional manipulation that is nearly always present is selecting a subset of the variables or the transformation to categorical y.

# %%
def additional_manipulation(df):
    df = df.copy(deep=True)    
    df['basal'].replace(to_replace=0, method='ffill', inplace = True)  
    return df
data_manipulated = additional_manipulation(data_feature_added)
data_manipulated.columns
# %% [markdown]
# Sample creation.
# whith the function create_samples_V2, final training samples are created using a window approach.
# Since the function is always the same, it is not reported here but imported
# 
# TODO: insert link detailing the window approach

# %%

data_sampled = create_samples_V2(data_manipulated,number_lags = 20,colonne_da_laggare=['basal', 'CHO','insulin', 'mealZone8-8','mealZone0-16', 'mealZone-8-24'],colonna_Y='glucose',pred_horizon=0)
data_sampled.dropna(inplace = True)
data_sampled.drop('glucose_t', axis = 1, inplace = True)


# %% [markdown]
# This functions simply splits the data in X and y. It works for training and testing data as well, in spite of the name.

# %%
x, y = extract_data(data_sampled, 0)
x

# %% [markdown]
# Since the sample creation using the window approach generates new features, in the "final_x_manipulations" step are aggregated all the necessary operations that results in the final structure for the features's dataset ( the X_train/test structure)

# %%
def final_x_manipulation(df):
     pass
     return df
x = final_x_manipulation(x)
x

# %% [markdown]
# Processing function: 
# here are reported all the functions detailed before in order to apply the same process to all training and test data.

# %%
def processing(data, vars):
    """
    vars = list of vars
    """
    data = data[vars]
    data_resampled = resample(data, 5)
    data_filled = fill_na(data_resampled)
    data_iterpolated = data_interpolation(data_filled, method = "polynomial", order = 1, limit = 4)
    data_feature_added = feature_eng(data_iterpolated, mealzone = True)
    data_manipulated = additional_manipulation(data_feature_added)
    
    data_sampled = create_samples_V2(data_manipulated,number_lags = 20,colonne_da_laggare=['basal', 'CHO','insulin', 'mealZone8-8','mealZone0-16', 'mealZone-8-24'],colonna_Y='glucose',pred_horizon=0)
    data_sampled.drop('glucose_t', axis = 1, inplace = True)
    data_sampled.dropna(inplace = True)

    x, y = extract_data(data_sampled, 0)
    x = final_x_manipulation(x)
    

    return x, y

#if it is printed "True True" the processing() function works as intended
xp, yp = processing(data, var_used_4_current_hyp)
print(xp.equals(x),yp.equals(y))

# %% [markdown]
# Modelling.
# %% [markdown]
# accuracy measure is a function that has to take as arguments ytest and ypred and has to return a dataframe. 
# This dataframe must have the different accuracy measures in the columns and a single row containing the results for each measure

# %%
def accuracy_measure(ytest, predictions): #TODO: add cod patient to index name
    columns_names = ['evs','me','mae','rmse','r2']
    metrics_values = [explained_variance_score(ytest, predictions),
    max_error(ytest, predictions),
    mean_absolute_error(ytest, predictions),
    mean_squared_error(ytest, predictions, squared = False),
    r2_score(ytest, predictions)]

    acc_measure_df = pd.DataFrame(columns = columns_names)
    acc_measure_df.loc[1] = metrics_values #maybe i can parametrize the loc value?
    return acc_measure_df

# %% [markdown]
# Find a model: here is the space for experimenting and finding the best model to then pass into the modelling function

# %%

#scikitlearn 0.23.2 is needed - i also have to install pycaret and than all other packages in the enviroment
#exp_reg001 = setup(data = caret_data, target = 'target',fold_shuffle=True, session_id=2, imputation_type='iterative')
#best = compare_models(exclude = ['ransac'])


# %%


# %% [markdown]
# in the modelling function is specified the model and all the steps that generate the final ypred values. 
# The inputs should be the patient's id and xtrain, xtest , ytrain, ytest.
# Are returned two objects:
# res_y_ypred which is a dataframe containing all ytest and ypred values, used in later operations
# acc_measure_df which is the dataframe containing the results. again, one column for every measure and one row containing all the values.

# %%
def modelling (xtrain, xtest , ytrain, ytest, cod_patient):#attenzione all'ordine degli argomenti 
    model = LinearRegression()
    model = model.fit(xtrain,ytrain)
    
    predictions = model.predict(xtest)
    
    
    acc_measure_df = accuracy_measure(ytest, predictions) # TODO: add index name as pid
    res_y_ypred = pd.DataFrame({'ytest':ytest, 'pred':predictions})
    
    return res_y_ypred, acc_measure_df


# %%

data = all_df_train_stage1.get('559')
xtrain, ytrain = processing(data,  var_used_4_current_hyp)

data = all_df_test_stage1.get('559')
xtest, ytest = processing(data, var_used_4_current_hyp)

res_y_ypred, acc_measure_df = modelling( xtrain, xtest, ytrain, ytest, '559')

# %% [markdown]
# Getting results.
# it is possible in the cells above to test the procedure with 1 patiece. Below are presented the funtion for automatically esperimenting on all patients.
# 
# With get_single_results_stage1/2(pid) is possible to get results fast for a single pid specified directly in the funtion.
# 
# Instead of running get_single_results_stage1/2(pid), it is possible to run recursive_get_single_result() just one time, and get all the results.
# 
# The function recursive_conglobate_get_single_result is used to see how well the models generalise to unseen patients. on the total of n valid pids, the training is done on the training and testing data of n-1 patients. Then it can be decided to include or not in the big training dataset the remaining training dataset of the patient to test. This function cycles trough all n patience.

# %%
def get_single_results_stage1(pid, valid_vars = var_used_4_current_hyp):
    pid = str(pid)
    data = all_df_train_stage1.get(pid)
    xtrain, ytrain = processing(data, valid_vars)
    data = all_df_test_stage1.get(pid)
    xtest, ytest = processing(data, valid_vars)
    res_y_ypred, acc_measure_df = modelling(xtrain, xtest, ytrain, ytest, pid)
    return res_y_ypred, acc_measure_df

def get_single_results_stage2(pid, valid_vars = var_used_4_current_hyp):
    pid = str(pid)
    data = all_df_train_stage2.get(pid)
    xtrain, ytrain = processing(data, valid_vars)
    data = all_df_test_stage2.get(pid)
    xtest, ytest = processing(data, valid_vars)
    res_y_ypred, acc_measure_df = modelling(xtrain, xtest, ytrain, ytest, pid )
    return res_y_ypred, acc_measure_df


# %%
res_y_ypred, acc_measure_df = get_single_results_stage1(pid = '559' , valid_vars = var_used_4_current_hyp)
acc_measure_df


# %%
def recursive_get_single_result(valid_pid_stage1 = valid_pid_stage1, valid_pid_stage2 = valid_pid_stage2, valid_vars = var_used_4_current_hyp):
    """
    get separate result from each end every patient. 
    for each pid, the training is done on the relative train set 
    and the testing on the test set
    """
    res_y_ypred_tot = pd.DataFrame(columns = ['ytest', 'pred'])
    acc_measure_df_tot = pd.DataFrame()
    for i in valid_pid_stage1:
        res_y_ypred, acc_measure_df = get_single_results_stage1(pid = str(i), valid_vars = valid_vars)
        res_y_ypred_tot = res_y_ypred_tot.append(res_y_ypred)
        acc_measure_df_tot = pd.concat([acc_measure_df_tot, acc_measure_df], axis = 0)
    for i in valid_pid_stage2: 
        res_y_ypred, acc_measure_df = get_single_results_stage2(pid = str(i), valid_vars = valid_vars)
        res_y_ypred_tot = res_y_ypred_tot.append(res_y_ypred)
        acc_measure_df_tot = pd.concat([acc_measure_df_tot, acc_measure_df], axis = 0)
    acc_measure_df_tot.index = valid_pid_stage1 + valid_pid_stage2
    return res_y_ypred_tot, acc_measure_df_tot 


# %%
res_y_ypred_tot, acc_measure_df_tot =  recursive_get_single_result(valid_pid_stage1 = valid_pid_stage1, valid_pid_stage2 = valid_pid_stage2, valid_vars = var_used_4_current_hyp)
acc_measure_df_tot


# %%
def recursive_conglobate_get_single_result(include_in_tr = True, valid_pid_stage1 = valid_pid_stage1, valid_pid_stage2 = valid_pid_stage2, valid_vars = var_used_4_current_hyp): #TODO: fix warning
    """
    for pid n, train the algorithm on all training and test data of the other n - 1.
    if include in training = True, the training data is n is used for training, otherwise
    all train and test data are used for testing

    result are given separately for each and every pid
    """
    res_y_ypred_tot = pd.DataFrame(columns = ['ytest', 'pred'])
    acc_measure_df_tot = pd.DataFrame()
    sogg = valid_pid_stage1 + valid_pid_stage2
    for num in sogg:
        group = [x for x in sogg if x != str(num)]
        # num = left out subject
        xtr = pd.DataFrame()
        ytr = pd.Series()
        for i in group:
            if i in valid_pid_stage1:
                data = all_df_train_stage1.get(i)
                xtrain, ytrain = processing(data , valid_vars)
                xtr = xtr.append(xtrain)
                ytr = ytr.append(ytrain, ignore_index=True)
                data = all_df_test_stage1.get(i)
                xtrain, ytrain = processing(data , valid_vars)
                xtr = xtr.append(xtrain)
                ytr = ytr.append(ytrain, ignore_index=True)
            else:
                data = all_df_train_stage2.get(i)
                xtrain, ytrain = processing(data , valid_vars)
                xtr = xtr.append(xtrain)
                ytr = ytr.append(ytrain, ignore_index=True)
                data = all_df_test_stage2.get(i)
                xtrain, ytrain = processing(data , valid_vars)
                xtr = xtr.append(xtrain)
                ytr = ytr.append(ytrain, ignore_index=True)
        if include_in_tr == False:
            if num in valid_pid_stage1:
                xte = pd.DataFrame()
                yte = pd.Series()
                data = all_df_train_stage1.get(num)
                xtrain, ytrain = processing(data , valid_vars)
                xte = xte.append(xtrain)
                yte = yte.append(ytrain, ignore_index=True)
                data = all_df_test_stage1.get(num)
                xtrain, ytrain = processing(data , valid_vars)
                xte = xte.append(xtrain)
                yte = yte.append(ytrain, ignore_index=True)
            else:
                xte = pd.DataFrame()
                yte = pd.Series()
                data = all_df_train_stage2.get(num)
                xtrain, ytrain = processing(data , valid_vars)
                xte = xte.append(xtrain)
                yte = yte.append(ytrain, ignore_index=True)
                data = all_df_test_stage2.get(num)
                xtrain, ytrain = processing(data , valid_vars)
                xte = xte.append(xtrain)
                yte = yte.append(ytrain, ignore_index=True)
        else:
            if num in valid_pid_stage1:
                data = all_df_train_stage1.get(num)
                xtrain, ytrain = processing(data , valid_vars)
                xtr = xtr.append(xtrain)
                ytr = ytr.append(ytrain, ignore_index=True)
                data = all_df_test_stage1.get(num)
                xte, yte = processing(data , valid_vars)
            else:
                data = all_df_train_stage2.get(num)
                xtrain, ytrain = processing(data , valid_vars)
                xtr = xtr.append(xtrain)
                ytr = ytr.append(ytrain, ignore_index=True)
                data = all_df_test_stage2.get(num)
                xte, yte = processing(data , valid_vars)
        
        res_y_ypred, acc_measure_df = modelling(xtr, xte, ytr, yte, str(num))
        res_y_ypred_tot = res_y_ypred_tot.append(res_y_ypred)
        acc_measure_df_tot = pd.concat([acc_measure_df_tot, acc_measure_df], axis = 0)
    #acc_measure_df_tot.columns = sogg
    acc_measure_df_tot.index = sogg
    #for some reason 0 are imputed instead of nans, correction:
    #acc_measure_df_tot = acc_measure_df_tot.fillna(0)
    return res_y_ypred_tot, acc_measure_df_tot


# %%
res_y_ypred_tot2, acc_measure_df_tot2 = recursive_conglobate_get_single_result(include_in_tr = False, valid_pid_stage1 = valid_pid_stage1, valid_pid_stage2 = valid_pid_stage2)


# %%
acc_measure_df_tot2


