#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:43:06 2021

@author: tommasobassignana
"""
#i need to import every module needed for the functions
import numpy as np
import pandas as pd

def extract_data(data, pred_horizon):
        """
        Extract the input variables (x), the time (t), and the objective (y) from the data samples.
        WARNING : need to be modified to include additional data, or override the function within the models
        :param data: df
        pred_horizon: integer, same used in create_samples_v2
        :return:
        """
        
        y = data["y_t+" + str(pred_horizon)]#ATT: le y sono una series, non un df!
        x = data.drop(["y_t+" + str(pred_horizon),'datetime_t'], axis=1)

        return x, y


def y_to_categorical(yseries_train, yseries_test):
    """
    function that transforms the rescaled y series in categorical values if the problem 
    is a classification task
    yseries_train = yseries_test = pandas series object
    """
    conditions = [
        (yseries_train == 0),#verified_hypo
        (yseries_train == 0.25),#couscin_hypo
        (yseries_train == 0.5),#normal
        (yseries_train == 0.75),#couscin_hyper
        (yseries_train == 1)#verified_hyper
        ]
    values = ['verified_hypo','couscin_hypo','normal','couscin_hyper','verified_hyper']
    yseries_train = np.select(conditions, values)

    conditions = [
        (yseries_test == 0),#verified_hypo
        (yseries_test == 0.25),#couscin_hypo
        (yseries_test == 0.5),#normal
        (yseries_test == 0.75),#couscin_hyper
        (yseries_test == 1)#verified_hyper
        ]
    values = ['verified_hypo','couscin_hypo','normal','couscin_hyper','verified_hyper']
    yseries_test = np.select(conditions, values)
    return yseries_train, yseries_test

def create_title(xtrainData, xtestData, model):
    col_list = str(list(xtrainData.columns))
    tr_rows = str(xtrainData.shape[0])
    tst_rows = str(xtrainData.shape[0])
    title = col_list + ';' + tr_rows + '-' + tst_rows + ';' + str(model)
    return title 

def create_samples_V2(df,number_lags,colonne_da_laggare,colonna_Y,pred_horizon):
    """
    function that takes a classic dataframes and create a new one with columns of lagged variables values for
    all columns in colonne_da_laggare. The number of lags are determined by number_lags variables. All the
    columns that are not lagged will be mantained with the new name column_t. colonna_y determines what is the target variables
    to predict. The numer of units of time*** in the future to shift the target variable is determined by the pred_horizon variable.
    ATT:tutte le colonne non laggate saranno portate nel nuovo df rinominate con _t
    colonne_da_laggare = list of str containing the names of the columns
    pred_horizon and number_lags are both expressed in units of time***
    colonna_Y = str
    ***the units of time are the frequency of the resampling of df. If df is resampled to 5 minutes, a pred_horizon = 6 means that 
    i'm trying to predict colonna_Y 30 minutes into the future. if the data is resampled to half-hour intervals, with a pred_horizon of 6,
    i'm trying to predict colonna_Y 3h into the future. 
    """
    new_df = pd.DataFrame()
    col_non_laggate = df.drop(colonne_da_laggare,axis=1).columns
    for feature in colonne_da_laggare:
        for lag in range(1, number_lags + 1):
            new_df[feature + '_t-' + str(lag)] = df[feature].shift(lag)
        new_df[feature+'_t'] = df[feature]
    new_df['y'+'_t+'+str(pred_horizon)]=df[colonna_Y].shift(-pred_horizon)
    for feature in col_non_laggate:
        new_df[feature + '_t'] = df[feature]
    #new_df.drop('datetime_t')perchè non va?
    return(new_df)

def to_cat_meal_type(data):
    """
    for the resempling operation meal type must be converted to categorical encoding. nans are substituted with 0.
    data = pandas dataset
    
    """
    data["meal_type"] = data.meal_type.fillna('0')
    data["meal_type"] = np.where((data.meal_type == 'Breakfast'),'1',data.meal_type)
    data["meal_type"] = np.where((data.meal_type == 'Snack'),'2',data.meal_type)
    data["meal_type"] = np.where((data.meal_type == 'Lunch'),'3',data.meal_type)
    data["meal_type"] = np.where((data.meal_type == 'Dinner'),'4',data.meal_type)
    data["meal_type"] = np.where((data.meal_type == 'HypoCorrection'),'5',data.meal_type)
    data["meal_type"] = data["meal_type"].astype(int)
    return data


def checkCarb(inpCarb, minCarb):
    """
    helper function for dummyCarbs
    """
    if(inpCarb>minCarb):
        return 1
    else:
        return 0

def dummyCarbs(df ):
    """ function that add a new column 'dummyCarbs' with 1 in the period that the quantity of carbs is recorded
    df = pandas dataframe object containing a CHO column
    return the column!
    """
    #apply fa passare come primo argomento alla funzione checkCarb i dati della colonna CHO che rappresenta il quantitativo di carbs assunti, mentre attraverso ad args si passa il secondo argomento 
    dummyCarbs = df["CHO"].apply(checkCarb, args=(1,))
    #df["dummyCarbs"] = dummyCarbs
    return dummyCarbs

def mealZone(df ):
    """
    create a new column mealZone with 1 if the observations falls 50 min before or 30 min after a meal(assuming that the resampling is every 5 minutes). 
    this numbers can be generalized for a mor flezible function:
    in the np.linspace line, i-n(8 in this case) indicate the periods before a meal; i+q(6 in this case) indicate the number of periods after a meal
    it is interesting to try n=0 to explicit the fact that for a window after a meal the glucose is being processed
    df = pandas dataframe object
    """
    mealZone = dummyCarbs(df).values
    mealIndex = np.nonzero(mealZone)[0]
    extendedMealIndex = []
    for i in mealIndex:
        to_append = np.linspace(i-8,i+6,6+8+1,dtype = int)
        extendedMealIndex.append(to_append)
    okExtendedIndex = []
    for sublist in extendedMealIndex:
        for element in sublist:
            okExtendedIndex.append(element)
    mealZone[okExtendedIndex] = 1
    df["mealZone"] = mealZone
    
    return df


def Y_cat(data_resampled, values):
    """creating Y variables - values might be adjusted
    data_resampled = pandas df object with a colum named glucose
    values = array representing the class number 
    """
    conditions = [
    (data_resampled['glucose'] <= 55),#verified_hypo
    (data_resampled['glucose'] > 55) & (data_resampled['glucose'] <= 80),#couscin_hypo
    (data_resampled['glucose'] > 80) & (data_resampled['glucose'] <= 170),#normal
    (data_resampled['glucose'] > 170) & (data_resampled['glucose'] <= 210),#couscin_hyper
    (data_resampled['glucose'] > 210)
    ]
    #values = [0,1,2,3,4]
    data_resampled['gluco_class'] = np.select(conditions, values)
    return data_resampled



def time_dummy(df, match_timestamp, new_col):
    """
    This function puts on a new_col 1 in every occurrence of the specified timestamp, 0 otherwise
    df = dataframe object, must have a datetime column but not a datetime index
    match_timestamp = str, specified timestamp in the right format es 05:00:00
    new_col = str, name of the new column to add in the df
    """

    dfp = df.set_index('datetime')
    dfp.index = pd.to_datetime(dfp.index)
    dfp[new_col] = np.where(dfp.index.strftime("%H:%M:%S") == match_timestamp, 1, 0)
    df[new_col] = dfp[new_col].values
    
    #forse sarebbe meglio farmi ritornare la colonna
    return df

def time_bool(df, match_timestamp, new_col):
    """
    This function puts True a new_col 1 in every occurrence of the specified timestamp, False otherwise
    df = dataframe object, must have a datetime column but not a datetime index
    match_timestamp = str, specified timestamp in the right format es 05:00:00
    new_col = str, name of the new column to add in the df
    """

    dfp = df.set_index('datetime')
    dfp.index = pd.to_datetime(dfp.index)
    dfp[new_col] = np.where(dfp.index.strftime("%H:%M:%S") == match_timestamp, True, False)
    df[new_col] = dfp[new_col].values
    
    #forse sarebbe meglio farmi ritornare la colonna
    return df


#last blood glucose measure for fasting istance
def get_past_BG(df, new_col, match_timestamp, glucose_col = 'glucose'):
#per ogni riga che individuo essere la amisurazione del mattino mi serve un valore di glucosio da associare come ultimo valore registrato, ad esempio quello delle 10 di sera del giorno precedente. 
#uso time_dummy per selezionarmi solo le righe delle 22 pm, la chiamo dummy22
#dove c'è 1 nella colonna dummy22 riporto nella colonna new_col il valore nella colonna glucose altrimenti imputo nan
#faccio un ffill dei valori di lastBGMeasureFast in modo che l'ultimo valore registrato sarà propagato per tutte le 24h successive e andrà a ricadere nella fasting istance 
#ci sarà sicuramente un modo più svelto e furbo con iloc ecc

#attenzione che se il valore di glucosio è nan viene propagato un nan
    """
    df = panda dataframe object
    new_col = str, name of the new column to add in the df containing the value that i want
    match_timestamp = str, specified timestamp in the right format es 05:00:00
    glucose_col = str, name of the column in db that contains all the BG values, default = glucose
    """
    dfp = df
    dfp = time_dummy(dfp, '22:00:00', 'dummy22')
    dfp = df.set_index('datetime')
    dfp.index = pd.to_datetime(dfp.index)
    dfp[new_col] = np.where(dfp['dummy22'] == 1, dfp[glucose_col], np.nan)
    dfp[new_col].fillna(method='ffill', inplace=True)
    
    return dfp.drop(["dummy22"], axis=1)





def col_to_check (cols_to_check, dict1, dict2, dict3, dict4):
    """
    cols_to_check = ["datetime",'glucose','CHO','q'] or similar
    list of columns to check for their presence in every test and train dataset 

    return pid (personal ids) number of unusable datasets
    """
    pid_to_remove1 = []
    pid_to_remove2 = []
    for num in ['559','563','570','575','588','591']:
        data = dict1.get(num)
        if set(cols_to_check).issubset(data.columns) == False:
            actual_diff = set(cols_to_check) - set(data.columns)  
            col_to_print_tr1 = actual_diff.intersection(set(cols_to_check))
            print('for TRAIN dataset1 ' + str(num) + ' the following columns are not present: ' + str(col_to_print_tr1))
            pid_to_remove1.append(num)
        data = dict2.get(num)
        if set(cols_to_check).issubset(data.columns) == False:
            actual_diff = set(cols_to_check) - set(data.columns)  
            col_to_print_te1 = actual_diff.intersection(set(cols_to_check))
            print('for TEST dataset1 ' + str(num) + ' the following columns are not present: ' + str(col_to_print_te1))
            pid_to_remove1.append(num)
    for num in ['540','544','552','567','584','596']:
        data = dict3.get(num)
        if set(cols_to_check).issubset(data.columns) == False:
            actual_diff = set(cols_to_check) - set(data.columns)  
            col_to_print_tr2 = actual_diff.intersection(set(cols_to_check))
            print('for TRAIN dataset2 ' + str(num) + ' the following columns are not present: ' + str(col_to_print_tr2))
            pid_to_remove2.append(num)
        data = dict4.get(num)
        if set(cols_to_check).issubset(data.columns) == False:
            actual_diff = set(cols_to_check) - set(data.columns)  
            col_to_print_te2 = actual_diff.intersection(set(cols_to_check))
            print('for TEST dataset2 ' + str(num) + ' the following columns are not present: ' + str(col_to_print_te2))
            pid_to_remove2.append(num)
    
    return pid_to_remove1, pid_to_remove2


def get_valid_df(pid_to_remove1, pid_to_remove2): 
    """
    """
    valid1 = set(['559','563','570','575','588','591']) - set(pid_to_remove1)
    valid2 = set(['540','544','552','567','584','596']) - set(pid_to_remove2)
    return list(valid1), list(valid2)



#da implementare
#creating hour and minute feature
#data_resampled["hour"] = data_resampled['datetime'].dt.hour
#data_resampled["minute"] = data_resampled['datetime'].dt.minute

#rolling_features
#rolling mean of the past hour pastHourRoll
#data_resampled['pastHourRoll'] = data_resampled['glucose'].rolling(window = 12, min_periods = 1).mean()
#rolling mean of the past hour pastDayRoll
#data_resampled['pastDayRoll'] = data_resampled['glucose'].rolling(window = 12*24, min_periods = 1).mean()

#trovare quanto dista una determinata osservazione rispetto al momento dell'aggregazione, es: voglio sapere quanto dista l'ultima misurazione del glucosio dal momento in cui creo la fasting istance 
#per una data colonna COL
#eventualmente posso creare una colonna bool solo per le righe per cui mi serve davvero calcolare questa differenza 
#creo una nuova colonna contenente il nome della colonna sopra e una desinenza NEW_COL
#creo un limite di periodi per la ricerca: es se i periodi sono di 5 min e ne imposto il limite a 24, eseguirò il controllo nelle 2 ore precedenti LIMIT
##per ogni riga del df
###calcolo l'indice i della riga I
###controllo se il valore all'indice i-1 è nan o valorizzato
###se è valorizzato calcolo la differenza di indice, quella è la differenza in periodi 
###se è nan procedo a controllare i-2
###continuo fino a quando non raggiungo il limite all'indietro oppure se trovo un valore interrompo e vado alla riga successiva
if __name__ == '__main__':
    print('main')