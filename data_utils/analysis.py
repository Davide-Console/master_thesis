import matplotlib.pyplot as plt
from data_utils.preparation import *

def analyze_df(dataframe, flag):
    # %% Preparing the dataset
    dataframe = dataframe[['Patient_ID','CurrentState','StartDate','EndDate','hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD','FutureState']]
    dataframe.describe().to_csv('dataset_analysis/'+flag+'/dataset_description.csv')
    
    # count of transitions
    count_00 = dataframe.loc[(dataframe['CurrentState'] == 0) & (dataframe['FutureState'] == 0)].shape[0]
    count_01 = dataframe.loc[(dataframe['CurrentState'] == 0) & (dataframe['FutureState'] == 1)].shape[0]
    count_02 = dataframe.loc[(dataframe['CurrentState'] == 0) & (dataframe['FutureState'] == 2)].shape[0]
    count_10 = dataframe.loc[(dataframe['CurrentState'] == 1) & (dataframe['FutureState'] == 0)].shape[0]
    count_11 = dataframe.loc[(dataframe['CurrentState'] == 1) & (dataframe['FutureState'] == 1)].shape[0]
    count_12 = dataframe.loc[(dataframe['CurrentState'] == 1) & (dataframe['FutureState'] == 2)].shape[0]
    count_20 = dataframe.loc[(dataframe['CurrentState'] == 2) & (dataframe['FutureState'] == 0)].shape[0]
    count_21 = dataframe.loc[(dataframe['CurrentState'] == 2) & (dataframe['FutureState'] == 1)].shape[0]
    count_22 = dataframe.loc[(dataframe['CurrentState'] == 2) & (dataframe['FutureState'] == 2)].shape[0]
    counts = {
    '0-->0': count_00,
    '0-->1': count_01,
    '0-->2': count_02,
    '1-->0': count_10,
    '1-->1': count_11,
    '1-->2': count_12,
    '2-->0': count_20,
    '2-->1': count_21,
    '2-->2': count_22
    }
    counts_df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count'])
    counts_df.to_csv('dataset_analysis/'+flag+'/transitions.csv', index=True)

    boxplot_comparison(dataframe, ['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age'], 'CurrentState', flag, True)
    boxplot_comparison(dataframe, ['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age'], 'FutureState', flag, True)
    
    # Distributions
    #find_distributions(dataframe, ['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'CurrentState',flag)
    #find_distributions(dataframe, ['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'FutureState',flag)

    # correlation
    #correlation_matrix = dataframe[['CurrentState','hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD','FutureState']].corr(method='spearman')
    correlation_matrix = dataframe[['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD']].corr(method='spearman')

    print(correlation_matrix)
    correlation_matrix.to_csv('dataset_analysis/'+flag+'/correlation_matrix.csv')
    
