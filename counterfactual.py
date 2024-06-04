import dice_ml
from dice_ml.utils import helpers
from data_utils.preparation import clip, find_ranges, load_dataset, remove_missing_values # helper functions
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from IPython.display import display, Image
import pickle

def main():
    # load dataset
    model_path = 'random_forest_pd.pkl'
    NGPD=1
    METHOD = 'genetic' # 'random' or 'genetic'

    dataframe = load_dataset('final_data/dataset_ranges4.csv', True)
    dataframe = dataframe[['Patient_ID','CurrentState','StartDate','EndDate','hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD','FutureState','LastState']]

    dataframe = dataframe[dataframe['Sex']!=-1]
    pat = dataframe[dataframe['Age']<=18]['Patient_ID'].unique()
    dataframe = dataframe[~dataframe['Patient_ID'].isin(pat)]

    dataframe = clip(dataframe)
    RANGES, MIN, MAX = find_ranges(dataframe)

    '''multilabel classification without steady state'''
    #dataframe = dataframe[~((dataframe['CurrentState'] == 0) & (dataframe['FutureState'] == 0)) & ~((dataframe['CurrentState'] == 1) & (dataframe['FutureState'] == 1))]
    '''binary classification with current population ==NG or ==PD'''
    dataframe = dataframe[(dataframe['CurrentState']==NGPD)&(dataframe['FutureState']!=NGPD)]
    '''multilabel with current population==NG or ==PD'''
    #dataframe = dataframe[(dataframe['CurrentState']==1)]

    # preprocess dataset
    def manualMinMax(df):
        for column in df.columns:
            max = df[column].max()
            min = df[column].min()
            df[column] = (df[column]-min)/(max-min)
        return df
    
    def inverse_minmax_scaling(df, min_values, max_values):
        original_df = df.copy()
        for column in df.columns:
            mask = ~df[column].isna()
            original_df.loc[mask, column] = df.loc[mask, column] * (max_values[column] - min_values[column]) + min_values[column]
        return original_df

    dataframe[['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age']] = manualMinMax(dataframe[['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age']])
    dataframe=dataframe.drop(columns=['tot_cholesterol'])#,'weight','height'])
    _, dataset = remove_missing_values(dataframe, True)
    dataset=dataset.drop(columns=['weight','height'])

    print(len(dataset[(dataset['CurrentState'] == 0) & (dataset['FutureState'] == 0)]))
    print(len(dataset[(dataset['CurrentState'] == 0) & (dataset['FutureState'] == 1)]))
    print(len(dataset[(dataset['CurrentState'] == 0) & (dataset['FutureState'] == 2)]))
    print(len(dataset[(dataset['CurrentState'] == 1) & (dataset['FutureState'] == 0)]))
    print(len(dataset[(dataset['CurrentState'] == 1) & (dataset['FutureState'] == 1)]))
    print(len(dataset[(dataset['CurrentState'] == 1) & (dataset['FutureState'] == 2)]))

    # split dataset into training and testing
    unique_patients = dataset['Patient_ID'].unique()
    patient_train, patient_test = train_test_split(unique_patients, test_size=0.2, random_state=313)
    train_data = dataset[dataset['Patient_ID'].isin(patient_train)]
    test_data = dataset[dataset['Patient_ID'].isin(patient_test)]
    X_train = train_data.drop(columns=['Patient_ID','CurrentState','StartDate','EndDate','LastState'])
    y_train = train_data['FutureState']
    X_test = test_data.drop(columns=['Patient_ID','CurrentState','StartDate','EndDate','LastState'])
    y_test = test_data['FutureState']

    if NGPD==1:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        y_train = y_train.replace(2,1)
        y_test = y_test.replace(2,1)
    elif NGPD==0:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        y_train = y_train.replace(1,0)
        y_train = y_train.replace(2,1)
        y_test = y_test.replace(1,0)
        y_test = y_test.replace(2,1)

    print('Records in train set:', len(y_train))
    print('Records with NG in train set:', len([x for x in y_train if x==0]))
    print('Records with PD in train set:', len([x for x in y_train if x==1]))
    print('Records with D in train set:', len([x for x in y_train if x==2]))
    print('Records in test set:', len(y_test))
    print('Records with NG in test set:', len([x for x in y_test if x==0]))
    print('Records with PD in test set:', len([x for x in y_test if x==1]))
    print('Records with D in test set:', len([x for x in y_test if x==2]))

    print(X_train.shape[0]/(train_data.shape[0]+test_data.shape[0]), X_test.shape[0]/(train_data.shape[0]+test_data.shape[0]))

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(X_test.drop('FutureState', axis=1))
    mask = (predictions == y_test)

    # Dataset for training an ML model
    d = dice_ml.Data(dataframe=X_train,
                    continuous_features=['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age'],
                    enable_categorical=True,
                    outcome_name='FutureState')

    # Pre-trained ML model
    m = dice_ml.Model(model_path=model_path,
                    backend='sklearn', model_type='classifier')
    
    # DiCE explanation instance
    exp = dice_ml.Dice(d,m, method=METHOD)
    X_test_sample = X_test[(mask) & (X_test['FutureState']==2)][:]
    query_instance = X_test_sample.drop('FutureState', axis=1)

    
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=10, desired_class=0,
                                    features_to_vary=['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi'],
                                    permitted_range=RANGES)
    # Visualize counterfactual explanation
    print(dice_exp.visualize_as_dataframe(show_only_changes=True))

    # Save generated counterfactual examples to disk
    counterfactuals_complete = pd.DataFrame()
    counterfactuals_view = pd.DataFrame()

    counter_availability=0
    counter_n_cf=0
    for i in range (len(query_instance)):
        if (dice_exp.cf_examples_list[i].final_cfs_df is None):
            row_df = pd.DataFrame(np.nan, index=[i],columns=dataset.columns)
            counter_availability += 1
        else:
            row_df = dice_exp.cf_examples_list[i].final_cfs_df
            counter_n_cf = counter_n_cf + len(row_df)
            real_row = pd.DataFrame(query_instance.iloc[i]).T  # Extract first row and convert to DataFrame
            real_row['FutureState'] = X_test_sample['FutureState'].iloc[i]
            counterfactuals_complete = pd.concat([counterfactuals_complete, real_row], ignore_index=True)
            counterfactuals_complete = pd.concat([counterfactuals_complete, row_df], ignore_index=True)
            for column in row_df.columns:
                alpha = real_row[column].iloc[0]
                row_df[column] = np.where(row_df[column].round(5) == alpha.round(5), np.nan, row_df[column])
            counterfactuals_view = pd.concat([counterfactuals_view, real_row], ignore_index=True)
            counterfactuals_view = pd.concat([counterfactuals_view, row_df], ignore_index=True)


    print('availability = ' + str(1-(counter_availability/len(query_instance))))
    print('mean n of cf per instance = ' + str(counter_n_cf/len(query_instance)))

    counterfactuals_view = counterfactuals_view.round(2)
    counterfactuals_view.to_csv('cf_exp_'+METHOD+'_view_scaled.csv')
    counterfactuals_view[['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age']] = inverse_minmax_scaling(counterfactuals_view[['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age']], MIN, MAX)
    counterfactuals_view = counterfactuals_view.round(2)
    counterfactuals_view.to_csv('cf_exp_'+METHOD+'_view.csv')

    counterfactuals_complete = counterfactuals_complete.round(2)
    counterfactuals_complete.to_csv('cf_exp_'+METHOD+'_complete_scaled.csv')
    counterfactuals_complete[['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age']] = inverse_minmax_scaling(counterfactuals_complete[['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age']], MIN, MAX)
    counterfactuals_complete = counterfactuals_complete.round(2)
    counterfactuals_complete.to_csv('cf_exp_'+METHOD+'_complete.csv')
    
    

if __name__=='__main__':
    main()
