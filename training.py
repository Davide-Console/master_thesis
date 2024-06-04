from training_utils.models_training import *
from sklearn.model_selection import train_test_split
from data_utils.preparation import *
import random

def main():
    random.seed(0)
    np.random.seed(0)

    target = 'FutureState'
    NGPD=1

    # load dataset
    dataframe = load_dataset('final_data/dataset_ranges4.csv', True)
    
    dataframe = dataframe[['Patient_ID','CurrentState','StartDate','EndDate','hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD','FutureState','LastState']]
    
    dataframe = dataframe[dataframe['Sex']!=-1]
    pat = dataframe[dataframe['Age']<=18]['Patient_ID'].unique()
    dataframe = dataframe[~dataframe['Patient_ID'].isin(pat)]
    
    dataframe = clip(dataframe)

    '''multilabel classification without steady state'''
    #dataframe = dataframe[~((dataframe['CurrentState'] == 0) & (dataframe[target] == 0)) & ~((dataframe['CurrentState'] == 1) & (dataframe[target] == 1))]
    '''binary classification with current population ==NG or ==PD'''
    dataframe = dataframe[(dataframe['CurrentState']==NGPD)&(dataframe[target]!=NGPD)]
    '''multilabel with current population==NG or ==PD'''
    #dataframe = dataframe[(dataframe['CurrentState']==0)]

    # preprocess dataset
    def manualMinMax(df):
        for column in df.columns:
            max = df[column].max()
            min = df[column].min()
            df[column] = (df[column]-min)/(max-min)
        return df

    #dataframe[['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age']] = manualMinMax(dataframe[['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age']])
    dataframe=dataframe.drop(columns=['tot_cholesterol'])#,'weight','height'])
    _, dataset = remove_missing_values(dataframe, True)
    #dataframe = standardise(dataframe, ['hdl','ldl','fbs','hba1c','triglycerides','sbp','dbp','bmi','Age'])
    dataset=dataset.drop(columns=['weight','height'])
    print(len(dataset[(dataset['CurrentState'] == 0) & (dataset[target] == 0)]))
    print(len(dataset[(dataset['CurrentState'] == 0) & (dataset[target] == 1)]))
    print(len(dataset[(dataset['CurrentState'] == 0) & (dataset[target] == 2)]))
    print(len(dataset[(dataset['CurrentState'] == 1) & (dataset[target] == 0)]))
    print(len(dataset[(dataset['CurrentState'] == 1) & (dataset[target] == 1)]))
    print(len(dataset[(dataset['CurrentState'] == 1) & (dataset[target] == 2)]))

    # split dataset into training and testing
    unique_patients = dataset['Patient_ID'].unique()
    patient_train, patient_test = train_test_split(unique_patients, test_size=0.2, random_state=313)
    train_data = dataset[dataset['Patient_ID'].isin(patient_train)]
    test_data = dataset[dataset['Patient_ID'].isin(patient_test)]
    X_train = train_data.drop(columns=['Patient_ID','CurrentState','StartDate','EndDate','FutureState','LastState'])
    y_train = train_data[target]
    X_test = test_data.drop(columns=['Patient_ID','CurrentState','StartDate','EndDate','FutureState','LastState'])
    y_test = test_data[target]
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

    # train model
    #train_decision_tree(X_train, y_train, X_test, y_test, sampling='none', xai=True)
    #a=input('press enter to continue')
    #train_random_forest(X_train, y_train, X_test, y_test, sampling='none', save=True)
    #a=input('press enter to continue')
    #train_bagging(X_train, y_train, X_test, y_test, sampling='none', save=True)
    #a=input('press enter to continue')
    train_xgboost(X_train, y_train, X_test, y_test, sampling='none', xai=True)



if __name__ == '__main__':
    main()