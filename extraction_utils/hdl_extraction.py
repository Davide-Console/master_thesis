import numpy as np
import pandas as pd

def test_apply(x):
    try:
        return float(x)
    except ValueError:
        return None

def extract():
    pd.set_option('display.max_rows', 5000)
    df=pd.read_csv('tables/all_labs.csv')
    print('Total number of records', df.shape)
    print('Corresponding number of patients:', df.groupby('Patient_ID').count().reset_index(drop=True).shape)

    df['Name_calc'].value_counts()

    #Extract HDL values when 'HDL' is present in exam name calc
    HDL_df_calc=df.loc[(df['Name_calc'].str.contains('HDL', case=True, regex=True , na=False)),:]
    HDL_df_calc_index= HDL_df_calc.index.to_list()
    print('')
    print('Records with HDL exam in Name calc',HDL_df_calc.shape)
    print('Corresponding number of patients:', HDL_df_calc.groupby('Patient_ID').count().reset_index(drop=True).shape)

    print('')
    #rimuovo records con info codificate x controllare i record rimanenti sulla base di name orig
    HDL_df_notCoded= df.loc[df.index.isin(HDL_df_calc_index) == False]
    #Extract HDL values when 'HDL' is present in exam name orig
    HDL_df_orig=HDL_df_notCoded.loc[(HDL_df_notCoded['Name_orig'].str.contains('|'.join(['HDL','high density lip']), case=True, regex=True , na=False)),:]
    print('')
    print('Records with HDL exam in Name Orig',HDL_df_orig.shape)
    print('Corresponding number of patients:', HDL_df_orig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    HDL_df_orig['Name_orig'].value_counts()

    #Extract HDL values when 'HDL' is present in exam name calc
    HDL_df_TestResultCalc=HDL_df_calc.dropna(subset=['TestResult_calc'], how='all')
    print('Records with HDL exam value in Test result calc',HDL_df_TestResultCalc.shape)
    print('Corresponding number of patients:', HDL_df_TestResultCalc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    #for the records without Test result calc we search for test result orig
    HDL_df_TestResultOrig= HDL_df_calc.loc[HDL_df_calc.index.isin(HDL_df_TestResultCalc.index.to_list()) == False]
    print(HDL_df_TestResultOrig.shape)
    #remove value orig with characters
    HDL_df_TestResultOrig=HDL_df_TestResultOrig.loc[(~HDL_df_TestResultOrig['TestResult_orig'].str.contains('|'.join(['Phone','/', 'see comment','below','PND','DNR','\>','<','-','Pending','deleted by lab','L','Z']), case=False, regex=True , na=False)) ,:]
    HDL_df_TestResultOrig['TestResult_orig'] = HDL_df_TestResultOrig['TestResult_orig'].apply(test_apply).dropna()

    HDL_df_TestResultOrig= HDL_df_TestResultOrig.loc[(HDL_df_TestResultOrig['TestResult_orig']<5) & (HDL_df_TestResultOrig['TestResult_orig']>0),:]
    print('Records with HDL exam value in Test orig calc',HDL_df_TestResultOrig.shape)
    print('Corresponding number of patients:', HDL_df_TestResultOrig.groupby('Patient_ID').count().reset_index(drop=True).shape)

    HDL_df_TestResultCalc=HDL_df_TestResultCalc.loc[:,['Lab_ID','Patient_ID','PerformedDate','TestResult_calc','DateCreated']].rename(columns={'TestResult_calc':'HDL [mmol/L]'})
    HDL_df_TestResultOrig=HDL_df_TestResultOrig.loc[:,['Lab_ID','Patient_ID','PerformedDate','TestResult_orig','DateCreated']].rename(columns={'TestResult_orig':'HDL [mmol/L]'})

    HDL_final=pd.concat([HDL_df_TestResultCalc,HDL_df_TestResultOrig])
    print('Total # of Records with HDL exam value ',HDL_final.shape)
    print('Corresponding number of patients:', HDL_final.groupby('Patient_ID').count().reset_index(drop=True).shape)

    HDL_final.to_csv('final_data/HDL_final.csv')

if __name__ == '__main__':
    extract()
