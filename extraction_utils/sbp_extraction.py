import pandas as pd
import numpy as np
import re

def test_apply(x):
    try:
        return float(x)
    except ValueError:
        match = re.search(r'\d+', x)
        if match:
            # If a number is found, convert it to an integer and return it
            return int(match.group())
        else:
            # If no number is found, return None
            return None
    

def extract():
    pd.set_option('display.max_rows', 5000)
    df=pd.read_csv('tables/all_exams.csv', sep =',')
    print('Total number of records', df.shape)
    print('Corresponding number of patients:', df.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #Extract sBP values when 'sBP' is present in exam name
    SBP_df=df.loc[(df['Exam1'].str.contains('|'.join(['sBP','BP']), case=True, regex=True , na=False)),:] 
    print('')
    print('Records with sbp exam',SBP_df.shape)
    print('Corresponding number of patients:', SBP_df.groupby('Patient_ID').count().reset_index(drop=True).shape)

    ##################################### VALUE CALC ##############################################
    #Extract only meaningful values of value calc where 'sBP' is present in exam name
    df['Result1_calc'] = df['Result1_calc'].apply(test_apply).dropna()
    SBP_coded_value_df= SBP_df.loc[(df['Result1_calc']<250) & (df['Result1_calc']>=10),:]
    SBP_coded_value_index= SBP_coded_value_df.index.to_list()
    print('Records with calculated value of sbp ( admitted values are between 10 and 250 mmHg):',SBP_coded_value_df.shape)
    print('Corresponding number of patients:', SBP_coded_value_df.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_calc')
    print(SBP_coded_value_df['Result1_calc'].describe())

    print('UNIT:')
    print(SBP_coded_value_df['UnitOfMeasure_calc'].value_counts())
    print('')
    SBP_df_notCoded= SBP_df.loc[SBP_df.index.isin(SBP_coded_value_index) == False]
    print('Records with not coded value of sbp :',SBP_df_notCoded.shape)
    print('Corresponding number of patients:', SBP_df_notCoded.groupby('Patient_ID').count().reset_index(drop=True).shape)

    ##################################### VALUE ORIG ##############################################
    #SBP_df_notCoded
    #remove value orig with characters
    SBP_df_Coded2=SBP_df_notCoded.loc[(~SBP_df_notCoded['Result1_orig'].str.contains('|'.join(['\`','o','u','e','/','l','^\s']), case=False, regex=True , na=False)) ,:] 
    SBP_df_Coded2['Result1_orig'] = SBP_df_Coded2['Result1_orig'].apply(test_apply).dropna()

    #remove value orig above 250 mmHg and below 10 mmHg
    SBP_df_Coded2=SBP_df_Coded2.loc[(SBP_df_Coded2['Result1_orig'].astype(float)<250) & (SBP_df_Coded2['Result1_orig'].astype(float)>=10),:] 
    print('Records with no \'Value calc\' of sbp and admitted numeric \'Value orig\' (between 10 and 250 mmHg))):',SBP_df_Coded2.shape)
    print('Corresponding number of patients:', SBP_df_Coded2.groupby('Patient_ID').count().reset_index(drop=True).shape)
    SBP_calc=SBP_coded_value_df.loc[:,['Exam_ID','Patient_ID','DateCreated','Result1_calc']].rename(columns={'Result1_calc':'sBP[mmHg]'})
    print(SBP_calc.head())
    SBP_orig=SBP_df_Coded2.loc[:,['Exam_ID','Patient_ID','DateCreated','Result1_orig']].rename(columns={'Result1_orig':'sBP[mmHg]'})
    print(SBP_orig.head())
    SBP_df_final= pd.concat([SBP_calc, SBP_orig])
    print(SBP_df_final.shape)
    print('Corresponding number of patients:', SBP_df_final.groupby('Patient_ID').count().reset_index(drop=True).shape)
    SBP_df_final.to_csv('final_data/sBP_final.csv')


if __name__ == "__main__":
    extract()
