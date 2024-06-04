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

    #consider only records until 6 months before diagnosis
    #df['DateCreated'] = pd.to_datetime(df['DateCreated'])

    print('EXAM 1:')
    print(df.Exam1.value_counts())
    print('EXAM 2:')
    print(df.Exam2.value_counts())
    print('UNIT:')
    print(df['UnitOfMeasure_calc'].value_counts())

    #Extract DP values when 'dBP' is present in exam name
    DBP_df=df.loc[(df['Exam2'].str.contains('|'.join(['dBP','dbP']), case=True, regex=True , na=False)),:] 
    print('')
    print('Records with dbp exam',DBP_df.shape)
    print('Corresponding number of patients:', DBP_df.groupby('Patient_ID').count().reset_index(drop=True).shape)

    ##################################### VALUE CALC ##############################################
    #Extract only meaningful values of value calc when DP values when 'dBP' is present in exam name
    DBP_coded_value_df= DBP_df.loc[(df['Result2_calc']<200) & (df['Result2_calc']>=10),:]
    DBP_coded_value_index= DBP_coded_value_df.index.to_list()
    print('Records with calculated value of dbp ( admitted values are between 10 and 200 mmHg):',DBP_coded_value_df.shape)
    print('Corresponding number of patients:', DBP_coded_value_df.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result2_calc')
    print(DBP_coded_value_df['Result2_calc'].describe())
    print('UNIT:')
    print(DBP_coded_value_df['UnitOfMeasure_calc'].value_counts())
    print('')
    DBP_df_notCoded= DBP_df.loc[DBP_df.index.isin(DBP_coded_value_index) == False]
    print('Records with not coded value of dbp :',DBP_df_notCoded.shape)
    print('Corresponding number of patients:', DBP_df_notCoded.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #################################### VALUE ORIG ######################################################################
    #remove value orig with characters
    DBP_df_Coded2=DBP_df_notCoded.loc[(~DBP_df_notCoded['Result2_orig'].str.contains('|'.join(['\`','o','u','e','/','l','^\s','s']), case=True, regex=True , na=False)) ,:] 
    DBP_df_Coded2['Result2_orig'] = DBP_df_Coded2['Result2_orig'].apply(test_apply).dropna()

    #remove value orig above 200 mmHg
    DBP_df_Coded2=DBP_df_Coded2.loc[(DBP_df_Coded2['Result2_orig'].astype(float)<200) & (DBP_df_Coded2['Result2_orig'].astype(float)>=10),:] 

    print('Records with no \'Value calc\' of dbp and admitted numeric \'Value orig\' (between 10 and 200 mmHg))):',DBP_df_Coded2.shape)
    print('Corresponding number of patients:', DBP_df_Coded2.groupby('Patient_ID').count().reset_index(drop=True).shape)

    DBP_calc=DBP_coded_value_df.loc[:,['Exam_ID','Patient_ID','DateCreated','Result2_calc']].rename(columns={'Result2_calc':'dBP[mmHg]'})
    print(DBP_calc.head())
    DBP_orig=DBP_df_Coded2.loc[:,['Exam_ID','Patient_ID','DateCreated','Result2_orig']].rename(columns={'Result2_orig':'dBP[mmHg]'})
    print(DBP_orig.head())
    DBP_df_final= pd.concat([DBP_calc, DBP_orig])
    print(DBP_df_final.shape)
    print('Corresponding number of patients:', DBP_df_final.groupby('Patient_ID').count().reset_index(drop=True).shape)
    DBP_df_final.to_csv('final_data/dBP_final.csv')


if __name__ == "__main__":
    extract()  # execute only if run as a script
