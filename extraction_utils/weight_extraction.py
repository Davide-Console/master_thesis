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

    #Extract weight values when 'weight' is present in exam name
    weight_df=df.loc[(df['Exam1'].str.contains('|'.join(['weight']), case=False, regex=True , na=False)),:] 
    print('')
    print('Records with weight exam',weight_df.shape)
    print('Corresponding number of patients:', weight_df.groupby('Patient_ID').count().reset_index(drop=True).shape)

    ##################################### VALUE CALC ##############################################
    #Extract only meaningful values of value calc where 'weight' is present in exam name
    df['Result1_calc'] = df['Result1_calc'].apply(test_apply).dropna()
    weight_coded_value_df= weight_df.loc[(df['Result1_calc']<350) & (df['Result1_calc']>=30),:]
    weight_coded_value_index= weight_coded_value_df.index.to_list()
    print('Records with calculated value of weight ( admitted values are between 30 and 350 Kg):',weight_coded_value_df.shape)
    print('Corresponding number of patients:', weight_coded_value_df.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_calc')
    print(weight_coded_value_df['Result1_calc'].describe())

    print('UNIT:')
    print(weight_coded_value_df['UnitOfMeasure_calc'].value_counts())
    print('')
    weight_df_notCoded= weight_df.loc[weight_df.index.isin(weight_coded_value_index) == False]
    print('Records with not coded value of weight :',weight_df_notCoded.shape)
    print('Corresponding number of patients:', weight_df_notCoded.groupby('Patient_ID').count().reset_index(drop=True).shape)
    weight_df_notCoded

    ##################################### VALUE ORIG ##############################################
    #remove value orig with characters
    weight_df_Coded2=weight_df_notCoded.loc[(~weight_df_notCoded['Result1_orig'].str.contains('|'.join(['o','\[','\`','\.\.','\/','\.$','l']), case=True, regex=True , na=False)) ,:] 
    weight_df_Coded2['Result1_orig'] = weight_df_Coded2['Result1_orig'].apply(test_apply).dropna()
    #remove value orig above 350 Kg
    weight_df_Coded2=weight_df_Coded2.loc[(weight_df_Coded2['Result1_orig'].astype(float)<350) & (weight_df_Coded2['Result1_orig'].astype(float)>=30),:] 

    print('Records with no \'Value calc\' of weight and admitted numeric \'Value orig\' (between 30 and 350 Kg))):',weight_df_Coded2.shape)
    print('Corresponding number of patients:', weight_df_Coded2.groupby('Patient_ID').count().reset_index(drop=True).shape)

    weight_calc=weight_coded_value_df.loc[:,['Exam_ID','Patient_ID','DateCreated','Result1_calc']].rename(columns={'Result1_calc':'weight[Kg]'})
    print(weight_calc.head())
    weight_orig=weight_df_Coded2.loc[:,['Exam_ID','Patient_ID','DateCreated','Result1_orig']].rename(columns={'Result1_orig':'weight[Kg]'})
    print(weight_orig.head())
    weight_df_final= pd.concat([weight_calc, weight_orig])
    print(weight_df_final.shape)
    print('Corresponding number of patients:', weight_df_final.groupby('Patient_ID').count().reset_index(drop=True).shape)
    weight_df_final.to_csv('final_data/weight_final.csv')


if __name__ == "__main__":
    extract()
