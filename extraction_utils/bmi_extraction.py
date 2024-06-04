import pandas as pd
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

    #Extract BMI values when 'BMI' is present in exam name
    BMI_df=df.loc[(df['Exam1'].str.contains('|'.join(['BMI']), case=False, regex=True , na=False)),:] 
    print('')
    print('Records with BMI exam',BMI_df.shape)
    print('Corresponding number of patients:', BMI_df.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #############################################VALUE CALC###########################################################
    #Extract only meaningful values of value calc where 'BMI' is present in exam name
    df['Result1_calc'] = df['Result1_calc'].apply(test_apply).dropna()
    BMI_coded_value_df= BMI_df.loc[(df['Result1_calc']<140) & (df['Result1_calc']>=10),:]
    BMI_coded_value_index= BMI_coded_value_df.index.to_list()
    print('Records with calculated value of BMI ( admitted values are between 10 and 140):',BMI_coded_value_df.shape)
    print('Corresponding number of patients:', BMI_coded_value_df.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_calc')
    print(BMI_coded_value_df['Result1_calc'].describe())

    BMI_df_notCoded= BMI_df.loc[BMI_df.index.isin(BMI_coded_value_index) == False]
    print('Records with not coded value of BMI :',BMI_df_notCoded.shape)
    print('Corresponding number of patients:', BMI_df_notCoded.groupby('Patient_ID').count().reset_index(drop=True).shape)
    BMI_df_notCoded

    ###############################################VALUE ORIG ######################################################
    #remove value orig with characters
    BMI_df_Coded2=BMI_df_notCoded.loc[(~BMI_df_notCoded['Result1_orig'].str.contains('|'.join(['o','\[','\`','\.\.','\/','\.$','l']), case=True, regex=True , na=False)) ,:] 
    #BMI_df_Coded2
    #remove value orig above 140 and below 10
    BMI_df_Coded2['Result1_orig'] = BMI_df_Coded2['Result1_orig'].apply(test_apply).dropna()
    BMI_df_Coded2=BMI_df_Coded2.loc[(BMI_df_Coded2['Result1_orig'].astype(float)<140) & (BMI_df_Coded2['Result1_orig'].astype(float)>=10),:] 

    print('Records with no \'Value calc\' of BMI and admitted numeric \'Value orig\' (between 30 and 350 Kg))):',BMI_df_Coded2.shape)
    print('Corresponding number of patients:', BMI_df_Coded2.groupby('Patient_ID').count().reset_index(drop=True).shape)

    BMI_calc=BMI_coded_value_df.loc[:,['Exam_ID','Patient_ID','DateCreated','Result1_calc']].rename(columns={'Result1_calc':'BMI[Kg/m^2]'})
    print(BMI_calc.head())
    BMI_orig=BMI_df_Coded2.loc[:,['Exam_ID','Patient_ID','DateCreated','Result1_orig']].rename(columns={'Result1_orig':'BMI[Kg/m^2]'})
    print(BMI_orig.head())
    BMI_df_final= pd.concat([BMI_calc, BMI_orig])
    print(BMI_df_final.shape)
    print('Corresponding number of patients:', BMI_df_final.groupby('Patient_ID').count().reset_index(drop=True).shape)
    BMI_df_final.to_csv('final_data/BMI_final.csv')


if __name__=='__main__':
    extract()
