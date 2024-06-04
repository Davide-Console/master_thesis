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
        

def inches2cm(x):
    return x*2.54


def feet2cm(x):
    return x*30.48
    

def extract():
    pd.set_option('display.max_rows', 5000)
    df=pd.read_csv('tables/all_exams.csv', sep =',')
    print('Total number of records', df.shape)
    print('Corresponding number of patients:', df.groupby('Patient_ID').count().reset_index(drop=True).shape)
    #Extract height values when 'height' is present in exam name
    height_df=df.loc[(df['Exam1'].str.contains('|'.join(['height']), case=False, regex=True , na=False)),:] 
    print('')
    print('Records with height exam',height_df.shape)
    print('Corresponding number of patients:', height_df.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #VALUE CALC
    #CM
    #Extract only meaningful values of value calc where 'height' is present in exam name and coded in centimeters
    height_df['Result1_calc'] = height_df['Result1_calc'].apply(test_apply).dropna()
    height_coded_value_cm_df= height_df.loc[(height_df['Result1_calc']<210) & (height_df['Result1_calc']>=80),:]
    height_coded_value_cm_index= height_coded_value_cm_df.index.to_list()
    print('Records with calculated value of height \033[1m in CM \033[0m( admitted values are between \033[1m 80 and 210 cm \033[0m):',height_coded_value_cm_df.shape)
    print('Corresponding number of patients:', height_coded_value_cm_df.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_calc')
    print(height_coded_value_cm_df['Result1_calc'].describe())
    height_df_notCodedcm= height_df.loc[height_df.index.isin(height_coded_value_cm_index) == False]
    print('Records with not coded value of height :',height_df_notCodedcm.shape)
    print('Corresponding number of patients:', height_df_notCodedcm.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #pollici
    #Extract only meaningful values of value calc where 'height' is present in exam name and coded in inches
    height_coded_value_inches_df= height_df_notCodedcm.loc[(height_df_notCodedcm['Result1_calc']<79) & (height_df_notCodedcm['Result1_calc']>=31),:]
    height_coded_value_inches_index= height_coded_value_inches_df.index.to_list()
    print('Records with calculated value of height \033[1m in INCHES \033[0m( admitted values are between \033[1m 31 and 79 inches \033[0m):',height_coded_value_inches_df.shape)
    print('Corresponding number of patients:', height_coded_value_inches_df.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_calc')
    print(height_coded_value_inches_df['Result1_calc'].describe())

    height_df_notCodedinches= height_df_notCodedcm.loc[height_df_notCodedcm.index.isin(height_coded_value_inches_index) == False]
    print('Records with not coded value of height :',height_df_notCodedinches.shape)
    print('Corresponding number of patients:', height_df_notCodedinches.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #PIEDI
    #Extract only meaningful values of value calc where 'height' is present in exam name and coded in feet
    height_coded_value_feet_df= height_df_notCodedinches.loc[(height_df_notCodedinches['Result1_calc']<6.56) & (height_df_notCodedinches['Result1_calc']>=2.62),:]
    height_coded_value_feet_index= height_coded_value_feet_df.index.to_list()
    print('Records with calculated value of height \033[1m in feet \033[0m( admitted values are between \033[1m 2.62 and 6.56 feet \033[0m):',height_coded_value_feet_df.shape)
    print('Corresponding number of patients:', height_coded_value_feet_df.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_calc')
    print(height_coded_value_feet_df['Result1_calc'].describe())

    height_df_notCodedfeet= height_df_notCodedinches.loc[height_df_notCodedinches.index.isin(height_coded_value_feet_index) == False]
    print('Records with not coded value of height :',height_df_notCodedfeet.shape)
    print('Corresponding number of patients:', height_df_notCodedfeet.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #SEARCH INSIDE VALUE ORIG
    #Considero solo i record non ancora codificati
    #piedi
    height_df_notCodedfeet['Result1_orig'] = height_df_notCodedfeet['Result1_orig'].apply(test_apply).dropna()
    height_coded_orig_feet= height_df_notCodedfeet.loc[(height_df_notCodedfeet['Result1_orig'].astype(float)<6.56) & (height_df_notCodedfeet['Result1_orig'].astype(float)>=2.62),:]
    height_coded_orig_feet_index= height_coded_orig_feet.index.to_list()
    print('Records with original value of height \033[1m in feet \033[0m( admitted values are between \033[1m 2.62 and 6.56 feet \033[0m):',height_coded_orig_feet.shape)
    print('Corresponding number of patients:', height_coded_orig_feet.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_orig')
    print(height_coded_orig_feet['Result1_orig'].describe())
    print(height_coded_orig_feet)
    height_notcoded_orig_feet= height_df_notCodedfeet.loc[height_df_notCodedfeet.index.isin(height_coded_value_feet_index) == False]
    print('Records with not coded value of height :',height_notcoded_orig_feet.shape)
    print('Corresponding number of patients:', height_notcoded_orig_feet.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #pollici
    height_coded_orig_inches= height_notcoded_orig_feet.loc[(height_notcoded_orig_feet['Result1_orig'].astype(float)<79) & (height_notcoded_orig_feet['Result1_orig'].astype(float)>=31),:]
    height_coded_orig_inches_index= height_coded_orig_inches.index.to_list()
    print('Records with original value of height \033[1m in inches \033[0m( admitted values are between \033[1m 31 and 79 inches \033[0m):',height_coded_orig_inches.shape)
    print('Corresponding number of patients:', height_coded_orig_inches.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_orig')
    print(height_coded_orig_inches['Result1_orig'].describe())
    print(height_coded_orig_inches)
    height_notCoded_orig_inches= height_notcoded_orig_feet.loc[height_notcoded_orig_feet.index.isin(height_coded_orig_inches_index) == False]
    print('Records with not coded value of height :',height_notCoded_orig_inches.shape)
    print('Corresponding number of patients:', height_notCoded_orig_inches.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #cm
    height_coded_orig_cm= height_notCoded_orig_inches.loc[(height_notCoded_orig_inches['Result1_orig'].astype(float)<200) & (height_notCoded_orig_inches['Result1_orig'].astype(float)>=80),:]
    height_coded_orig_cm_index= height_coded_orig_cm.index.to_list()
    print('Records with original value of height \033[1m in cm \033[0m( admitted values are between \033[1m 80 and 200 cm \033[0m):',height_coded_orig_cm.shape)
    print('Corresponding number of patients:', height_coded_orig_cm.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('Result1_orig')
    print(height_coded_orig_cm['Result1_orig'].describe())
    print(height_coded_orig_cm)
    height_notCoded_orig_cm= height_notCoded_orig_inches.loc[height_notCoded_orig_inches.index.isin(height_coded_orig_cm_index) == False]
    print('Records with not coded value of height :',height_notCoded_orig_cm.shape)
    print('Corresponding number of patients:', height_notCoded_orig_cm.groupby('Patient_ID').count().reset_index(drop=True).shape)

    print(height_notCoded_orig_cm)

    #drop rows where Value calc and value orig are not present
    heightNa=height_notCoded_orig_cm.dropna(subset=['Result1_orig'], how='all')
    print('Remove records where Value calc AND value orig are not present', heightNa)

    height_coded_value_cm_df=height_coded_value_cm_df.loc[:,['Exam_ID','Patient_ID','DateCreated','UnitOfMeasure_orig','Result1_calc']].rename(columns={'Result1_calc':'height[cm]'})
    height_coded_orig_cm=height_coded_orig_cm.loc[:,['Exam_ID','Patient_ID','DateCreated','UnitOfMeasure_orig','Result1_orig']].rename(columns={'Result1_orig':'height[cm]'})
    height_cm=pd.concat([height_coded_value_cm_df,height_coded_orig_cm])
    height_cm.to_csv('final_data/height_cm.csv')
    print('Total number of Records with height \033[1m in cm \033[0m( admitted values are between \033[1m 80 and 200 cm \033[0m):',height_cm.shape)
    print('Corresponding number of patients:', height_cm.groupby('Patient_ID').count().reset_index(drop=True).shape)

    height_coded_value_feet_df=height_coded_value_feet_df.loc[:,['Exam_ID','Patient_ID','DateCreated','UnitOfMeasure_orig','Result1_calc']].rename(columns={'Result1_calc':'height[cm]'})
    height_coded_orig_feet=height_coded_orig_feet.loc[:,['Exam_ID','Patient_ID','DateCreated','UnitOfMeasure_orig','Result1_orig']].rename(columns={'Result1_orig':'height[cm]'})
    height_feet=pd.concat([height_coded_value_feet_df,height_coded_orig_feet])
    height_feet.to_csv('final_data/height_feet.csv')
    height_feet['height[cm]']=height_feet['height[cm]'].apply(feet2cm)
    print('Total number of Records with height \033[1m in feet \033[0m( admitted values are between \033[1m 2.62 and 6.56 ft \033[0m):',height_feet.shape)
    print('Corresponding number of patients:', height_feet.groupby('Patient_ID').count().reset_index(drop=True).shape)

    height_coded_value_inches_df=height_coded_value_inches_df.loc[:,['Exam_ID','Patient_ID','DateCreated','UnitOfMeasure_orig','Result1_calc']].rename(columns={'Result1_calc':'height[cm]'})
    height_coded_orig_inches=height_coded_orig_inches.loc[:,['Exam_ID','Patient_ID','DateCreated','UnitOfMeasure_orig','Result1_orig']].rename(columns={'Result1_orig':'height[cm]'})
    height_inches=pd.concat([height_coded_value_inches_df,height_coded_orig_inches])
    height_inches.to_csv('final_data/height_inches.csv')
    height_inches['height[cm]']=height_inches['height[cm]'].apply(inches2cm)
    print('Total number of Records with height \033[1m in inches \033[0m( admitted values are between \033[1m 31 and 79 cm \033[0m):',height_inches.shape)
    print('Corresponding number of patients:', height_inches.groupby('Patient_ID').count().reset_index(drop=True).shape)

    #save height files to be cleaned and merged
    height_final = pd.concat([height_cm,height_feet,height_inches])
    height_final.to_csv('final_data/height_final.csv')


    


if __name__ == "__main__":
    extract()
