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

    # Extract LDL values when 'LDL' is present in exam name calc
    LDL_df_calc = df.loc[(df['Name_calc'].str.contains('LDL', case=True, regex=True, na=False)), :]
    LDL_df_calc_index = LDL_df_calc.index.to_list()
    print('')
    print('Records with LDL exam in Name calc', LDL_df_calc.shape)
    print('Corresponding number of patients:', LDL_df_calc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('')
    # rimuovo records con info codificate x controllare i record rimanenti sulla base di name orig
    LDL_df_notCoded = df.loc[df.index.isin(LDL_df_calc_index) == False]
    # Extract LDL values when 'LDL' is present in exam name orig
    LDL_df_orig = LDL_df_notCoded.loc[(LDL_df_notCoded['Name_orig'].str.contains('|'.join(['LDL', 'low density lip']),
                                                                                 case=True, regex=True, na=False)), :]
    print('')
    print('Records with LDL exam in Name Orig', LDL_df_orig.shape)
    print('Corresponding number of patients:', LDL_df_orig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    LDL_df_orig['Name_orig'].value_counts()

    # Extract LDL values when 'LDL' is present in exam name calc
    LDL_df_TestResultCalc = LDL_df_calc.dropna(subset=['TestResult_calc'], how='all')
    print('Records with LDL exam value in Test result calc', LDL_df_TestResultCalc.shape)
    print('Corresponding number of patients:',
          LDL_df_TestResultCalc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    # for the records without Test result calc we search for test result orig
    LDL_df_TestResultOrig = LDL_df_calc.loc[LDL_df_calc.index.isin(LDL_df_TestResultCalc.index.to_list()) == False]
    # print(LDL_df_TestResultOrig.shape)
    # remove value orig with characters
    LDL_df_TestResultOrig = LDL_df_TestResultOrig.loc[(~LDL_df_TestResultOrig['TestResult_orig'].str.contains('|'.join(
        ['Phone', 'see comment', 'below', 'PND', 'DNR', '\>', '<', '-', 'Pending', 'deleted by lab', 'L', 'Z', 'TNP',
         'Note']), case=False, regex=True, na=False)), :]
    LDL_df_TestResultOrig['TestResult_orig'] = LDL_df_TestResultOrig['TestResult_orig'].apply(test_apply).dropna()

    LDL_df_TestResultOrig = LDL_df_TestResultOrig.loc[(LDL_df_TestResultOrig['TestResult_orig'].astype(float) < 10) & (
                LDL_df_TestResultOrig['TestResult_orig'].astype(float) > 0), :]
    print('Records with LDL exam value in Test result orig', LDL_df_TestResultOrig.shape)
    print('Corresponding number of patients:',
          LDL_df_TestResultOrig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print(LDL_df_TestResultOrig)
    LDL_df_TestResultCalc = LDL_df_TestResultCalc.loc[:,
                            ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_calc', 'PerformedDate']].rename(
        columns={'TestResult_calc': 'LDL [mmol/L]'})
    LDL_df_TestResultOrig = LDL_df_TestResultOrig.loc[:,
                            ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_orig', 'PerformedDate']].rename(
        columns={'TestResult_orig': 'LDL [mmol/L]'})
    LDL_final = pd.concat([LDL_df_TestResultCalc, LDL_df_TestResultOrig])
    print('Total # of Records with LDL exam value ', LDL_final.shape)
    print('Corresponding number of patients:', LDL_final.groupby('Patient_ID').count().reset_index(drop=True).shape)

    LDL_final.to_csv('final_data/LDL_final.csv')

if __name__ == '__main__':
    extract()