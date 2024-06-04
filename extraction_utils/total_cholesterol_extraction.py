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

    # Extract tot_cholycerides values when 'tot_chol' is present in exam name calc


    tot_chol_df_calc = df.loc[(df['Name_calc'].str.contains('TOTAL CHOLESTEROL', case=False, regex=True, na=False)), :]
    tot_chol_df_calc_index = tot_chol_df_calc.index.to_list()
    print('')
    print('Records with tot_chol exam in Name calc', tot_chol_df_calc.shape)
    print('Corresponding number of patients:', tot_chol_df_calc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('')
    # rimuovo records con info codificate x controllare i record rimanenti sulla base di name orig
    tot_chol_df_notCoded = df.loc[df.index.isin(tot_chol_df_calc_index) == False]
    # Extract tot_chol values when 'tot_chol' is present in exam name orig
    tot_chol_df_orig = tot_chol_df_notCoded.loc[(tot_chol_df_notCoded['Name_orig'].str.contains(
        '|'.join(['tot', 'total chol']), case=True, regex=True, na=False)), :]
    print('')
    print('Records with tot_chol exam in Name Orig', tot_chol_df_orig.shape)
    print('Corresponding number of patients:', tot_chol_df_orig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    tot_chol_df_orig['Name_orig'].value_counts()

    # Extract tot_chol values when 'tot_chol' is present in exam name calc
    tot_chol_df_TestResultCalc = tot_chol_df_calc.dropna(subset=['TestResult_calc'], how='all')
    print('Records with tot_chol exam value in Test result calc', tot_chol_df_TestResultCalc.shape)
    print('Corresponding number of patients:',
          tot_chol_df_TestResultCalc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    # for the records without Test result calc we search for test result orig
    tot_chol_df_TestResultOrig = tot_chol_df_calc.loc[
        tot_chol_df_calc.index.isin(tot_chol_df_TestResultCalc.index.to_list()) == False]
    # print(tot_chol_df_TestResultOrig.shape)
    # remove value orig with characters
    tot_chol_df_TestResultOrig = tot_chol_df_TestResultOrig.loc[(~tot_chol_df_TestResultOrig[
        'TestResult_orig'].str.contains('|'.join(
        ['Phone', 'see comment', 'below', 'PND', 'DNR', '\>', '<', '-', 'Pending', 'deleted by lab', 'L', 'Z', 'TNP',
         'Note']), case=False, regex=True, na=False)), :]
    tot_chol_df_TestResultOrig['TestResult_orig'] = tot_chol_df_TestResultOrig['TestResult_orig'].apply(test_apply).dropna()

    tot_chol_df_TestResultOrig = tot_chol_df_TestResultOrig.loc[
                                 (tot_chol_df_TestResultOrig['TestResult_orig'].astype(float) < 10) & (
                                             tot_chol_df_TestResultOrig['TestResult_orig'].astype(float) > 0), :]
    print('Records with tot_chol exam value in Test result orig', tot_chol_df_TestResultOrig.shape)
    print('Corresponding number of patients:',
          tot_chol_df_TestResultOrig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print(tot_chol_df_TestResultOrig)
    tot_chol_df_TestResultCalc = tot_chol_df_TestResultCalc.loc[:,
                                 ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_calc', 'PerformedDate']].rename(
        columns={'TestResult_calc': 'tot_chol [mmol/L]'})
    tot_chol_df_TestResultOrig = tot_chol_df_TestResultOrig.loc[:,
                                 ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_orig', 'PerformedDate']].rename(
        columns={'TestResult_orig': 'tot_chol [mmol/L]'})
    tot_chol_final = pd.concat([tot_chol_df_TestResultCalc, tot_chol_df_TestResultOrig])
    print('Total # of Records with tot_chol exam value ', tot_chol_final.shape)
    print('Corresponding number of patients:',
          tot_chol_final.groupby('Patient_ID').count().reset_index(drop=True).shape)

    tot_chol_final.to_csv('final_data/tot_cholesterol_final.csv')

if __name__ == '__main__':
    extract()
    