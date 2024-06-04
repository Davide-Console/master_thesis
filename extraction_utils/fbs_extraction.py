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

    # Extract fbsycerides values when 'fbs' is present in exam name calc
    fbs_df_calc = df.loc[(df['Name_calc'].str.contains('fasting glucose', case=False, regex=True, na=False)), :]
    fbs_df_calc_index = fbs_df_calc.index.to_list()
    print('')
    print('Records with fbs exam in Name calc', fbs_df_calc.shape)
    print('Corresponding number of patients:', fbs_df_calc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('')
    # rimuovo records con info codificate x controllare i record rimanenti sulla base di name orig
    fbs_df_notCoded = df.loc[df.index.isin(fbs_df_calc_index) == False]
    # Extract fbs values when 'fbs' is present in exam name orig
    fbs_df_orig = fbs_df_notCoded.loc[(fbs_df_notCoded['Name_orig'].str.contains(
        '|'.join(['fasting', 'glucose', 'fbs']), case=True, regex=True, na=False)), :]
    print('')
    print('Records with fbs exam in Name Orig', fbs_df_orig.shape)
    print('Corresponding number of patients:', fbs_df_orig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    fbs_df_orig['Name_orig'].value_counts()

    # Extract fbs values when 'fbs' is present in exam name calc
    fbs_df_TestResultCalc = fbs_df_calc.dropna(subset=['TestResult_calc'], how='all')
    print('Records with fbs exam value in Test result calc', fbs_df_TestResultCalc.shape)
    print('Corresponding number of patients:',
          fbs_df_TestResultCalc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    # for the records without Test result calc we search for test result orig
    fbs_df_TestResultOrig = fbs_df_calc.loc[fbs_df_calc.index.isin(fbs_df_TestResultCalc.index.to_list()) == False]
    # print(fbs_df_TestResultOrig.shape)
    # remove value orig with characters
    fbs_df_TestResultOrig = fbs_df_TestResultOrig.loc[(~fbs_df_TestResultOrig['TestResult_orig'].str.contains('|'.join(
        ['Phone', 'see comment', 'below', 'PND', 'DNR', '\>', '<', '-', 'Pending', 'deleted by lab', 'L', 'Z', 'TNP',
         'Note', '\/', 'H']), case=False, regex=True, na=False)), :]
    fbs_df_TestResultOrig['TestResult_orig'] = fbs_df_TestResultOrig['TestResult_orig'].apply(test_apply).dropna()

    fbs_df_TestResultOrig = fbs_df_TestResultOrig.loc[(fbs_df_TestResultOrig['TestResult_orig'].astype(float) < 100) & (
                fbs_df_TestResultOrig['TestResult_orig'].astype(float) > 0), :]
    print('Records with fbs exam value in Test result orig', fbs_df_TestResultOrig.shape)
    print('Corresponding number of patients:',
          fbs_df_TestResultOrig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print(fbs_df_TestResultOrig)
    fbs_df_TestResultCalc = fbs_df_TestResultCalc.loc[:,
                            ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_calc', 'PerformedDate']].rename(
        columns={'TestResult_calc': 'fbs [mmol/L]'})
    fbs_df_TestResultOrig = fbs_df_TestResultOrig.loc[:,
                            ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_orig', 'PerformedDate']].rename(
        columns={'TestResult_orig': 'fbs [mmol/L]'})
    fbs_final = pd.concat([fbs_df_TestResultCalc, fbs_df_TestResultOrig])
    print('Total # of Records with fbs exam value ', fbs_final.shape)
    print('Corresponding number of patients:', fbs_final.groupby('Patient_ID').count().reset_index(drop=True).shape)

    fbs_final.to_csv('final_data/fbs_final.csv')


if __name__ == '__main__':
    extract()
