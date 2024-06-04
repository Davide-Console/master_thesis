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

    # Extract hba1c values when 'hba1c' is present in exam name calc
    hba1c_df_calc = df.loc[(df['Name_calc'].str.contains('HBA1C', case=False, regex=True, na=False)), :]
    hba1c_df_calc_index = hba1c_df_calc.index.to_list()
    print('')
    print('Records with hba1c exam in Name calc', hba1c_df_calc.shape)
    print('Corresponding number of patients:', hba1c_df_calc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('')
    # rimuovo records con info codificate x controllare i record rimanenti sulla base di name orig
    hba1c_df_notCoded = df.loc[df.index.isin(hba1c_df_calc_index) == False]
    # Extract hba1c values when 'hba1c' is present in exam name orig
    hba1c_df_orig = hba1c_df_notCoded.loc[(hba1c_df_notCoded['Name_orig'].str.contains(
        '|'.join(['glyc', 'haemoglobin', 'moglobin']), case=True, regex=True, na=False)), :]
    print('')
    print('Records with hba1c exam in Name Orig', hba1c_df_orig.shape)
    print('Corresponding number of patients:', hba1c_df_orig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    hba1c_df_orig['Name_orig'].value_counts()

    # Extract hba1c values when 'hba1c' is present in exam name calc
    hba1c_df_TestResultCalc = hba1c_df_calc.dropna(subset=['TestResult_calc'], how='all')
    print('Records with hba1c exam value in Test result calc', hba1c_df_TestResultCalc.shape)
    print('Corresponding number of patients:',
          hba1c_df_TestResultCalc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    # for the records without Test result calc we search for test result orig
    hba1c_df_TestResultOrig = hba1c_df_calc.loc[
        hba1c_df_calc.index.isin(hba1c_df_TestResultCalc.index.to_list()) == False]
    # print(hba1c_df_TestResultOrig.shape)
    # remove value orig with characters
    hba1c_df_TestResultOrig = hba1c_df_TestResultOrig.loc[(~hba1c_df_TestResultOrig['TestResult_orig'].str.contains(
        '|'.join(['Phone', 'see comment', 'below', 'PND', 'DNR', '\>', '<', '-', 'Pending', 'deleted by lab', 'L', 'Z',
                  'TNP', 'Note', 'H']), case=False, regex=True, na=False)), :]
    hba1c_df_TestResultOrig['TestResult_orig'] = hba1c_df_TestResultOrig['TestResult_orig'].apply(test_apply).dropna()

    hba1c_df_TestResultOrig = hba1c_df_TestResultOrig.loc[
                              (hba1c_df_TestResultOrig['TestResult_orig'].astype(float) < 20) & (
                                          hba1c_df_TestResultOrig['TestResult_orig'].astype(float) > 0), :]
    print('Records with hba1c exam value in Test result orig', hba1c_df_TestResultOrig.shape)
    print('Corresponding number of patients:',
          hba1c_df_TestResultOrig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print(hba1c_df_TestResultOrig)
    hba1c_df_TestResultCalc = hba1c_df_TestResultCalc.loc[:,
                              ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_calc', 'PerformedDate']].rename(
        columns={'TestResult_calc': 'hba1c [mmol/L]'})
    hba1c_df_TestResultOrig = hba1c_df_TestResultOrig.loc[:,
                              ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_orig', 'PerformedDate']].rename(
        columns={'TestResult_orig': 'hba1c [mmol/L]'})

    df_UoM = hba1c_df_calc.copy()
    df_UoM['TestResult_calc'] = pd.to_numeric(df_UoM['TestResult_calc'], errors='coerce')
    df_UoM['TestResult_orig'] = pd.to_numeric(df_UoM['TestResult_orig'], errors='coerce')
    df_UoM['TestResult_calc'].fillna(-1, inplace=True)
    df_UoM['TestResult_orig'].fillna(-1, inplace=True)
    condition = (df_UoM['TestResult_orig'] == -1) & (df_UoM['TestResult_calc'] == -1) & (pd.to_numeric(df_UoM['UnitOfMeasure_orig'], errors='coerce').notnull())
    df_UoM = df_UoM.loc[condition, ['Lab_ID','Patient_ID','DateCreated','UnitOfMeasure_orig','PerformedDate']].rename(columns={'UnitOfMeasure_orig':'hba1c [mmol/L]'})

    hba1c_final = pd.concat([hba1c_df_TestResultCalc, hba1c_df_TestResultOrig, df_UoM])
    print('Total # of Records with hba1c exam value ', hba1c_final.shape)
    print('Corresponding number of patients:', hba1c_final.groupby('Patient_ID').count().reset_index(drop=True).shape)

    hba1c_final.to_csv('final_data/hba1c_final.csv')


if __name__ == '__main__':
    extract()
