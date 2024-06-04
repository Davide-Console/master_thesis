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

    # Extract triglycerides values when 'trigl' is present in exam name calc
    trigl_df_calc = df.loc[(df['Name_calc'].str.contains('triglycerides', case=False, regex=True, na=False)), :]
    trigl_df_calc_index = trigl_df_calc.index.to_list()
    print('')
    print('Records with trigl exam in Name calc', trigl_df_calc.shape)
    print('Corresponding number of patients:', trigl_df_calc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print('')
    # rimuovo records con info codificate x controllare i record rimanenti sulla base di name orig
    trigl_df_notCoded = df.loc[df.index.isin(trigl_df_calc_index) == False]
    # Extract trigl values when 'trigl' is present in exam name orig
    trigl_df_orig = trigl_df_notCoded.loc[
                    (trigl_df_notCoded['Name_orig'].str.contains('|'.join(['trigl']), case=True, regex=True, na=False)),
                    :]
    print('')
    print('Records with trigl exam in Name Orig', trigl_df_orig.shape)
    print('Corresponding number of patients:', trigl_df_orig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    trigl_df_orig['Name_orig'].value_counts()

    # Extract trigl values when 'trigl' is present in exam name calc
    trigl_df_TestResultCalc = trigl_df_calc.dropna(subset=['TestResult_calc'], how='all')
    print('Records with trigl exam value in Test result calc', trigl_df_TestResultCalc.shape)
    print('Corresponding number of patients:',
          trigl_df_TestResultCalc.groupby('Patient_ID').count().reset_index(drop=True).shape)
    # for the records without Test result calc we search for test result orig
    trigl_df_TestResultOrig = trigl_df_calc.loc[
        trigl_df_calc.index.isin(trigl_df_TestResultCalc.index.to_list()) == False]
    # print(trigl_df_TestResultOrig.shape)
    # remove value orig with characters
    trigl_df_TestResultOrig = trigl_df_TestResultOrig.loc[(~trigl_df_TestResultOrig['TestResult_orig'].str.contains(
        '|'.join(['Phone', 'see comment', 'below', 'PND', 'DNR', '\>', '<', '-', 'Pending', 'deleted by lab', 'L', 'Z',
                  'TNP', 'Note']), case=False, regex=True, na=False)), :]
    trigl_df_TestResultOrig['TestResult_orig'] = trigl_df_TestResultOrig['TestResult_orig'].apply(test_apply).dropna()

    trigl_df_TestResultOrig = trigl_df_TestResultOrig.loc[
                              (trigl_df_TestResultOrig['TestResult_orig'].astype(float) < 10) & (
                                          trigl_df_TestResultOrig['TestResult_orig'].astype(float) > 0), :]
    print('Records with trigl exam value in Test result orig', trigl_df_TestResultOrig.shape)
    print('Corresponding number of patients:',
          trigl_df_TestResultOrig.groupby('Patient_ID').count().reset_index(drop=True).shape)
    print(trigl_df_TestResultOrig)
    trigl_df_TestResultCalc = trigl_df_TestResultCalc.loc[:,
                              ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_calc', 'PerformedDate']].rename(
        columns={'TestResult_calc': 'trigl [mmol/L]'})
    trigl_df_TestResultOrig = trigl_df_TestResultOrig.loc[:,
                              ['Lab_ID','Patient_ID', 'DateCreated', 'TestResult_orig', 'PerformedDate']].rename(
        columns={'TestResult_orig': 'trigl [mmol/L]'})
    trigl_final = pd.concat([trigl_df_TestResultCalc, trigl_df_TestResultOrig])
    print('Total # of Records with trigl exam value ', trigl_final.shape)
    print('Corresponding number of patients:', trigl_final.groupby('Patient_ID').count().reset_index(drop=True).shape)

    trigl_final.to_csv('final_data/triglycerides_final.csv')


if __name__ == '__main__':
    extract()
