import pandas as pd
import numpy as np
import re

def main():
    df = pd.read_csv('tables/sex&age.csv', sep=',')
    print('Total number of records:', df.shape[0])
    print('Corresponding number of patients:', df['Patient_ID'].nunique())
    # extract sex (binary)
    df_sex = df.drop(['BirthYear', 'BirthMonth'], axis=1)
    df_sex['Sex'] = np.where(df_sex['Sex'].str.lower() == 'male', 1, np.where(df_sex['Sex'].str.lower() == 'female', 0, -1))
    df_sex.sort_values(by=['Patient_ID'], inplace=True)
    df_sex.to_csv('final_data/sex_final.csv', index=False)

    # extract age (mm/yyyy)
    df['BirthYear'] = np.where(df['BirthYear'].isnull(), '-1', df['BirthYear'].astype(str).str[:-2].astype(str).str.zfill(4))
    df['BirthMonth'] = np.where(df['BirthMonth'].isnull(), '-1', df['BirthMonth'].astype(str).str[:-2].astype(str).str.zfill(2))
    df['BirthDate'] = np.where((df['BirthYear'].astype(str) == '-1'),
                               '-1',
                               np.where((df['BirthYear'] != '-1')&(df['BirthMonth'] == '-1'),
                                        df['BirthYear'] + '.' + '01' + '.' + '01',
                                        df['BirthYear'] + '.' + df['BirthMonth'] + '.' + '01'))
    df.drop(['Sex', 'BirthYear', 'BirthMonth'], axis=1, inplace=True)
    df.to_csv('final_data/birthdate_final.csv', index=False)


if __name__ == "__main__":
    main()