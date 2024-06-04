import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('final_data/spandates_final.csv')

    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])

    df.sort_values(by=['Patient_ID', 'StartDate'], inplace=True, ignore_index=True)

    df.to_csv('final_ordered.csv')

    # Count the number of occurrences of each PatientID
    patient_counts = df['Patient_ID'].value_counts()
    # Select patients with more than one record
    duplicate_patients = patient_counts[patient_counts > 1].index
    # Filter the DataFrame to select only records of duplicate patients
    df = df[df['Patient_ID'].isin(duplicate_patients)]
    df = df.drop('DateDifference', axis=1)
    df = df.drop('Group', axis=1)
    df = df.drop('Gap', axis=1)

    # load dataframes
    hdl_df = pd.read_csv('final_data/HDL_final.csv') # id_df=0
    ldl_df = pd.read_csv('final_data/LDL_final.csv') # id_df=1
    fbs_df = pd.read_csv('final_data/FBS_final.csv') # id_df=2
    hba1c_df = pd.read_csv('final_data/hba1c_final.csv') # id_df=3
    tot_cholesterol_df = pd.read_csv('final_data/tot_cholesterol_final.csv') # id_df=4
    triglycerides_df = pd.read_csv('final_data/triglycerides_final.csv') # id_df=5
    sbp_df = pd.read_csv('final_data/sBP_final.csv') # id_df=6
    dbp_df = pd.read_csv('final_data/dBP_final.csv') # id_df=7
    weight_df = pd.read_csv('final_data/weight_final.csv') # id_df=8
    height_df = pd.read_csv('final_data/height_final.csv') # id_df=9
    bmi_df = pd.read_csv('final_data/BMI_final.csv') # id_df=10

    # Convert date columns to datetime objects
    df['StartDate'] = pd.to_datetime(df['StartDate'], format='%Y-%m-%d')
    df['EndDate'] = pd.to_datetime(df['EndDate'], format='%Y-%m-%d')
    hdl_df['DateCreated'] = pd.to_datetime(hdl_df['DateCreated'], format='%Y-%m-%d')
    ldl_df['DateCreated'] = pd.to_datetime(ldl_df['DateCreated'], format='%Y-%m-%d')
    fbs_df['DateCreated'] = pd.to_datetime(fbs_df['DateCreated'], format='%Y-%m-%d')
    hba1c_df['DateCreated'] = pd.to_datetime(hba1c_df['DateCreated'], format='%Y-%m-%d')
    tot_cholesterol_df['DateCreated'] = pd.to_datetime(tot_cholesterol_df['DateCreated'], format='%Y-%m-%d')
    triglycerides_df['DateCreated'] = pd.to_datetime(triglycerides_df['DateCreated'], format='%Y-%m-%d')
    sbp_df['DateCreated'] = pd.to_datetime(sbp_df['DateCreated'], errors='coerce', format='%Y-%m-%d')
    dbp_df['DateCreated'] = pd.to_datetime(dbp_df['DateCreated'], errors='coerce', format='%Y-%m-%d')
    weight_df['DateCreated'] = pd.to_datetime(weight_df['DateCreated'], errors='coerce', format='%Y-%m-%d')
    height_df['DateCreated'] = pd.to_datetime(height_df['DateCreated'], errors='coerce', format='%Y-%m-%d')
    bmi_df['DateCreated'] = pd.to_datetime(bmi_df['DateCreated'], errors='coerce', format='%Y-%m-%d')

    # Function to calculate the mean LabValue for a patient
    def calculate_mean_lab_value(id_df, patient_id, start_date, end_date):
        if id_df == 0:
            mask = (hdl_df['Patient_ID'] == patient_id) & (hdl_df['DateCreated'] >= start_date) & (hdl_df['DateCreated'] < end_date)
            matching_results = hdl_df[mask]['HDL [mmol/L]']
            low_bound = 0.6
            high_bound = 3.0
        elif id_df == 1:
            mask = (ldl_df['Patient_ID'] == patient_id) & (ldl_df['DateCreated'] >= start_date) & (ldl_df['DateCreated'] < end_date)
            matching_results = ldl_df[mask]['LDL [mmol/L]']
            low_bound = 0.7
            high_bound = 8.0
        elif id_df == 2:
            mask = (fbs_df['Patient_ID'] == patient_id) & (fbs_df['DateCreated'] >= start_date) & (fbs_df['DateCreated'] < end_date)
            matching_results = fbs_df[mask]['fbs [mmol/L]']
            low_bound = 1.3
            high_bound = 23.0
        elif id_df == 3:
            mask = (hba1c_df['Patient_ID'] == patient_id) & (hba1c_df['DateCreated'] >= start_date) & (hba1c_df['DateCreated'] < end_date)
            matching_results = hba1c_df[mask]['hba1c [mmol/L]']
            low_bound = 0.05
            high_bound = 18.5
        elif id_df == 4:
            mask = (tot_cholesterol_df['Patient_ID'] == patient_id) & (tot_cholesterol_df['DateCreated'] >= start_date) & (tot_cholesterol_df['DateCreated'] < end_date)
            matching_results = tot_cholesterol_df[mask]['tot_chol [mmol/L]']
            low_bound = 2
            high_bound = 13.0
        elif id_df == 5:
            mask = (triglycerides_df['Patient_ID'] == patient_id) & (triglycerides_df['DateCreated'] >= start_date) & (triglycerides_df['DateCreated'] < end_date)
            matching_results = triglycerides_df[mask]['trigl [mmol/L]']
            low_bound = 0.1
            high_bound = 20.0
        elif id_df == 6:
            mask = (sbp_df['Patient_ID'] == patient_id) & (sbp_df['DateCreated'] >= start_date) & (sbp_df['DateCreated'] < end_date)
            matching_results = sbp_df[mask]['sBP[mmHg]']
            low_bound = 50
            high_bound = 266
        elif id_df == 7:
            mask = (dbp_df['Patient_ID'] == patient_id) & (dbp_df['DateCreated'] >= start_date) & (dbp_df['DateCreated'] < end_date)
            matching_results = dbp_df[mask]['dBP[mmHg]']
            low_bound = 20
            high_bound = 192
        elif id_df == 8:
            mask = (weight_df['Patient_ID'] == patient_id) & (weight_df['DateCreated'] >= start_date) & (weight_df['DateCreated'] < end_date)
            matching_results = weight_df[mask]['weight[Kg]']
            low_bound = 30
            high_bound = 350.0
        elif id_df == 9:
            mask = (height_df['Patient_ID'] == patient_id) & (height_df['DateCreated'] >= start_date) & (height_df['DateCreated'] < end_date)
            matching_results = height_df[mask]['height[cm]']
            low_bound = 80
            high_bound = 210.0
        elif id_df == 10:
            mask = (bmi_df['Patient_ID'] == patient_id) & (bmi_df['DateCreated'] >= start_date) & (bmi_df['DateCreated'] < end_date)
            matching_results = bmi_df[mask]['BMI[Kg/m^2]']
            low_bound = 10.0
            high_bound = 60.0
        matching_results = matching_results[((matching_results>=low_bound)&(matching_results<=high_bound))]
        return matching_results.astype(float).mean()

    # Apply the function to calculate the mean LabValue
    df['hdl'] = df.apply(lambda row: calculate_mean_lab_value(0, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('----------')
    df['ldl'] = df.apply(lambda row: calculate_mean_lab_value(1, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('*---------')
    df['fbs'] = df.apply(lambda row: calculate_mean_lab_value(2, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('**--------')
    df['hba1c'] = df.apply(lambda row: calculate_mean_lab_value(3, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('***-------')
    df['tot_cholesterol'] = df.apply(lambda row: calculate_mean_lab_value(4, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('****------')
    df['triglycerides'] = df.apply(lambda row: calculate_mean_lab_value(5, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('*****-----')
    df['sbp'] = df.apply(lambda row: calculate_mean_lab_value(6, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('******----')
    df['dbp'] = df.apply(lambda row: calculate_mean_lab_value(7, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('*******---')
    df['weight'] = df.apply(lambda row: calculate_mean_lab_value(8, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('********--')
    df['height'] = df.apply(lambda row: calculate_mean_lab_value(9, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('*********-')
    df['bmi'] = df.apply(lambda row: calculate_mean_lab_value(10, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('**********')

    df.to_csv('final_data/dataset_ranges1.csv')

def add_labels():
    df = pd.read_csv('final_data/dataset_ranges1.csv')

    df_sex = pd.read_csv('final_data/sex_final.csv')
    df = df.merge(df_sex, on='Patient_ID', how='left')
    df_bday = pd.read_csv('final_data/birthdate_final.csv')
    df = df.merge(df_bday, on='Patient_ID', how='left')
    df['StartDate'] = pd.to_numeric(pd.to_datetime(df['StartDate']))
    df['EndDate'] = pd.to_numeric(pd.to_datetime(df['EndDate']))
    df['BirthDate'] = pd.to_datetime(df['BirthDate'])
    df['mean_date'] = pd.to_datetime(df[['StartDate', 'EndDate']].mean(axis=1))
    df['Age'] = ((df['mean_date'] - df['BirthDate']) / np.timedelta64(1, 'D')).round() / 365.25
    df = df.drop(columns=['mean_date', 'BirthDate'])

    df['Label'] = df.groupby('Patient_ID')['MedicalCondition'].shift(-1)
    df = df.dropna(subset=['Label'])
    df.to_csv('final_data/dataset_ranges2.csv')

def add_comorbidities():

    def set_comorbidities(id, patient_id, end_date):
        if id==0:
            print('0: '+str(patient_id))
            temp = df_HT[(df_HT['Patient_ID'] == patient_id) & (df_HT['DateOfOnset']-pd.DateOffset(months=6) <= end_date)]
            if temp.empty:
                return 0
            else:
                return 1
        if id==1:
            print('1: '+str(patient_id))
            temp = df_OA[(df_OA['Patient_ID'] == patient_id) & (df_OA['DateOfOnset']-pd.DateOffset(months=6) <= end_date)]
            if temp.empty:
                return 0
            else:
                return 1        
        if id==2:
            print('2: '+str(patient_id))
            temp = df_D[(df_D['Patient_ID'] == patient_id) & (df_D['DateOfOnset']-pd.DateOffset(months=6) <= end_date)]
            if temp.empty:
                return 0
            else:
                return 1
        if id==3:
            print('3: '+str(patient_id))
            temp = df_COPD[(df_COPD['Patient_ID'] == patient_id) & (df_COPD['DateOfOnset']-pd.DateOffset(months=6) <= end_date)]
            if temp.empty:
                return 0
            else:
                return 1
    
    df = pd.read_csv('final_data/dataset_ranges2.csv')
    df_comorbidities = pd.read_csv('tables/comorbidities.csv')
    df_comorbidities['DateOfOnset'] = pd.to_datetime(df_comorbidities['DateOfOnset'])
    df_HT = df_comorbidities[df_comorbidities['Disease']=='Hypertension']
    df_OA = df_comorbidities[df_comorbidities['Disease']=='Osteoarthritis']
    df_D = df_comorbidities[df_comorbidities['Disease']=='Depression']
    df_COPD = df_comorbidities[df_comorbidities['Disease']=='COPD']
    df['EndDate'] = pd.to_datetime(df['EndDate'])


    df['Hypertension'] = df.apply(lambda row: set_comorbidities(0, row['Patient_ID'], row['EndDate']), axis=1)
    df['Osteoarthritis'] = df.apply(lambda row: set_comorbidities(1, row['Patient_ID'], row['EndDate']), axis=1)
    df['Depression'] = df.apply(lambda row: set_comorbidities(2, row['Patient_ID'], row['EndDate']), axis=1)
    df['COPD'] = df.apply(lambda row: set_comorbidities(3, row['Patient_ID'], row['EndDate']), axis=1)
    df.to_csv('final_data/dataset_ranges3.csv')


def add_last_state():
    ds=pd.read_csv('final_data/dataset_ranges3.csv')
    df=pd.read_csv('final_data/latest_medical_condition.csv')
    merged_dataset = pd.merge(ds, df, on='Patient_ID', how='left')
    merged_dataset.to_csv('final_data/dataset_m.csv')


def fill_missing_values():
    # FILL HEIGHTS
    def insert_h(row):
        '''
        Function to fill missing heights
        '''
        if pd.notna(row['height']):
            return row['height']
        else:
            mask = (height_df['Patient_ID'] == row['Patient_ID']) & (height_df['DateCreated'] <= row['StartDate']) & ((height_df['height[cm]'] >= 80.0) & (height_df['height[cm]'] <= 210.0))
            result = height_df[mask]['height[cm]']
            if result.empty:
                return row['height']
            else:
                return result.iloc[-1]

    df = pd.read_csv('final_data/dataset_m.csv')
    df['StartDate'] = pd.to_datetime(df['StartDate'], format='%Y-%m-%d')
    height_df = pd.read_csv('final_data/height_final.csv')
    print(height_df.columns)
    height_df['DateCreated'] = pd.to_datetime(height_df['DateCreated'], errors='coerce', format='%Y-%m-%d')
    height_df.sort_values(by=['Patient_ID', 'DateCreated'], inplace=True, ignore_index=True)
    df['height'] = df.apply(insert_h, axis=1)

    # FILL WEIGHTS
    def insert_w(row):
        if pd.notna(row['weight']):
            return row['weight']
        else:
            mask = (weight_df['Patient_ID'] == row['Patient_ID']) & (weight_df['DateCreated'] <= row['StartDate']) & ((weight_df['weight[Kg]'] >= 30.0) & (weight_df['weight[Kg]'] <= 350.0))
            result = weight_df[mask]['weight[Kg]']
            if result.empty:
                return row['weight']
            else:
                return result.iloc[-1]

    weight_df = pd.read_csv('final_data/weight_final.csv')
    print(weight_df.columns)
    weight_df['DateCreated'] = pd.to_datetime(weight_df['DateCreated'], errors='coerce', format='%Y-%m-%d')
    weight_df.sort_values(by=['Patient_ID', 'DateCreated'], inplace=True, ignore_index=True)
    df['weight'] = df.apply(insert_w, axis=1)

    # FILL BMI, height, weight based on 2 of the others
    df['bmi'] = np.where(df['bmi'].notnull(), df['bmi'], df['weight'] / (df['height'] / 100) ** 2)
    df['weight'] = np.where(((df['weight'].isnull()) & (df['height'].notnull()) & (df['bmi'].notnull())), df['bmi'] * (df['height'] / 100) ** 2, df['weight'])
    df['height'] = np.where(((df['height'].isnull()) & (df['weight'].notnull()) & (df['bmi'].notnull())), np.sqrt(df['weight'] / df['bmi']) * 100, df['height'])

    df.rename(columns={'MedicalCondition': 'CurrentState'}, inplace=True)
    df.rename(columns={'Label': 'FutureState'}, inplace=True)
    df.to_csv('final_data/dataset_ranges4.csv')


if __name__=='__main__':
    main()
    add_labels()
    add_comorbidities()
    add_last_state()
    fill_missing_values()
