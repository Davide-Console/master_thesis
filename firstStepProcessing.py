import pandas as pd
from tqdm import tqdm
import numpy as np
import re


def find_pattern_in_array(arr, pattern):
    pattern_length = len(pattern)
    for i in range(len(arr) - pattern_length + 1):
        if arr[i:i + pattern_length] == pattern:
            return 1
    return 0


def main():
    step = input("which step you want to perform?\n1: filter step\n2: create tables\n3: create timeline tables\n4: find patterns\n5: find diabetics not used\n")
    ### FILTER STEP
    if '1' in step:
        print('|__________|\tFiltering Data')
        file_path = 'tables/labs_of_2labs_patients.csv'
        column_names = ['Patient_ID', 'PerformedDate', 'Name_orig', 'Name_calc', 'LabValue',
                        'UoM_orig', 'UoM_calc', 'nDates', 'MedicalCondition']

        # Read CSV and convert string to float. If LabValue cannot be converted is set to -1 (will fall in NG)
        df = pd.read_csv(file_path, names=column_names, index_col=False)

        # Convert 'Date' column to datetime
        df['PerformedDate'] = pd.to_datetime(df['PerformedDate'], errors='coerce', format='%Y-%m-%d')

        filtered_df = df[df.groupby('Patient_ID')['PerformedDate'].transform('nunique') > 1]

        print(filtered_df.head(100))
        filtered_df.to_csv('tables/labs_of_2labs_patients_filtered_2dates.csv')



    ### CREATE TABLES
    if '2' in step:
        print('|**________|\tCreating Tables')
        # The function takes the dataframe with the lab values and for each record define if the medical condition of the
        # patient is:
        # - Normoglycemia (NG)
        # - Prediabetes (PD)
        # - T2DM (DM)

        # Specify the CSV file path
        file_path = 'tables/labs_of_2labs_patients_filtered_2dates.csv'

        # Read CSV and convert string to float. If LabValue cannot be converted is set to -1 (will fall in NG)
        df = pd.read_csv(file_path, index_col=False)
        df['LabValue'] = pd.to_numeric(df['LabValue'], errors='coerce')
        df['LabValue'].fillna(-1, inplace=True)
        print(df.head(10))

        # Load dataframe of diabetics patients
        file_path = 'tables/t2dm_(PatientsID+DateOfOnset)_list.csv'
        df_diabetics = pd.read_csv(file_path, index_col=False)
        print(df_diabetics.head(10))

        # if LabValue is in UoM_orig, and LabValue is not numeric, copy it in LabValue
        condition = (df['LabValue'] == -1) & (pd.to_numeric(df['UoM_orig'], errors='coerce').notnull())
        df.loc[condition, 'LabValue'] = df['UoM_orig']

        # if LabValue is not numeric, delete record
        df = df[df['LabValue'] != -1]

        # Define the regex pattern for case-insensitive matching
        # FBS for PD is in range 5.6-6.9
        pattern1 = r'glucose'
        pattern2 = r'fasting'
        pattern3 = r'plasma'
        ConditionFG1 = (df['Name_calc'].str.contains(pattern1, case=False) | df['Name_calc'].str.contains(pattern2, case=False) | df['Name_calc'].str.contains(pattern3, case=False))
        ConditionFG2 = (df['LabValue'].astype(float) >= 5.6)# & (df['LabValue'].astype(float) <= 6.9))

        # IGT for PD is in range 7.8-11.0
        pattern1 = r'tolerance'
        pattern2 = r'hr'
        ConditionIGT1 = (df['Name_calc'].str.contains(pattern1, case=False) | df['Name_calc'].str.contains(pattern2, case=False))
        ConditionIGT2 = (df['LabValue'].astype(float) >= 7.8)# & (df['LabValue'].astype(float) <= 11.0))

        # HB1AC for PD is in range 5.7-6.4
        pattern1 = r'hba1c'
        pattern2 = r'a1c'
        pattern3 = r'blood'
        pattern4 = r'calcium'
        ConditionHBA1C1 = (df['Name_calc'].str.contains(pattern1, case=False) | df['Name_calc'].str.contains(pattern2, case=False) | df['Name_calc'].str.contains(pattern3, case=False) | df['Name_calc'].str.contains(pattern4, case=False))
        ConditionHBA1C2 = (df['LabValue'].astype(float) >= 5.7)# & (df['LabValue'].astype(float) <= 6.4))

        # Set Medicalcondition = PD if at least 1/3 conditions is respected
        df.loc[(ConditionFG1 & ConditionFG2) | (ConditionIGT1 & ConditionIGT2) | (ConditionHBA1C1 & ConditionHBA1C2),
               'MedicalCondition'] = 'PD'

        # Set all other records to Medicalcondition = NG
        df['MedicalCondition'].fillna('NG', inplace=True)

        # if a lab is performed after the onset, MedicalCondition = DM regardless of the values
        df['PerformedDate'] = pd.to_datetime(df['PerformedDate'], errors='coerce', format='%Y-%m-%d')
        df_diabetics['DateOfOnset'] = pd.to_datetime(df_diabetics['DateOfOnset'], errors='coerce', format='%Y-%m-%d')

        for index, row in tqdm(df_diabetics.iterrows(), total=df_diabetics.shape[0]):
            patientID = row['Patient_ID']
            dateOfOnset = row['DateOfOnset'] - pd.DateOffset(months=6)  # we consider dateOfOnset - 6months
            condition1 = (df['Patient_ID'] == patientID)
            condition2 = ((~df['PerformedDate'].isna()) & (df['PerformedDate'] >= dateOfOnset))
            df.loc[condition1 & condition2, 'MedicalCondition'] = 'DM'

        # Save the DataFrame
        print(df.head(100))
        df.to_csv('firstDB/results_onset-6months.csv')



    ### CREATE TIMELINE TABLE
    if '3' in step:
        print('|****______|\tCreating Timeline Tables')
        # Load dataframe with lab values and medical condition
        df = pd.read_csv('firstDB/results_onset-6months.csv', index_col=0)
        print(df.head(10))

        # Sort values by date
        df = df.sort_values(by='PerformedDate', ascending=True)
        print(df.head(10))

        # Create a list of all the patients present in the dataframe
        patients_list = df['Patient_ID'].unique()
        print('number of patients: ' + str(len(patients_list)))

        # Initialize dataframe which will have PatientID and Timeline_Array (containing a list of the sequence of the
        # medical conditions of that patient)
        df_timeline = pd.DataFrame(columns=['Patient_ID', 'Timeline_Array'])

        # Timeline_Array contains an array with 0=NG, 1=PD, 2=DM
        mapping = {'NG': 0, 'PD': 1, 'DM': 2}
        df['MedicalCondition'] = df['MedicalCondition'].replace(mapping)

        # If in a date, the patient has more than 1 record, the MedicalCondition has the following order of priorities:
        # DM -> PD -> NG (i.e. the max is selected)
        df = df.groupby(['Patient_ID', 'PerformedDate'])['MedicalCondition'].max().reset_index()

        # PerformedDate is converted to datetime type
        df['PerformedDate'] = pd.to_datetime(df['PerformedDate'])
        grouped_data = []
        current_patient = None
        current_week_start = None
        max_medical_condition = None

        # For each patient, if it has 2 records within a week, they are considered as one and the Medical Condition is
        # selected considering the above priorities
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            if current_patient is None or current_week_start is None or current_patient != row['Patient_ID'] or (
                    row['PerformedDate'] - current_week_start).days > 7:
                if current_patient is not None:
                    grouped_data.append([current_patient, current_week_start, max_medical_condition])
                current_patient = row['Patient_ID']
                current_week_start = row['PerformedDate']
                max_medical_condition = row['MedicalCondition']
            else:
                max_medical_condition = max(max_medical_condition, row['MedicalCondition'])

        if current_patient is not None:
            grouped_data.append([current_patient, current_week_start, max_medical_condition])

        # Create a new DataFrame with the grouped results
        df = pd.DataFrame(grouped_data, columns=['Patient_ID', 'PerformedDate', 'MedicalCondition'])

        df.to_csv('firstDB/timeline_temp.csv')

        # put values in df_timeline
        for patient in tqdm(patients_list):
            condition = df['Patient_ID'] == patient

            timeline_array = df.loc[condition, 'MedicalCondition'].values
            timeline_array = np.array(timeline_array, dtype=object)

            df_timeline.loc[len(df_timeline)] = [patient, timeline_array]

        df_timeline.to_csv('firstDB/df_timeline_onset-6months_1weekLSB.csv')



    ### FIND PATTERN
    if '4' in step:
        print('|******____|\t Finding Patterns')
        # Load dataframe
        df = pd.read_csv('firstDB/timeline_temp.csv', index_col=0)

        # Cast PerformedDate with datetime type
        df['PerformedDate'] = pd.to_datetime(df['PerformedDate'])

        # Create the column Group with a counter that increases every time there is a shift in MedicalCondition.
        # MedicalCondition   |   Group
        #        0           |     0
        #        0           |     0
        #        1           |     1
        #        0           |     2
        #        1           |     3
        #        1           |     3
        #        2           |     4
        df['Group'] = (df['MedicalCondition'] != df['MedicalCondition'].shift()).cumsum()
        print(df.head(25))

        # Filter out groups of more than one consecutive MedicalCondition
        df = df.groupby(['Patient_ID', 'MedicalCondition', 'Group']).filter(
            lambda x: (x['MedicalCondition'] == 2).any() or (len(x) > 1))

        print(df.head(25))

        # For each of the elegible records, the days of the medical condition are computed
        df = df.groupby(['Patient_ID', 'MedicalCondition', 'Group']).agg(
            DateDifference=pd.NamedAgg(column='PerformedDate', aggfunc=lambda x: (x.max() - x.min()).days),
            StartDate=pd.NamedAgg(column='PerformedDate', aggfunc='min'),
            EndDate=pd.NamedAgg(column='PerformedDate', aggfunc='max')
        ).reset_index()

        print(df.head(25))

        # Save and reload dataframe (workaround)
        df.to_csv('firstDB/spandates_temp.csv')
        df = pd.read_csv('firstDB/spandates_temp.csv', index_col=0)

        # Remove records where medical condition lasts for less than 6 months
        df = df[(df['DateDifference'] > 180) | (df['MedicalCondition'] == 2)]
        df = df.sort_values(by=['Patient_ID', 'Group'], ascending=[True, True])

        # Compute gaps
        df['StartDate'] = pd.to_datetime(df['StartDate'])
        df['EndDate'] = pd.to_datetime(df['EndDate'])
        # Sort the DataFrame by PatientID and StartDate
        df.sort_values(by=['Patient_ID', 'StartDate'], inplace=True)
        # Calculate the gap for each patient
        df['Gap'] = df.groupby('Patient_ID')['StartDate'].shift(-1) - df['EndDate']
        # Convert Gap column to integer (in days)
        df['Gap'] = df['Gap'].dt.days.fillna(0).astype(int)
        # Create a new DataFrame with specific rows based on the number of records for each patient
        result = pd.DataFrame()
        for patient_id, group in tqdm(df.groupby('Patient_ID')):
            if len(group) == 1:
                # If the patient has only one record, add a row with gap=0
                result = result.append({'Patient_ID': str(patient_id), 'Gap': str(0)}, ignore_index=True)
            else:
                # If the patient has more than one record, add rows with the gaps
                gaps = group.iloc[:-1]['Gap'].tolist()
                for gap in gaps:
                    result = result.append({'Patient_ID': str(patient_id), 'Gap': str(gap)}, ignore_index=True)
        # Create a new DataFrame with one row per patient and their respective gaps
        result = df[['Patient_ID', 'Gap']]
        result.to_csv("firstDB/gaps_spandates.csv")

        df.to_csv('firstDB/spandates.csv')

        # Create the timeline dataframe
        timeline_df = df.groupby('Patient_ID')['MedicalCondition'].agg(list).reset_index()
        timeline_df = timeline_df.rename(columns={'MedicalCondition': 'Timeline'})
        timeline_df.to_csv('firstDB/timeline_final.csv')



    ### FIND DIABETICS NOT USED
    if '5' in step:
        print('|********__|\tFinding Diabetics not Used')
        file_path = 'tables/t2dm_(PatientsID+DateOfOnset)_list.csv'
        df_diabetics = pd.read_csv(file_path)
        print(df_diabetics.head(10))
        pT2DM = df_diabetics['Patient_ID'].to_list()

        file_path = 'firstDB/timeline_final.csv'
        df_timeline = pd.read_csv(file_path)
        print(df_timeline.head(10))
        df_timeline.loc[df_timeline['Timeline'].str.contains('2', case=False, regex=True, na=False),'T2DM'] = 'yes'
        df_timeline = df_timeline.dropna(subset=['T2DM'], how='all')
        pT2DM_used = df_timeline['Patient_ID'].to_list()

        T2DM_not_used = list(set(pT2DM) - set(pT2DM_used))

        dict = {'Patient_ID': T2DM_not_used}
        df = pd.DataFrame(dict)
        df.to_csv('firstDB/diabetics_not_used.csv')

        print('|**********|\tDone!')


if __name__ == '__main__':
    main()
