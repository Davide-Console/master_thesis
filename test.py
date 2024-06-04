import pandas as pd

def main():
    df = pd.read_csv('final_data/spandates_final.csv')

    df['StartDate'] = pd.to_datetime(df['StartDate'], format='ISO8601')
    df['EndDate'] = pd.to_datetime(df['EndDate'], format='ISO8601')

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
    bmi_df = pd.read_csv('final_data/BMI_final.csv') # id_df=10

    # Convert date columns to datetime objects
    bmi_df['DateCreated'] = pd.to_datetime(bmi_df['DateCreated'], errors='coerce', format='%Y-%m-%d')

    # Function to calculate the mean LabValue for a patient
    def calculate_mean_lab_value(id_df, patient_id, start_date, end_date):
        if id_df == 0:
            mask = (hdl_df['Patient_ID'] == patient_id) & (hdl_df['DateCreated'] >= start_date) & (hdl_df['DateCreated'] < end_date)
            matching_results = hdl_df[mask]['HDL [mmol/L]']
        elif id_df == 1:
            mask = (ldl_df['Patient_ID'] == patient_id) & (ldl_df['DateCreated'] >= start_date) & (ldl_df['DateCreated'] < end_date)
            matching_results = ldl_df[mask]['LDL [mmol/L]']
        elif id_df == 2:
            mask = (fbs_df['Patient_ID'] == patient_id) & (fbs_df['DateCreated'] >= start_date) & (fbs_df['DateCreated'] < end_date)
            matching_results = fbs_df[mask]['fbs [mmol/L]']
        elif id_df == 3:
            mask = (hba1c_df['Patient_ID'] == patient_id) & (hba1c_df['DateCreated'] >= start_date) & (hba1c_df['DateCreated'] < end_date)
            matching_results = hba1c_df[mask]['hba1c [mmol/L]']
        elif id_df == 4:
            mask = (tot_cholesterol_df['Patient_ID'] == patient_id) & (tot_cholesterol_df['DateCreated'] >= start_date) & (tot_cholesterol_df['DateCreated'] < end_date)
            matching_results = tot_cholesterol_df[mask]['tot_chol [mmol/L]']
        elif id_df == 5:
            mask = (triglycerides_df['Patient_ID'] == patient_id) & (triglycerides_df['DateCreated'] >= start_date) & (triglycerides_df['DateCreated'] < end_date)
            matching_results = triglycerides_df[mask]['trigl [mmol/L]']
        elif id_df == 6:
            mask = (sbp_df['Patient_ID'] == patient_id) & (sbp_df['DateCreated'] >= start_date) & (sbp_df['DateCreated'] < end_date)
            matching_results = sbp_df[mask]['sBP[mmHg]']
        elif id_df == 7:
            mask = (dbp_df['Patient_ID'] == patient_id) & (dbp_df['DateCreated'] >= start_date) & (dbp_df['DateCreated'] < end_date)
            matching_results = dbp_df[mask]['dBP[mmHg]']
        elif id_df == 8:
            mask = (weight_df['Patient_ID'] == patient_id) & (weight_df['DateCreated'] >= start_date) & (weight_df['DateCreated'] < end_date)
            matching_results = weight_df[mask]['weight[Kg]']
        elif id_df == 9:
            mask = (height_df['Patient_ID'] == patient_id) & (height_df['DateCreated'] >= start_date) & (height_df['DateCreated'] < end_date)
            matching_results = height_df[mask]['height[cm]']
        elif id_df == 10:
            mask = (bmi_df['Patient_ID'] == patient_id) & (bmi_df['DateCreated'] >= start_date) & (bmi_df['DateCreated'] < end_date)
            matching_results = bmi_df[mask]['BMI[Kg/m^2]']
        return matching_results.astype(float).mean()

    df['bmi'] = df.apply(lambda row: calculate_mean_lab_value(10, row['Patient_ID'], row['StartDate'], row['EndDate']), axis=1)
    print('**********')

    df.to_csv('final_data/dataset.csv')


if __name__=='__main__':
    main()
    