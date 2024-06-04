import os
import pandas as pd
from extraction_utils import hdl_extraction, ldl_extraction, fbs_extraction, hba1c_extraction, total_cholesterol_extraction, triglycerides_extraction
from extraction_utils import sbp_extraction, dbp_extraction, weight_extraction, height_extraction, bmi_extraction


def main():
    # Create folder for collected data
    directory_path = "final_data"
    if not os.path.exists(directory_path):
        # If it doesn't exist, create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    # Extract age and sex data
    

    # Extract labs data
    hdl_extraction.extract()
    ldl_extraction.extract()
    fbs_extraction.extract()
    hba1c_extraction.extract()
    total_cholesterol_extraction.extract()
    triglycerides_extraction.extract()

    # Extract exams data
    sbp_extraction.extract()
    dbp_extraction.extract()
    weight_extraction.extract()
    height_extraction.extract()
    bmi_extraction.extract()

def check():
    # check some stuff
    df_hdl = pd.read_csv("final_data/HDL_final.csv")
    df_ldl = pd.read_csv("final_data/LDL_final.csv")
    df_fbs = pd.read_csv("final_data/FBS_final.csv")
    df_hba1c = pd.read_csv("final_data/hba1c_final.csv")
    df_total_cholesterol = pd.read_csv("final_data/tot_cholesterol_final.csv")
    df_triglycerides = pd.read_csv("final_data/triglycerides_final.csv")
    df_sbp = pd.read_csv("final_data/sBP_final.csv")
    df_dbp = pd.read_csv("final_data/dBP_final.csv")
    df_weight = pd.read_csv("final_data/weight_final.csv")
    df_height = pd.read_csv("final_data/height_final.csv")
    df_bmi = pd.read_csv("final_data/BMI_final.csv")

    print('---TOTAL PATIENTS---')
    print('269480')
    print('---HDL---')
    print('Records: ', df_hdl.shape[0])
    print('Patients: ', df_hdl.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---LDL---')
    print('Records: ', df_ldl.shape[0])
    print('Patients: ', df_ldl.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---FBS---')
    print('Records: ', df_fbs.shape[0])
    print('Patients: ', df_fbs.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---HBA1C---')
    print('Records: ', df_hba1c.shape[0])
    print('Patients: ', df_hba1c.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---Total Cholesterol---')
    print('Records: ', df_total_cholesterol.shape[0])
    print('Patients: ', df_total_cholesterol.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---Triglycerides---')
    print('Records: ', df_triglycerides.shape[0])
    print('Patients: ', df_triglycerides.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---SBP---')
    print('Records: ', df_sbp.shape[0])
    print('Patients: ', df_sbp.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---DBP---')
    print('Records: ', df_dbp.shape[0])
    print('Patients: ', df_dbp.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---Weight---')
    print('Records: ', df_weight.shape[0])
    print('Patients: ', df_weight.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---Height---')
    print('Records: ', df_height.shape[0])
    print('Patients: ', df_height.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---BMI---')
    print('Records: ', df_bmi.shape[0])
    print('Patients: ', df_bmi.groupby('Patient_ID').count().reset_index(drop=True).shape[0])
    print('---PATIENTS WITH ALL RECORDS---')
    p_hdl = df_hdl['Patient_ID'].unique().tolist()
    p_ldl = df_ldl['Patient_ID'].unique().tolist()
    p_fbs = df_fbs['Patient_ID'].unique().tolist()
    p_hba1c = df_hba1c['Patient_ID'].unique().tolist()
    p_total_cholesterol = df_total_cholesterol['Patient_ID'].unique().tolist()
    p_triglycerides = df_triglycerides['Patient_ID'].unique().tolist()
    p_sbp = df_sbp['Patient_ID'].unique().tolist()
    p_dbp = df_dbp['Patient_ID'].unique().tolist()
    p_weight = df_weight['Patient_ID'].unique().tolist()
    p_height = df_height['Patient_ID'].unique().tolist()
    p_bmi = df_bmi['Patient_ID'].unique().tolist()
    common_values = set(p_hdl).intersection(p_ldl, p_fbs, p_hba1c, p_total_cholesterol, p_triglycerides, p_sbp, p_dbp, p_weight, p_height, p_bmi)
    common_values = list(common_values)
    print('Patients: ', len(common_values))


if __name__ == "__main__":
    #main()
    check()
