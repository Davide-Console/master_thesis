import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

def find_pattern_in_array(arr, pattern):
    pattern_length = len(pattern)
    count = 0
    for i in range(len(arr) - pattern_length + 1):
        if arr[i:i + pattern_length] == pattern:
            count += 1
    return count


def get_longest_string(group):
    return group.loc[group['Timeline'].apply(len).idxmax()]


def main():
    # AGGREGATE SPANDATES WITH GAPS
    df_1 = pd.read_csv('firstDB/spandates.csv', index_col=0)
    df_2 = pd.read_csv('secondDB/spandates.csv', index_col=0)

    df_1 = df_1[~df_1['Patient_ID'].isin(df_2['Patient_ID'])]
    df = pd.concat([df_1, df_2], ignore_index=True)

    df.to_csv('final_data/spandates.csv')

    # counting gaps' ranges occurrences
    df_1 = pd.read_csv('firstDB/results_onset-6months.csv', index_col=0)
    df_2 = pd.read_csv('secondDB/results_onset-6months.csv', index_col=0)

    df_1 = df_1[~df_1['Patient_ID'].isin(df_2['Patient_ID'])]
    df_reference = pd.concat([df_1, df_2], ignore_index=True)
    df_reference.sort_values(by=['Patient_ID', 'PerformedDate'], inplace=True, ignore_index=True)
    df_reference.to_csv('final_data/results_onset-6months.csv')
    mapping = {'NG': 0, 'PD': 1, 'DM': 2}
    df_reference['MedicalCondition'] = df_reference['MedicalCondition'].replace(mapping)

    M_0 = 0
    M_0_1 = 0
    M_0_1_c1 = 0
    M_0_1_c2 = 0
    M_0_1_c3 = 0
    M_0_1_c4 = 0
    M_0_1_c5 = 0
    M_0_1_c6 = 0
    M_0_1_c7 = 0
    M_0_1_c8 = 0
    M_0_1_c9 = 0
    M_0_1_c10 = 0
    M_0_1_c11 = 0
    M_0_1_c12 = 0
    M_0_1_c13 = 0
    M_0_1_c14 = 0
    M_1_2 = 0
    M_1_2_c1 = 0
    M_1_2_c2 = 0
    M_1_2_c3 = 0
    M_1_2_c4 = 0
    M_1_2_c5 = 0
    M_1_2_c6 = 0
    M_1_2_c7 = 0
    M_1_2_c8 = 0
    M_1_2_c9 = 0
    M_1_2_c10 = 0
    M_1_2_c11 = 0
    M_1_2_c12 = 0
    M_1_2_c13 = 0
    M_1_2_c14 = 0
    M_2plus = 0
    M_2plus_c1 = 0
    M_2plus_c2 = 0
    M_2plus_c3 = 0
    M_2plus_c4 = 0
    M_2plus_c5 = 0
    M_2plus_c6 = 0
    M_2plus_c7 = 0
    M_2plus_c8 = 0
    M_2plus_c9 = 0
    M_2plus_c10 = 0
    M_2plus_c11 = 0
    M_2plus_c12 = 0
    M_2plus_c13 = 0
    M_2plus_c14 = 0
    pd.to_datetime(df['StartDate'], format='%Y-%m-%d')
    pd.to_datetime(df['EndDate'], format='%Y-%m-%d')
    pd.to_datetime(df_reference['PerformedDate'], errors='coerce', format='%Y-%m-%d')

    df_temp = pd.DataFrame(columns=['Patient_ID', 'MedicalCondition', 'Group', 'DateDifference', 'StartDate', 'EndDate', 'Gap'])
    
    len_max=df.shape[0]
    for index_i, row_i in tqdm(df.iterrows(), total=len_max):
        patient_ID = row_i['Patient_ID']
        current_gap = row_i['Gap']
        df.loc[df.index[index_i], 'StartDate'] = str(np.datetime64(df.iloc[index_i]['StartDate']) - pd.DateOffset(days=7))
        if current_gap == 0:
            M_0 += 1
        elif current_gap < len_max:
            start_date = row_i['EndDate']
            end_date = df.iloc[index_i + 1]['StartDate']
            start_med_condition = row_i['MedicalCondition']
            end_med_condition = df.iloc[index_i+1]['MedicalCondition']
            filtered_df = df_reference[(df_reference['Patient_ID'] == patient_ID) & (df_reference['PerformedDate'] > start_date) & (df_reference['PerformedDate'] < end_date)]
            count = len(filtered_df)
            if current_gap > 730:
                M_2plus += 1
                if start_med_condition==0 and end_med_condition==1:
                    if count==2:
                        M_2plus_c1 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))                        
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        counter_j=0
                        for index_j, row_j in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
                            try:
                                gap=np.datetime64(filtered_df.iloc[counter_j+1]['PerformedDate'])-np.datetime64(row_j['PerformedDate'])
                                df_temp.loc[len(df_temp)] = [row_j['Patient_ID'], row_j['MedicalCondition'], -1, 0, np.datetime64(row_j['PerformedDate'])-pd.DateOffset(days=7), np.datetime64(filtered_df.iloc[counter_j+1]['PerformedDate'])-pd.DateOffset(days=7), gap]
                                counter_j+=1
                            except:
                                gap=np.datetime64(end_date)-np.datetime64(row_j['PerformedDate'])
                                corrected_date = np.datetime64(df.iloc[index_i+1]['StartDate'])-pd.DateOffset(days=7)
                                df_temp.loc[len(df_temp)] = [row_j['Patient_ID'], row_j['MedicalCondition'], -1, 0, np.datetime64(row_j['PerformedDate'])-pd.DateOffset(days=7), corrected_date, gap]
                    elif count>2:
                        M_2plus_c2 +=1
                elif start_med_condition==1 and end_med_condition==0:
                    if count==2:
                        M_2plus_c3 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        counter_j=0
                        for index_j, row_j in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
                            try:
                                gap=np.datetime64(filtered_df.iloc[counter_j+1]['PerformedDate'])-np.datetime64(row_j['PerformedDate'])
                                df_temp.loc[len(df_temp)] = [row_j['Patient_ID'], row_j['MedicalCondition'], -1, 0, np.datetime64(row_j['PerformedDate'])-pd.DateOffset(days=7), np.datetime64(filtered_df.iloc[counter_j+1]['PerformedDate'])-pd.DateOffset(days=7), gap]
                                counter_j+=1
                            except:
                                gap=np.datetime64(end_date)-np.datetime64(row_j['PerformedDate'])
                                corrected_date = np.datetime64(df.iloc[index_i+1]['StartDate'])-pd.DateOffset(days=7)
                                df_temp.loc[len(df_temp)] = [row_j['Patient_ID'], row_j['MedicalCondition'], -1, 0, np.datetime64(row_j['PerformedDate'])-pd.DateOffset(days=7), corrected_date, gap]
                    elif count>2:
                        M_2plus_c4 +=1
                elif start_med_condition == 0 and end_med_condition == 2:
                    if count == 1:
                        M_2plus_c5 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_2plus_c6 += 1
                elif start_med_condition == 1 and end_med_condition == 2:
                    if count == 1:
                        M_2plus_c7 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_2plus_c8 += 1
                elif start_med_condition == 0 and end_med_condition == 0:
                    if count == 1:
                        M_2plus_c9 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_2plus_c10 += 1
                elif start_med_condition == 1 and end_med_condition == 1:
                    if count == 1:
                        M_2plus_c11 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_2plus_c12 += 1
                elif start_med_condition == 2 and end_med_condition == 2:
                    if count == 1:
                        M_2plus_c13 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_2plus_c14 += 1
            elif current_gap > 365:
                M_1_2 += 1
                if start_med_condition==0 and end_med_condition==1:
                    if count==2:
                        M_1_2_c1 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                    elif count>2:
                        M_1_2_c2 +=1
                elif start_med_condition==1 and end_med_condition==0:
                    if count==2:
                        M_1_2_c3 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                    elif count>2:
                        M_1_2_c4 +=1
                elif start_med_condition == 0 and end_med_condition == 2:
                    if count == 1:
                        M_1_2_c5 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_1_2_c6 += 1
                elif start_med_condition == 1 and end_med_condition == 2:
                    if count == 1:
                        M_1_2_c7 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_1_2_c8 += 1
                elif start_med_condition == 0 and end_med_condition == 0:
                    if count == 1:
                        M_1_2_c9 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_1_2_c10 += 1
                elif start_med_condition == 1 and end_med_condition == 1:
                    if count == 1:
                        M_1_2_c11 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_1_2_c12 += 1
                elif start_med_condition == 2 and end_med_condition == 2:
                    if count == 1:
                        M_1_2_c13 += 1
                        # add column to spandates with row contained in df_reference having the same patient_ID and performedDate between start_date and end_date
                        df.loc[df.index[index_i], 'Gap'] = str((np.datetime64(filtered_df.iloc[0, 2]) - np.datetime64(start_date)).astype(int))
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7))
                        gap=np.datetime64(end_date)-np.datetime64(filtered_df.iloc[0]['PerformedDate'])
                        df_temp.loc[len(df_temp)] = [filtered_df.iloc[0]['Patient_ID'], filtered_df.iloc[0]['MedicalCondition'], -1, 0, np.datetime64(filtered_df.iloc[0]['PerformedDate']) - pd.DateOffset(days=7), df.iloc[index_i+1]['StartDate'], gap]
                    elif count>1:
                        M_1_2_c14 += 1
            elif current_gap > 0:
                M_0_1 += 1
                if start_med_condition==0 and end_med_condition==1:
                    if count==2:
                        M_0_1_c1 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                    elif count>2:
                        M_0_1_c2 +=1
                elif start_med_condition==1 and end_med_condition==0:
                    if count==2:
                        M_0_1_c3 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                    elif count>2:
                        M_0_1_c4 +=1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                elif start_med_condition == 0 and end_med_condition == 2:
                    if count == 1:
                        M_0_1_c5 += 1
                    elif count>1:
                        M_0_1_c6 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                elif start_med_condition == 1 and end_med_condition == 2:
                    if count == 1:
                        M_0_1_c7 += 1
                    elif count>1:
                        M_0_1_c8 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                elif start_med_condition == 0 and end_med_condition == 0:
                    if count == 1:
                        M_0_1_c9 += 1
                    elif count>1:
                        M_0_1_c10 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                elif start_med_condition == 1 and end_med_condition == 1:
                    if count == 1:
                        M_0_1_c11 += 1
                    elif count>1:
                        M_0_1_c12 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                elif start_med_condition == 2 and end_med_condition == 2:
                    if count == 1:
                        M_0_1_c13 += 1
                    elif count>1:
                        M_0_1_c14 += 1
                        df.loc[df.index[index_i], 'EndDate'] = str(np.datetime64(df.iloc[index_i+1]['StartDate']) - pd.DateOffset(days=7))
                        
    df_count = pd.DataFrame(columns=['Range', 'Count', 'Case1', 'Case2', 'Case3', 'Case4', 'Case5', 'Case6', 'Case7', 'Case8', 'Case9', 'Case10', 'Case11', 'Case12', 'Case13', 'Case14'])
    df_count.loc[len(df_count)] = ['No gap', M_0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    df_count.loc[len(df_count)] = ['0-1 years gap', M_0_1, M_0_1_c1, M_0_1_c2, M_0_1_c3, M_0_1_c4, M_0_1_c5, M_0_1_c6, M_0_1_c7, M_0_1_c8, M_0_1_c9, M_0_1_c10, M_0_1_c11, M_0_1_c12, M_0_1_c13, M_0_1_c14]
    df_count.loc[len(df_count)] = ['1-2 years gap', M_1_2, M_1_2_c1, M_1_2_c2, M_1_2_c3, M_1_2_c4, M_1_2_c5, M_1_2_c6, M_1_2_c7, M_1_2_c8, M_1_2_c9, M_1_2_c10, M_1_2_c11, M_1_2_c12, M_1_2_c13, M_1_2_c14]
    df_count.loc[len(df_count)] = ['2+ years gap', M_2plus, M_2plus_c1, M_2plus_c2, M_2plus_c3, M_2plus_c4, M_2plus_c5, M_2plus_c6, M_2plus_c7, M_2plus_c8, M_2plus_c9, M_2plus_c10, M_2plus_c11, M_2plus_c12, M_2plus_c13, M_2plus_c14]
    df_count.to_csv('final_data/gaps_count.csv')

    df_temp.to_csv('final_data/more data.csv')
    df_final = pd.concat([df, df_temp], ignore_index=True)
    df_final.sort_values(by=['Patient_ID', 'StartDate'], inplace=True, ignore_index=True)
    df_final.to_csv('final_data/spandates_final.csv')

    # AGGREGATE TIMELINE FINAL AND COUNT
    # Create the timeline dataframe
    df = df_final.groupby('Patient_ID')['MedicalCondition'].agg(list).reset_index()
    df = df.rename(columns={'MedicalCondition': 'Timeline'})
    df.to_csv('final_data/timeline_final.csv')
    
    '''df_1 = pd.read_csv('firstDB/timeline_final.csv', index_col=0)
    df_2 = pd.read_csv('secondDB/timeline_final.csv', index_col=0)

    df = pd.concat([df_1, df_2])
    df = df.groupby('Patient_ID', group_keys=False).apply(get_longest_string)
    df = df.iloc[:, 1:]
    df.to_csv('final_data/timeline_final.csv')'''

def count():
    df=pd.read_csv('final_data/spandates_final.csv')
    df = df.groupby('Patient_ID')['MedicalCondition'].agg(list).reset_index()
    df = df.rename(columns={'MedicalCondition': 'Timeline'})
    df.to_csv('final_data/timeline_final.csv')
    # counting transitions and patients
    EMPTY = 0
    NG = 0
    PD = 0
    DM = 0
    NG2PD = 0
    NG2DM = 0
    PD2DM = 0
    PD2NG = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        array_to_test = str(row['Timeline'])
        if array_to_test == ['']:
            EMPTY = EMPTY + 1
            continue
        #array_to_test = array_to_test.replace('[', '').replace(']', '')
        array_to_test = ast.literal_eval(array_to_test)
        array_to_test = [array_to_test]
        array_to_test = list(itertools.chain(*array_to_test))
        if all(element == 0 for element in array_to_test):
            NG += 1
        elif all(element == 1 for element in array_to_test):
            PD += 1
        elif all(element == 2 for element in array_to_test):
            DM += 1
        else:
            NG2PD = NG2PD + find_pattern_in_array(array_to_test, [0, 1])
            NG2DM = NG2DM + find_pattern_in_array(array_to_test, [0, 2])
            PD2DM = PD2DM + find_pattern_in_array(array_to_test, [1, 2])
            PD2NG = PD2NG + find_pattern_in_array(array_to_test, [1, 0])

    df_count = pd.DataFrame(columns=['Transition', 'Count'])
    df_count.loc[len(df_count)] = ['Number of Patients', len(df)]
    df_count.loc[len(df_count)] = ['always NG', NG]
    df_count.loc[len(df_count)] = ['always PD', PD]
    df_count.loc[len(df_count)] = ['always T2DM', DM]
    df_count.loc[len(df_count)] = ['NO records', EMPTY]
    df_count.loc[len(df_count)] = ['NG --> PD', NG2PD]
    df_count.loc[len(df_count)] = ['NG --> T2DM', NG2DM]
    df_count.loc[len(df_count)] = ['PD --> T2DM', PD2DM]
    df_count.loc[len(df_count)] = ['PD --> NG', PD2NG]
    df_count.to_csv('final_data/df_count_onset-6months.csv')

    # COUNTS ON DIABETICS NEVER USED
    file_path = 'tables/t2dm_(PatientsID+DateOfOnset)_list.csv'
    df_diabetics = pd.read_csv(file_path)
    pT2DM = df_diabetics['Patient_ID'].to_list()

    file_path = 'final_data/timeline_final.csv'
    df_sd = pd.read_csv(file_path)
    df_sd.loc[df_sd['Timeline'].astype(str).str.contains('2', case=False, regex=True, na=False), 'T2DM'] = 'yes'
    df_sd = df_sd.dropna(subset=['T2DM'], how='all')
    pT2DM_used = df_sd['Patient_ID'].to_list()

    T2DM_not_used = list(set(pT2DM) - set(pT2DM_used))

    dict = {'Patient_ID': T2DM_not_used}
    df = pd.DataFrame(dict)
    df.to_csv('final_data/diabetics_never_used.csv')


if __name__ == '__main__':
    main()
    count()