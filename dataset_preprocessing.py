from data_utils.analysis import *
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from data_utils.preparation import *
import os

def boxplot_comparison2(dataframe, features, target):
    all_features = features + target
    X = dataframe[all_features]
    # Melt the dataframe to 'long' format for boxplot
    melted_df = pd.melt(X, id_vars=target, value_vars=features, var_name='Feature', value_name='Value')
    target_labels = {0: 'NG', 1: 'PD', 2: 'T2DM'}
    melted_df['FutureState'] = melted_df['FutureState'].map(target_labels)
    melted_df['CurrentState'] = melted_df['CurrentState'].map(target_labels)

    x_positions = []
    y_positions = []
    asterisk_counts = []
    for i, feature in enumerate(features):
        for j in range(2):
            if j==0:
                k='NG'
            else:
                k='PD'
            melted_df_nonan = melted_df.dropna(subset=['Value'], inplace=False)
            melted_df_0 = melted_df_nonan[(melted_df_nonan['Feature'] == feature) & 
                                    (melted_df_nonan['CurrentState'] == k) & 
                                    (melted_df_nonan['FutureState'] == 'NG')]
            melted_df_1 = melted_df_nonan[(melted_df_nonan['Feature'] == feature) &
                                    (melted_df_nonan['CurrentState'] == k) &
                                    (melted_df_nonan['FutureState'] == 'PD')]
            melted_df_2 = melted_df_nonan[(melted_df_nonan['Feature'] == feature) &
                                    (melted_df_nonan['CurrentState'] == k) &
                                    (melted_df_nonan['FutureState'] == 'T2DM')]
            stat, p_value = kruskal(melted_df_0['Value'], melted_df_1['Value'], melted_df_2['Value'])
            print(p_value)
            if p_value < 0.05:
                data_group = melted_df[(melted_df['Feature'] == feature) & 
                                   (melted_df['CurrentState'] == j) & 
                                   (melted_df['FutureState'] == 'PD')]['Value'].tolist()
                x_positions.append(i)
                y_positions.append(0)
                asterisk_counts.append(1)

    # Plotting asterisks on boxplot
    g = sns.catplot(x='Feature', y='Value', hue='FutureState', col='CurrentState', kind='box', 
                    data=melted_df, col_order=['NG', 'PD'], height=5, aspect=1, palette=['#ffb6c1', '#ffd700', '#add8e6'], legend=False)
    
    for i in range(2):
        for x, y, count in zip(x_positions, y_positions, asterisk_counts):
            plt.subplot(1, 2, i+1)
            plt.text(x, y, '*' * count, fontsize=18, color='red', ha='center', va='bottom')

    g.set_axis_labels('Feature', 'Value')
    g.fig.suptitle('Boxplots of Features Divided by CurrentState and FutureState')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':    
    dataframe = load_dataset('final_data/dataset_ranges4.csv', True)
    dataframe = dataframe[['Patient_ID','CurrentState','StartDate','EndDate','hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD','FutureState','LastState']]
    shapiro_df = dataframe[['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age']]
    st_list=[]
    for column in shapiro_df.columns:
        _, column_df = remove_missing_values(shapiro_df[column])
        stat, p_value = shapiro(column_df)
        print(f"Shapiro-Wilk test for {column}:")
        print(f"Statistic: {stat}, p-value: {p_value}")
        if p_value>0.05:
            st_list.append('GAUSSIAN')
        else:
            st_list.append('NON GAUSSIAN')
    print('HDL')
    print('Missing Values (hdl=0.0): ' + str((dataframe['hdl'] == 0.0).sum()))
    print('Below MIN values (hdl<0.6): ' + str(((dataframe['hdl']>0) & (dataframe['hdl']<0.6)).sum()))
    print('Above MAX values (hdl>3): ' + str(((dataframe['hdl']>3).sum())))
    print(st_list[0])
    print('LDL')
    print('Missing Values (ldl=0.0): ' + str((dataframe['ldl'] == 0.0).sum()))
    print('Below MIN values (ldl<0.7): ' + str(((dataframe['ldl']>0) & (dataframe['ldl']<0.7)).sum()))
    print('Above MAX values (hdl>8): ' + str(((dataframe['ldl']>8).sum())))
    print(st_list[1])
    print('FBS')
    print('Missing Values (fbs=0.0): ' + str((dataframe['fbs'] == 0.0).sum()))
    print('Below MIN values (fbs<3.9): ' + str(((dataframe['fbs']>0) & (dataframe['fbs']<1.3)).sum()))
    print('Above MAX values (fbs>7.8): ' + str(((dataframe['fbs']>23).sum())))
    print(st_list[2])
    print('HBA1C')
    print('Missing Values (hba1c=0.0): ' + str(((dataframe['hba1c'] == 0.0)|(dataframe['hba1c']=='nan')).sum()))
    print('Below MIN values (hba1c<0.05): ' + str(((dataframe['hba1c']>0) & (dataframe['hba1c']<0.05)).sum()))
    print('Above MAX values (hba1c>18.5): ' + str(((dataframe['hba1c']>18.5).sum())))
    print(st_list[3])
    print('Total Cholesterol')
    print('Missing Values (tot_cholesterol=0.0): ' + str((dataframe['tot_cholesterol'] == 0.0).sum()))
    print('Below MIN values (tot_cholesterol<2): ' + str(((dataframe['tot_cholesterol']>0) & (dataframe['tot_cholesterol']<2)).sum()))
    print('Above MAX values (tot_cholesterol>13): ' + str(((dataframe['tot_cholesterol']>13).sum())))
    print(st_list[4])
    print('Triglycerides')
    print('Missing Values (triglycerides=0.0): ' + str((dataframe['triglycerides'] == 0.0).sum()))
    print('Below MIN values (triglycerides<0.1): ' + str(((dataframe['triglycerides']>0) & (dataframe['triglycerides']<0.1)).sum()))
    print('Above MAX values (triglycerides>20): ' + str(((dataframe['triglycerides']>20).sum())))
    print(st_list[5])
    print('SBP')
    print('Missing Values (sbp=0.0): ' + str((dataframe['sbp'] == 0.0).sum()))
    print('Below MIN values (sbp<50): ' + str(((dataframe['sbp']>0) & (dataframe['sbp']<50)).sum()))
    print('Above MAX values (sbp>266): ' + str(((dataframe['sbp']>266).sum())))
    print(st_list[6])
    print('DBP')
    print('Missing Values (dbp=0.0): ' + str((dataframe['dbp'] == 0.0).sum()))
    print('Below MIN values (dbp<20): ' + str(((dataframe['dbp']>0) & (dataframe['dbp']<20)).sum()))
    print('Above MAX values (dbp>192): ' + str(((dataframe['dbp']>192).sum())))
    print(st_list[7])
    print('Weight')
    print('Missing Values (weight=0.0): ' + str((dataframe['weight'] == 0.0).sum()))
    print('Below MIN values (weight<40): ' + str(((dataframe['weight']>0) & (dataframe['weight']<20)).sum()))
    print('Above MAX values (weight>250): ' + str(((dataframe['weight']>250).sum())))
    print(st_list[8])
    print('Height')
    std_dev_by_patient = dataframe.groupby('Patient_ID')['height'].std()
    threshold = 50
    selected_patients = std_dev_by_patient[std_dev_by_patient > threshold].index
    print(dataframe[dataframe['Patient_ID'].isin(selected_patients)]['Patient_ID'].unique().tolist())
    result_df = dataframe[dataframe['Patient_ID'].isin(selected_patients)]
    for patient_id, group in result_df.groupby('Patient_ID'):
        plt.plot(group['StartDate'],group['height'], label=f'Patient {patient_id}', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Height')
    plt.title('Height Over Time for Each Patient')
    plt.show()
    print('Missing Values (height=0.0): ' + str((dataframe['height'] == 0.0).sum()))
    print('Below MIN values (height<100): ' + str(((dataframe['height']>0) & (dataframe['height']<100)).sum()))
    print('Above MAX values (height>210): ' + str(((dataframe['height']>210).sum())))
    print(st_list[9])
    print('BMI')
    print('Missing Values (bmi=0.0): ' + str((dataframe['bmi'] == 0.0).sum()))
    print('Below MIN values (bmi<10): ' + str(((dataframe['bmi']>0) & (dataframe['bmi']<10)).sum()))
    print('Above MAX values (bmi>60): ' + str(((dataframe['bmi']>60).sum())))
    print(st_list[10])
    print('AGE')
    print('Patients with age<18: ' + str(((dataframe['Age']<18)).sum()))
    print(st_list[11])

    dataframe = dataframe[dataframe['Age']>=18]
    #_, dataframe = remove_missing_values(dataframe, True)
    #dataframe, scaler = minmaxscaler(dataframe, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], True)
    
    def manualMinMax(df):
        for column in df.columns:
            max = df[column].max()
            min = df[column].min()
            df[column] = (df[column]-min)/(max-min)
        return df

    dataframe[['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age']] = manualMinMax(dataframe[['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age']])
    
    print('\n\nTEST')
    flag='TEST'
    if not os.path.exists('dataset_analysis/TEST'):
        os.makedirs('dataset_analysis/TEST')
    #analyze_df(dataframe, flag)
    

    print('\n\nALL RECORDS')
    flag='ALL_R'
    if not os.path.exists('dataset_analysis/ALL_R'):
        os.makedirs('dataset_analysis/ALL_R')
    #analyze_df(dataframe, flag)
    
    print('\n\nALL PATIENTS')
    flag='ALL_P'
    if not os.path.exists('dataset_analysis/ALL_P'):
        os.makedirs('dataset_analysis/ALL_P')
    dataframe['StartDate'] = pd.to_datetime(dataframe['StartDate'], format='%Y-%m-%d')
    idx = dataframe.groupby('Patient_ID')['StartDate'].idxmax()
    dataframe_a = dataframe.loc[idx]
    analyze_df(dataframe_a, flag)

    print('\n\nALWAYS NORMOGLYCEMIA vs ALWAYS PREDIABETES RECORDS')
    flag='NGvsPD_R'
    if not os.path.exists('dataset_analysis/NGvsPD_R'):
        os.makedirs('dataset_analysis/NGvsPD_R')
    def filter_function(group, state):
        return (group['CurrentState'].eq(state) & group['FutureState'].eq(state)).all()
    dataframe0 = dataframe.groupby('Patient_ID').filter(filter_function,state=0)
    dataframe1 = dataframe.groupby('Patient_ID').filter(filter_function,state=1)
    dataframe_c = pd.concat([dataframe0, dataframe1], ignore_index=True)
    #analyze_df(dataframe_c, flag)

    print('\n\nALWAYS NORMOGLYCEMIA vs ALWAYS PREDIABETES PATIENTS')
    flag='NGvsPD_P'
    if not os.path.exists('dataset_analysis/NGvsPD_P'):
        os.makedirs('dataset_analysis/NGvsPD_P')
    dataframe_c['StartDate'] = pd.to_datetime(dataframe_c['StartDate'], format='%Y-%m-%d')
    idx = dataframe_c.groupby('Patient_ID')['StartDate'].idxmax()
    dataframe_c = dataframe_c.loc[idx]
    #analyze_df(dataframe_c, flag)

    print('\n\nRECORDS OF PATIENTS WHICH WILL END UP IN DIABETES')
    flag='D_R'
    if not os.path.exists('dataset_analysis/D_R'):
        os.makedirs('dataset_analysis/D_R')
    patients_with_state_2 = dataframe[dataframe['FutureState'] == 2]['Patient_ID'].unique()
    dataframe_d = dataframe[dataframe['Patient_ID'].isin(patients_with_state_2)]
    #analyze_df(dataframe_d, flag)

    print('\n\nPATIENTS WHICH WILL END UP IN DIABETES')
    flag='D_P'
    if not os.path.exists('dataset_analysis/D_P'):
        os.makedirs('dataset_analysis/D_P')
    dataframe_d['StartDate'] = pd.to_datetime(dataframe_d['StartDate'], format='%Y-%m-%d')
    idx = dataframe_d.groupby('Patient_ID')['StartDate'].idxmax()
    dataframe_d = dataframe_d.loc[idx]
    #analyze_df(dataframe_d, flag)

    print('\n\nRECORDS OF PATIENTS WHICH WILL NOT END UP IN DIABETES')
    flag='!D_R'
    if not os.path.exists('dataset_analysis/!D_R'):
        os.makedirs('dataset_analysis/!D_R')
    dataframe_nd = dataframe[~dataframe['Patient_ID'].isin(patients_with_state_2)]
    #analyze_df(dataframe_nd, flag)

    print('\n\nPATIENTS WHICH WILL NOT END UP IN DIABETES')
    flag='!D_P'
    if not os.path.exists('dataset_analysis/!D_P'):
        os.makedirs('dataset_analysis/!D_P')
    dataframe_nd['StartDate'] = pd.to_datetime(dataframe_nd['StartDate'], format='%Y-%m-%d')
    idx = dataframe_nd.groupby('Patient_ID')['StartDate'].idxmax()
    dataframe_nd = dataframe_nd.loc[idx]
    #analyze_df(dataframe_nd, flag)


    print('\n\nRECORDS OF PATIENTS WHICH WILL END UP IN PREDIABETIES')
    flag='P_R'
    if not os.path.exists('dataset_analysis/P_R'):
        os.makedirs('dataset_analysis/P_R')
    dataframe_p = dataframe[dataframe['LastState'] == 1]
    #analyze_df(dataframe_p, flag)

    print('\n\nPATIENTS WHICH WILL END UP IN PREDIABETIES')
    flag='P_P'
    if not os.path.exists('dataset_analysis/P_P'):
        os.makedirs('dataset_analysis/P_P')
    dataframe_p['StartDate'] = pd.to_datetime(dataframe_p['StartDate'], format='%Y-%m-%d')
    idx = dataframe_p.groupby('Patient_ID')['StartDate'].idxmax()
    dataframe_p = dataframe_p.loc[idx]
    #analyze_df(dataframe_p, flag)
    
    print('\n\nRECORDS OF PATIENTS WHICH WILL END UP IN NORMOGLYCEMIA')
    flag='N_R'
    if not os.path.exists('dataset_analysis/N_R'):
        os.makedirs('dataset_analysis/N_R')
    dataframe_n = dataframe[dataframe['LastState'] == 0]
    #analyze_df(dataframe_n, flag)

    print('\n\nPATIENTS WHICH WILL END UP IN NORMOGLYCEMIA')
    flag='N_P'
    if not os.path.exists('dataset_analysis/N_P'):
        os.makedirs('dataset_analysis/N_P')
    dataframe_n['StartDate'] = pd.to_datetime(dataframe_n['StartDate'], format='%Y-%m-%d')
    idx = dataframe_n.groupby('Patient_ID')['StartDate'].idxmax()
    dataframe_n = dataframe_n.loc[idx]
    #analyze_df(dataframe_n, flag)
    
    print('\n\nRECORDS OF PATIENTS WHICH ARE CURRENTLY IN NORMOGLYCEMIA')
    flag='CN_R'
    if not os.path.exists('dataset_analysis/CN_R'):
        os.makedirs('dataset_analysis/CN_R')
    dataframe_cn = dataframe[dataframe['CurrentState'] == 0]
    #boxplot_comparison(dataframe_cn, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], 'LastState', flag, True)
    #boxplot_comparison(dataframe_cn, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], 'FutureState', flag, True)
    #find_distributions(dataframe_cn, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'LastState',flag)
    #find_distributions(dataframe_cn, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'FutureState',flag)

    print('\n\nPATIENTS WHICH ARE CURRENTLY IN NORMOGLYCEMIA')
    flag='CN_P'
    if not os.path.exists('dataset_analysis/CN_P'):
        os.makedirs('dataset_analysis/CN_P')
    dataframe_cn['StartDate'] = pd.to_datetime(dataframe_cn['StartDate'], format='%Y-%m-%d')
    idx = dataframe_cn.groupby('Patient_ID')['StartDate'].idxmax()
    dataframe_cn = dataframe_cn.loc[idx]
    #boxplot_comparison(dataframe_cn, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], 'LastState', flag, True)
    #boxplot_comparison(dataframe_cn, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], 'FutureState', flag, True)
    #find_distributions(dataframe_cn, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'LastState',flag)
    #find_distributions(dataframe_cn, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'FutureState',flag)
    boxplot_comparison2(dataframe_cn, ['hdl','ldl','triglycerides'], ['FutureState', 'CurrentState'])
    boxplot_comparison2(dataframe_cn, ['fbs','hba1c','bmi'], ['FutureState', 'CurrentState'])
    boxplot_comparison2(dataframe_cn, ['sbp','dbp','Age'], ['FutureState', 'CurrentState'])

    print('\n\nRECORDS OF PATIENTS WHICH ARE CURRENTLY IN PREDIABETES')
    flag='CP_R'
    if not os.path.exists('dataset_analysis/CP_R'):
        os.makedirs('dataset_analysis/CP_R')
    dataframe_cp = dataframe[dataframe['CurrentState'] == 1]
    #boxplot_comparison(dataframe_cp, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], 'LastState', flag, True)
    #boxplot_comparison(dataframe_cp, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], 'FutureState', flag, True)
    #find_distributions(dataframe_cp, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'LastState',flag)
    #find_distributions(dataframe_cp, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'FutureState',flag)

    print('\n\nPATIENTS WHICH ARE CURRENTLY IN PREDIABETES')
    flag='CP_P'
    if not os.path.exists('dataset_analysis/CP_P'):
        os.makedirs('dataset_analysis/CP_P')
    dataframe_cp['StartDate'] = pd.to_datetime(dataframe_cp['StartDate'], format='%Y-%m-%d')
    idx = dataframe_cp.groupby('Patient_ID')['StartDate'].idxmax()
    dataframe_cp = dataframe_cp.loc[idx]
    #boxplot_comparison(dataframe_cp, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], 'LastState', flag, True)
    #boxplot_comparison(dataframe_cp, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age'], 'FutureState', flag, True)
    #find_distributions(dataframe_cp, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'LastState',flag)
    #find_distributions(dataframe_cp, ['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Sex','Age','Hypertension','Osteoarthritis','Depression','COPD'],'FutureState',flag)
    boxplot_comparison2(dataframe_cp, ['hdl','ldl','triglycerides'], ['FutureState', 'CurrentState'])
    boxplot_comparison2(dataframe_cp, ['fbs','hba1c','bmi'], ['FutureState', 'CurrentState'])
    boxplot_comparison2(dataframe_cp, ['sbp','dbp','Age'], ['FutureState', 'CurrentState'])

    print('\n\nBLOOD EXAMS FEATURES COMPARISON BETWEEN POPULATIONS IN 2015')
    dataframe_d_blood_exams = dataframe_d[['CurrentState','FutureState','LastState','hdl','ldl','tot_cholesterol','triglycerides']]
    dataframe_p_blood_exams = dataframe_p[['CurrentState','FutureState','LastState','hdl','ldl','tot_cholesterol','triglycerides']]
    dataframe_n_blood_exams = dataframe_n[['CurrentState','FutureState','LastState','hdl','ldl','tot_cholesterol','triglycerides']]
    dataframe_blood_exams = pd.concat([dataframe_d_blood_exams, dataframe_p_blood_exams, dataframe_n_blood_exams], ignore_index=True)
    #boxplot_comparison2(dataframe_blood_exams, ['hdl','ldl','triglycerides'], ['FutureState', 'CurrentState'])

    print('\n\nPREDIABETES BIOMARKERS COMPARISON BETWEEN POPULATIONS IN 2015')
    dataframe_d_biomarkers = dataframe_d[['CurrentState','FutureState','LastState','fbs','hba1c','bmi']]
    dataframe_p_biomarkers = dataframe_p[['CurrentState','FutureState','LastState','fbs','hba1c','bmi']]
    dataframe_n_biomarkers = dataframe_n[['CurrentState','FutureState','LastState','fbs','hba1c','bmi']]
    dataframe_biomarkers = pd.concat([dataframe_d_biomarkers, dataframe_p_biomarkers, dataframe_n_biomarkers], ignore_index=True)
    #boxplot_comparison2(dataframe_biomarkers, ['fbs','hba1c','bmi'], ['FutureState', 'CurrentState'])

    print('\n\nHEALTH FEATURES COMPARISON BETWEEN POPULATIONS IN 2015')
    dataframe_d_health = dataframe_d[['CurrentState','FutureState','LastState','sbp','dbp','Age']]
    dataframe_p_health = dataframe_p[['CurrentState','FutureState','LastState','sbp','dbp','Age']]
    dataframe_n_health = dataframe_n[['CurrentState','FutureState','LastState','sbp','dbp','Age']]
    dataframe_health = pd.concat([dataframe_d_health, dataframe_p_health, dataframe_n_health], ignore_index=True)
    #boxplot_comparison2(dataframe_health, ['sbp','dbp','Age'], ['FutureState', 'CurrentState'])
