import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu, kruskal


def load_dataset(filepath, verbose=False):
    """
    Load the dataset given a csv file. Also counts the identical values between Arrival delay in days and Shipping
    delay in days

    :param filepath: path to the file from the working directory
    :param verbose: True to print some dataset description
    :return: the dataset stored in a pandas.DataFrame
    """
    df = pd.read_csv(filepath, index_col=False)
    if verbose is True:
        print(df.head(5))
        print(df.describe())
    return df


def remove_duplicates(dataframe, verbose=False):
    '''
    Removes the duplicate in a pandas.DataFrame

    :param dataframe: the dataset stored in a pandas.Dataframe object
    :param verbose: True to print some information on the execution
    :return: a pandas.DataFrame without duplicates samples
    '''
    duplicates = dataframe.duplicated(keep='first')
    if verbose is True:
        print("There are %d duplicates" % sum(duplicates))

    dataframe.drop_duplicates(inplace=True)

    if verbose is True:
        print("The dataset contains %d records" % dataframe.shape[0])

    return dataframe


def remove_missing_values(dataframe, verbose=False):
    '''
    Removes all the records with at least one missing values

    :param dataframe: the dataset stored in a pandas.Dataframe object
    :param verbose: True to print some information on the execution
    :return: a the original dataset and a dataset without samples containing missing values
    '''
    if verbose is True:
        print("There are %d missing values" % sum(dataframe.isna().sum()))
        print(dataframe.isna().sum())

    dataframe_wo_nan = dataframe.dropna(axis=0, how='any', inplace=False)
    dataframe_wo_nan.reset_index(drop=True, inplace=True)

    if verbose is True:
        print("The dataset without missing values contains %d records" % dataframe_wo_nan.shape[0])

    return dataframe, dataframe_wo_nan


def categorical_to_dummy(dataframe, variables2convert, verbose=False):
    '''
    Converts the selected attributes into dummy variables. Drops the last dummy variable for each attribute

    :param dataframe: the pandas.DataFrame with the attributes to convert
    :param variables2convert: a list containing the column names to convert
    :param verbose: True to print information on the execution
    :return: the dataset with the dummy variables converted
    '''
    for variable in variables2convert:
        dummy = pd.get_dummies(dataframe[variable], drop_first=True)
        dataframe = dataframe.drop([variable], axis=1, inplace=False)
        dataframe = pd.concat([dataframe, dummy], axis=1)

    if verbose is True:
        print(dataframe.head(5))

    return dataframe


def standardise(dataframe, features, verbose=False):
    """
    Applies the sklearn.preprocessing.StandardScaler to the features selected

    :param dataframe: the dataframe containing the variables to scale
    :param features: a list of all the attributes to be scaled
    :param verbose: True to print some information on the execution
    :return: the dataset with the converted attributes and the StandardScaler() fitted
    """
    scaler = StandardScaler()
    dataframe_stand = dataframe.copy()  # copy to keep the variables that should not be scaled
    scaler.fit(dataframe_stand[features].astype(float))
    dataframe_stand = pd.DataFrame(scaler.transform(dataframe_stand[features].astype(float)))
    dataframe_stand.columns = dataframe[features].columns

    dataframe[features] = dataframe_stand[features]

    if verbose is True:
        dataframe_stand.hist()
        plt.show()
        print(dataframe.head(5))

    return dataframe, scaler


def minmaxscaler(dataframe, features, verbose=False):
    scaler = MinMaxScaler()
    dataframe_stand = dataframe.copy()  # copy to keep the variables that should not be scaled
    scaler.fit(dataframe_stand[features].astype(float))
    dataframe_stand = pd.DataFrame(scaler.transform(dataframe_stand[features].astype(float)))
    dataframe_stand.columns = dataframe[features].columns

    dataframe[features] = dataframe_stand[features]

    if verbose is True:
        dataframe_stand.hist()
        plt.show()
        print(dataframe.head(5))

    return dataframe, scaler


def feature_2_log(dataframe, feature, log_base):
    """
    Apply a logarithmic function to a specific feature of a dataset

    :param dataframe: the dataset containing the feature to transform
    :param feature: the attribute to apply the log function to
    :param log_base: the base of the logarithmic function to apply
    :return: the dataset with the converted attribute
    """
    if min(dataframe.loc[:, feature]) < 0:  # offset to be added to the variable to avoid the log(0) issue
        offset = math.ceil(abs(min(dataframe.loc[:, feature])))
    else:
        offset = 1
    dataframe.loc[:, feature] = dataframe[feature].apply(lambda x: math.log(x + offset, log_base))

    return dataframe


def apply_pca(df, features, n_components, verbose=False):
    pca2 = PCA(n_components=n_components)
    pca2.fit(df[features])
    df_pca = pd.DataFrame(pca2.transform(df[['hdl','ldl','fbs','hba1c','tot_cholesterol','triglycerides','sbp','dbp','weight','height','bmi','Age']]))
    if verbose:
        print(pca2.explained_variance_ratio_)
        print(pca2.singular_values_)
        print(pca2.components_)
        print(pca2.n_components_)
        print(pca2.n_features_)
        print(pca2.n_samples_)
        explained_var=pd.DataFrame(pca2.explained_variance_ratio_).transpose()
        ax = sns.barplot(data=explained_var)
        ax.set(xlabel='Principal Component', ylabel='Explained Variance')
        plt.show()
        df_pca.boxplot()
        plt.show()

    return df_pca


def find_distributions(dataframe, features, target, flag):
    features.append(target)
    X = dataframe[features]
    
    X0 = X[X[target]==0]
    X1 = X[X[target]==1]
    X2 = X[X[target]==2]

    fig, axes = plt.subplots(ncols=5, nrows=4)#, figsize=(20,15))
    fig.tight_layout()

    for i, ax in zip(range(X.columns.size), axes.flat):
        sns.histplot(X0.iloc[:,i], color="blue", ax=ax)
        sns.histplot(X1.iloc[:,i], color="red", ax=ax)
        sns.histplot(X2.iloc[:,i], color="green", ax=ax)
    plt.show()

    fig=sns.pairplot(X,hue=target)
    fig.savefig('dataset_analysis/'+flag+'/pairplot_'+target+'.png')


def boxplot_comparison(dataframe, features, target, flag, verbose=False):
    features.append(target)
    X = dataframe[features]
    #_, X = remove_missing_values(X, verbose)
    X0 = X[X[target]==0]
    X1 = X[X[target]==1]
    X2 = X[X[target]==2]

    results = []
    if not X1.empty and not X2.empty:
        print('Kruskal-Wallis H test for '+flag+': ')
        for column in X.columns:
            x0 = X0[column].dropna(axis=0, how='any', inplace=False)
            x1 = X1[column].dropna(axis=0, how='any', inplace=False)
            x2 = X2[column].dropna(axis=0, how='any', inplace=False)
            stat, p_value = kruskal(x0, x1, x2)
            print(f"Kruskal-Wallis H test for {column}:")
            print(f"Statistic: {stat}, p-value: {p_value}")
            if p_value < 0.05:
                print("Likely DIFFERENT.")
                a = 'No'
            else:
                print("Likely SIMILAR.")
                a = 'Yes'
            results.append([column, stat, p_value, a])
        result_df = pd.DataFrame(results, columns=['Feature', 'Statistic', 'p-Value', 'Similar'])
        result_df.to_csv('dataset_analysis/'+flag+'/kruskal_results_'+target+'.csv', index=False)

        result_df = pd.DataFrame()
        for column in X.columns:
            p_values = []
            stats = []
            x0 = X0[column].dropna(axis=0, how='any', inplace=False)
            x1 = X1[column].dropna(axis=0, how='any', inplace=False)
            x2 = X2[column].dropna(axis=0, how='any', inplace=False)
            for i, (x, y) in enumerate([(x0, x1), (x0, x2), (x1, x2)], start=1):
                stat, p_value = mannwhitneyu(x, y)
                p_values.append(p_value)
                stats.append(stat)
                print(f"Test {i}: Statistic={stat}, p-value={p_value}")
            bonferroni_alpha = 0.05 / len(p_values)  # Bonferroni adjusted alpha
            result_dfs = pd.DataFrame({
                'Feature': [column for i in range(1, 4)],
                'Test': [f'Test {i}' for i in range(1, 4)],
                'Statistic': stats,
                'P-value': p_values,
                'Significant': [p > bonferroni_alpha for p in p_values]
                })
            result_df=pd.concat([result_df, result_dfs])
        result_df.to_csv('dataset_analysis/'+flag+'/pairwise_'+target+'.csv', index=False)
    elif X2.empty:
        print('Mann-Whitney U test for '+flag+': ')
        for column in X.columns:
            x0 = X0[column].dropna(axis=0, how='any', inplace=False)
            x1 = X1[column].dropna(axis=0, how='any', inplace=False)
            stat, p_value = mannwhitneyu(x0, x1)
            print(f"Mann-Whitney U test for {column}:")
            print(f"Statistic: {stat}, p-value: {p_value}")
            if p_value < 0.05:
                print("Likely DIFFERENT.")
                a = 'No'
            else:
                print("Likely SIMILAR.")
                a = 'Yes'
            results.append([column, stat, p_value, a])
        result_df = pd.DataFrame(results, columns=['Feature', 'Statistic', 'p-Value', 'Similar'])
        result_df.to_csv('dataset_analysis/'+flag+'/ranksum_results_'+target+'.csv', index=False)
    
    # Melt the dataframe to 'long' format for boxplot
    melted_df = pd.melt(X, id_vars=[target], var_name='Feature', value_name='Value')
    target_labels = {0: 'NG', 1: 'PD', 2: 'T2DM'}
    melted_df[target] = melted_df[target].map(target_labels)

    # Plot boxplot using seaborn
    plt.figure(figsize=(12, 8))
    fig=sns.boxplot(x='Feature', y='Value', hue=target, data=melted_df, palette=[ '#add8e6','#ffd700','#ffb6c1'], legend=False)

    for i, feature in enumerate(features[:-1]):  # Exclude the target from features
        if results[i][-1] == 'No':
            plt.text(i, 0, '*', fontsize=18, color='red', ha='center', va='bottom')


    plt.title('Boxplot Comparison of Features')
    plt.savefig('dataset_analysis/'+flag+'/boxplot_comparison_'+target+'.png') 
    plt.show()


def clip(df):
    df['hdl'] = df['hdl'].clip(lower=0.6, upper=3.0)
    df['ldl'] = df['ldl'].clip(lower=0.7, upper=8.0)
    df['fbs'] = df['fbs'].clip(lower=1.3, upper=23.0)
    df['hba1c'] = df['hba1c'].clip(lower=0.05, upper=18.5)
    df['tot_cholesterol'] = df['tot_cholesterol'].clip(lower=2, upper=13.0)
    df['triglycerides'] = df['triglycerides'].clip(lower=0.1, upper=20.0)
    df['sbp'] = df['sbp'].clip(lower=50, upper=266)
    df['dbp'] = df['dbp'].clip(lower=20, upper=192)
    df['weight'] = df['weight'].clip(lower=30, upper=350.0)
    df['height'] = df['height'].clip(lower=80, upper=210.0)
    df['bmi'] = df['bmi'].clip(lower=10.0, upper=60.0)
    return df


def find_ranges(df):
    ranges = {}
    min = {}
    max = {}
    for column in df.columns:
        if column=='hdl':
            ranges[column] = [1.0, 3.0]
            max[column] = df[column].max()
            min[column] = df[column].min()
            ranges[column][0] = (ranges[column][0]-df[column].min())/(df[column].max()-df[column].min())
            ranges[column][1] = (ranges[column][1]-df[column].min())/(df[column].max()-df[column].min())
        elif column=='ldl':
            ranges[column] = [1.5, 4.1]
            max[column] = df[column].max()
            min[column] = df[column].min()
            ranges[column][0] = (ranges[column][0]-df[column].min())/(df[column].max()-df[column].min())
            ranges[column][1] = (ranges[column][1]-df[column].min())/(df[column].max()-df[column].min())
        elif column=='fbs':
            ranges[column] = [3.2, 5.6]
            max[column] = df[column].max()
            min[column] = df[column].min()
            ranges[column][0] = (ranges[column][0]-df[column].min())/(df[column].max()-df[column].min())
            ranges[column][1] = (ranges[column][1]-df[column].min())/(df[column].max()-df[column].min())
        elif column=='hba1c':
            ranges[column] = [3.3, 5.7]
            max[column] = df[column].max()
            min[column] = df[column].min()
            ranges[column][0] = (ranges[column][0]-df[column].min())/(df[column].max()-df[column].min())
            ranges[column][1] = (ranges[column][1]-df[column].min())/(df[column].max()-df[column].min())
        elif column=='triglycerides':
            ranges[column] = [0.5, 2.3]
            max[column] = df[column].max()
            min[column] = df[column].min()
            ranges[column][0] = (ranges[column][0]-df[column].min())/(df[column].max()-df[column].min())
            ranges[column][1] = (ranges[column][1]-df[column].min())/(df[column].max()-df[column].min())
        elif column=='sbp':
            ranges[column] = [70, 130]
            max[column] = df[column].max()
            min[column] = df[column].min()
            ranges[column][0] = (ranges[column][0]-df[column].min())/(df[column].max()-df[column].min())
            ranges[column][1] = (ranges[column][1]-df[column].min())/(df[column].max()-df[column].min())
        elif column=='dbp':
            ranges[column] = [60, 80]
            max[column] = df[column].max()
            min[column] = df[column].min()
            ranges[column][0] = (ranges[column][0]-df[column].min())/(df[column].max()-df[column].min())
            ranges[column][1] = (ranges[column][1]-df[column].min())/(df[column].max()-df[column].min())
        elif column=='bmi':
            ranges[column] = [18.5, 30.0]
            max[column] = df[column].max()
            min[column] = df[column].min()
            ranges[column][0] = (ranges[column][0]-df[column].min())/(df[column].max()-df[column].min())
            ranges[column][1] = (ranges[column][1]-df[column].min())/(df[column].max()-df[column].min())
        elif column=='Age':
            max[column] = df[column].max()
            min[column] = df[column].min()
    return ranges, min, max