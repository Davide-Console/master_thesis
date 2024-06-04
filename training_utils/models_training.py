from sklearn.inspection import PartialDependenceDisplay
from matplotlib import pyplot as plt
import sklearn as sk
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree
import xgboost as xgb
import shap
import numpy as np
from dtreeviz.trees import dtreeviz # remember to load the package
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
print(os.environ["PATH"])

RS_GS = array = [i for i in range(0, 1440)]

def specificity(y_true, y_pred, class_label):
    tn, fp, fn, tp = confusion_matrix(y_true == class_label, y_pred == class_label).ravel()
    return tn / (tn + fp)

def dice_coefficient(y_true, y_pred):
    intersection = sum(y_true * y_pred)
    return (2.0 * intersection) / (sum(y_true) + sum(y_pred))

def train_decision_tree(X_train, y_train, X_test, y_test, sampling=None, xai=False, save=False):
    idx=0
    clf = DecisionTreeClassifier()
    num_label=np.unique(y_train).shape[0]
    min_label=np.min(y_train)
    pipeline=make_pipeline(clf)
    # Define the grid of parameters to search
    param_grid = {
        'decisiontreeclassifier__random_state': [42],
#        'decisiontreeclassifier__criterion': ['gini'],
#        'decisiontreeclassifier__max_depth': [3],
#        'decisiontreeclassifier__min_samples_split': [2],
#        'decisiontreeclassifier__min_samples_leaf': [1],
        'decisiontreeclassifier__criterion': ['gini', 'entropy'],
        'decisiontreeclassifier__max_depth': [None, 3, 4, 5, 10, 15, 20],
        'decisiontreeclassifier__min_samples_split': [2, 3, 5, 7, 10],
        'decisiontreeclassifier__min_samples_leaf': [1, 2, 4, 5, 7, 10, 15, 20],
    }
    if sampling=='under':
        idx=1
        # Instantiate the RandomUnderSampler
        undersampler = RandomUnderSampler(random_state=42)
        # Create a pipeline including the undersampling and classifier
        pipeline = make_pipeline(undersampler, clf)
    elif sampling=='over':
        idx=1
        # Instantiate the RandomUnderSampler
        oversampler = RandomOverSampler(random_state=42)
        # Create a pipeline including the undersampling and classifier
        pipeline = make_pipeline(oversampler, clf)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', verbose=3, n_jobs=-1, return_train_score=True, refit='f1_macro')

    # Fit the GridSearchCV to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best estimator
    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    predictions_train = best_clf.predict(X_train)
    predictions_test = best_clf.predict(X_test)

    # Evaluate the best model
    f1_train = sk.metrics.f1_score(y_train, predictions_train, average='macro')
    f1_test = sk.metrics.f1_score(y_test, predictions_test, average='macro')
    print(f"Best Model - F1 train: {f1_train:.2f} - F1 test: {f1_test:.2f}")
    print("Best Parameters:", best_params)
    print("TRAINING")
    print("Classification Report:")
    print(classification_report(y_train, predictions_train))
    average_specificity = 0
    for class_label in range(num_label):
        spec = specificity(y_train, predictions_train, class_label)
        average_specificity += spec
    average_specificity /= num_label
    print(f"Average Specificity: {average_specificity}")
    print("TESTING")
    print("Classification Report:")
    print(classification_report(y_test, predictions_test))
    average_specificity = 0
    for class_label in range(num_label):
        spec = specificity(y_test, predictions_test, class_label)
        average_specificity += spec
    average_specificity /= num_label
    print(f"Average Specificity: {average_specificity}")

    tree_rules = export_text(best_clf[idx], feature_names=X_train.columns)
    print("Decision Rules:")
    print(tree_rules)


    with open('decision_tree_rules.txt', 'w') as file:
        file.write(tree_rules)

    # Plot the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(best_clf[idx], filled=True, rounded=True, class_names=['NG', 'PD', 'T2DM'], feature_names=X_train.columns)
    plt.title("Decision Tree Visualization")
    plt.show()

    dot_data = export_graphviz(best_clf[idx], out_file=None, filled=True, rounded=True, special_characters=True)
    with open('decision_tree.dot', 'w') as file:
        file.write(dot_data)

    if xai:
        # Plot the feature importances
        importances = best_clf[idx].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

        #shap        
        explainer = shap.TreeExplainer(best_clf[idx], X_train)
        shap_values = explainer(X_train)
        shap_values_2 = explainer.shap_values(X_train)
        
        if num_label==2 and min_label==0:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['NG', 'T2DM'], feature_names=X_train.columns)
        elif num_label==2 and min_label==1:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['PD', 'T2DM'], feature_names=X_train.columns)
        else:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['NG', 'PD', 'T2DM'], feature_names=X_train.columns)

        for i in range(num_label):
            shap.plots.beeswarm(shap_values[:,:,i])
            
        for i in range(5):
            if i==0:
                features_idx=[1,3,6,8]
            elif i==1:
                features_idx=[2,3,5]
            elif i==2:
                features_idx=[6,7,11]
            elif i==3:
                features_idx=[8,9,10]
            elif i==4:
                features_idx=[12,13,14]
            if num_label==3:
                fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(12, 14), sharey=True)
                mlp_disp1 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_train,features_idx, ax=ax1 ,line_kw={"color": "green"}
                ,kind='average',target=0)
                mlp_disp2 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_train,features_idx, ax=ax2, line_kw={"color": "orange"}
                ,kind='average',target=1)
                mlp_disp3 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_train,features_idx, ax=ax3, line_kw={"color": "red"}
                ,kind='average',target=2)
                ax1.set_title("Normoglycemia")
                ax2.set_title("Prediabetes")
                ax3.set_title("T2DM")
                plt.show()
            elif num_label==2:
                fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 15), sharey=True)
                mlp_disp1 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx[0:2], ax=ax1
                ,kind='both')
                mlp_disp2 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx[2:4], ax=ax2
                ,kind='both')
                if min_label==0:
                    plt.suptitle("Normoglycemia - T2DM")
                elif min_label==1:
                    plt.suptitle("Prediabetes - T2DM")
                plt.show()
            
        for j in range(len(predictions_train)):
            print('---')
            if j==0: #y_train.iloc[j]==predictions_train[j] and X_train.iloc[j]['fbs']<5.6 and X_train.iloc[j]['hba1c']<5.7:
                for i in range(num_label):
                    print(predictions_train[j], y_train.iloc[j])
                    shap.plots.waterfall(shap_values[j,:,i])

        if num_label==2 and min_label==0:
            viz = dtreeviz(best_clf[idx], X_train, y_train,
                    target_name="FutureState",
                    feature_names=X_train.columns,
                    class_names=['NG', 'T2DM'])
        elif num_label==2 and min_label==1:
            viz = dtreeviz(best_clf[idx], X_train, y_train,
                    target_name="FutureState",
                    feature_names=X_train.columns,
                    class_names=['PD', 'T2DM'])
        else:
            viz = dtreeviz(best_clf[idx], X_train, y_train,
                    target_name="FutureState",
                    feature_names=X_train.columns,
                    class_names=['NG', 'PD', 'T2DM'])
        viz.view()

    if save:
        import pickle
        with open('decision_tree.pkl', 'wb') as file:
            pickle.dump(best_clf[idx], file)



def train_random_forest(X_train, y_train, X_test, y_test, sampling=None, xai=False, save=False):
    idx=0
    crf = RandomForestClassifier()
    num_label=np.unique(y_train).shape[0]
    min_label=np.min(y_train)
    pipeline=make_pipeline(crf)
    # Define the grid of parameters to search
    param_grid = {
        'randomforestclassifier__random_state': [42],
        'randomforestclassifier__n_estimators': [20],
        'randomforestclassifier__criterion': ['gini'],
        'randomforestclassifier__max_depth': [7],
        'randomforestclassifier__min_samples_split': [5],
        'randomforestclassifier__min_samples_leaf': [4],
        'randomforestclassifier__max_features': ['sqrt']
#        'randomforestclassifier__n_estimators': [5, 10, 15, 20],
#        'randomforestclassifier__criterion': ['gini', 'entropy'],
#        'randomforestclassifier__max_depth': [None, 3, 5, 7, 10],
#        'randomforestclassifier__min_samples_split': [5, 10, 20, 30],
#        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
#        'randomforestclassifier__max_features': [None, 'sqrt', 'log2']
    }
    if sampling=='under':
        idx=1
        # Instantiate the RandomUnderSampler
        undersampler = RandomUnderSampler(random_state=42)
        # Create a pipeline including the undersampling and classifier
        pipeline = make_pipeline(undersampler, crf)
    elif sampling=='over':
        idx=1
        # Instantiate the RandomUnderSampler
        oversampler = RandomOverSampler(random_state=42)
        # Create a pipeline including the undersampling and classifier
        pipeline = make_pipeline(oversampler, crf)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', verbose=3, n_jobs=-1, refit='f1_macro')

    # Fit the GridSearchCV to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best estimator
    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    predictions_train = best_clf.predict(X_train)
    predictions_test = best_clf.predict(X_test)

    # Evaluate the best model
    f1_train = sk.metrics.f1_score(y_train, predictions_train, average='macro')
    f1_test = sk.metrics.f1_score(y_test, predictions_test, average='macro')
    print(f"Best Model - F1 train: {f1_train:.2f} - F1 test: {f1_test:.2f}")
    print("Best Parameters:", best_params)
    print("TRAINING")
    print("Classification Report:")
    print(classification_report(y_train, predictions_train))
    average_specificity = 0
    for class_label in range(num_label):
        spec = specificity(y_train, predictions_train, class_label)
        average_specificity += spec
    average_specificity /= num_label
    print(f"Average Specificity: {average_specificity}")
    print("TESTING")
    print("Classification Report:")
    print(classification_report(y_test, predictions_test))
    average_specificity = 0
    for class_label in range(num_label):
        spec = specificity(y_test, predictions_test, class_label)
        average_specificity += spec
    average_specificity /= num_label
    print(f"Average Specificity: {average_specificity}")

    if xai:
        # Plot the feature importances
        importances = best_clf[idx].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

        #shap        
        explainer = shap.TreeExplainer(best_clf[idx], X_train)
        shap_values = explainer(X_train, check_additivity=False)
        shap_values_2 = explainer.shap_values(X_train, check_additivity=False)
        
        if num_label==2 and min_label==0:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['NG', 'T2DM'], feature_names=X_train.columns)
        elif num_label==2 and min_label==1:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['PD', 'T2DM'], feature_names=X_train.columns)
        else:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['NG', 'PD', 'T2DM'], feature_names=X_train.columns)

        for i in range(num_label):
            shap.plots.beeswarm(shap_values[:,:,i])
            
        for i in range(5):
            if i==0:
                features_idx=[0,1,4]
            elif i==1:
                features_idx=[2,3,5]
            elif i==2:
                features_idx=[6,7,11]
            elif i==3:
                features_idx=[8,9,10]
            elif i==4:
                features_idx=[12,13,14]
            if num_label==3:
                fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(12, 14), sharey=True)
                mlp_disp1 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx, ax=ax1 ,line_kw={"color": "green"}
                ,kind='both',target=0)
                mlp_disp2 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx, ax=ax2, line_kw={"color": "orange"}
                ,kind='both',target=1)
                mlp_disp3 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx, ax=ax3, line_kw={"color": "red"}
                ,kind='both',target=2)
                ax1.set_title("Normoglycemia")
                ax2.set_title("Prediabetes")
                ax3.set_title("T2DM")
                plt.show()
            elif num_label==2:
                mlp_disp1 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx
                ,kind='both')
                if min_label==0:
                    plt.suptitle("Normoglycemia - T2DM")
                elif min_label==1:
                    plt.suptitle("Prediabetes - T2DM")
                plt.show()

        for i in range(num_label):
            id_to_explain = 211
            print(y_train.iloc[id_to_explain])
            shap.plots.waterfall(shap_values[id_to_explain,:,i])

    if save:
        import pickle
        with open('random_forest_pd.pkl', 'wb') as file:
            pickle.dump(best_clf[idx], file)



def train_xgboost(X_train, y_train, X_test, y_test, sampling=None, xai=False, save=False):
    idx=0
    xgb_clf = xgb.XGBClassifier()
    num_label=np.unique(y_train).shape[0]
    min_label=np.min(y_train)
    pipeline=make_pipeline(xgb_clf)
    param_grid = {
        'xgbclassifier__seed': [42],
        'xgbclassifier__n_estimators': [15],
        'xgbclassifier__max_depth': [7],
        'xgbclassifier__learning_rate': [0.2],
        'xgbclassifier__gamma': [0.1],
        'xgbclassifier__alpha': [0.1],
        'xgbclassifier__min_child_weight': [0],
        'xgbclassifier__subsample': [0.5],
        'xgbclassifier__colsample_bytree': [1],
#        'xgbclassifier__n_estimators': [5, 10, 15],
#        'xgbclassifier__max_depth': [3, 7, 10],
#        'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
#        'xgbclassifier__gamma': [0, 0.1, 0.2],
#        'xgbclassifier__alpha': [0, 0.1, 0.2],
#        'xgbclassifier__min_child_weight': [0, 0.1, 0.2],
#        'xgbclassifier__subsample': [0.5, 0.7, 0.9],
#        'xgbclassifier__colsample_bytree': [1],
    }
    if sampling=='under':
        idx=1
        # Instantiate the RandomUnderSampler
        undersampler = RandomUnderSampler(random_state=42)
        # Create a pipeline including the undersampling and classifier
        pipeline = make_pipeline(undersampler, xgb_clf)
    elif sampling=='over':
        idx=1
        # Instantiate the RandomUnderSampler
        oversampler = RandomOverSampler(random_state=42)
        # Create a pipeline including the undersampling and classifier
        pipeline = make_pipeline(oversampler, xgb_clf)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', verbose=3, n_jobs=-1, refit='f1_macro')
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best estimator
    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    predictions_train = best_clf.predict(X_train)
    predictions_test = best_clf.predict(X_test)

    # Evaluate the best model
    f1_train = sk.metrics.f1_score(y_train, predictions_train, average='macro')
    f1_test = sk.metrics.f1_score(y_test, predictions_test, average='macro')
    print(f"Best Model - F1 train: {f1_train:.2f} - F1 test: {f1_test:.2f}")
    print("Best Parameters:", best_params)
    print("TRAINING")
    print("Classification Report:")
    print(classification_report(y_train, predictions_train))
    average_specificity = 0
    for class_label in range(num_label):
        spec = specificity(y_train, predictions_train, class_label)
        average_specificity += spec
    average_specificity /= num_label
    print(f"Average Specificity: {average_specificity}")
    print("TESTING")
    print("Classification Report:")
    print(classification_report(y_test, predictions_test))
    average_specificity = 0
    for class_label in range(num_label):
        spec = specificity(y_test, predictions_test, class_label)
        average_specificity += spec
    average_specificity /= num_label
    print(f"Average Specificity: {average_specificity}")

    if xai:
        # Plot the feature importances
        importances = best_clf[idx].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()
        #shap
        '''
        explainer = shap.TreeExplainer(best_clf[idx], X_train)
        shap_values = explainer(X_train, check_additivity=False)
        shap_values_2 = explainer.shap_values(X_train, check_additivity=False)
        
        if num_label==2 and min_label==0:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['NG', 'T2DM'], feature_names=X_train.columns)
        elif num_label==2 and min_label==1:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['PD', 'T2DM'], feature_names=X_train.columns)
        else:
            shap.summary_plot(shap_values_2, X_train.values, plot_type="bar", class_names=['NG', 'PD', 'T2DM'], feature_names=X_train.columns)

        
        for i in range(num_label):
            shap.plots.beeswarm(shap_values[:,:,i])
        '''

        for i in range(1):
            if i==0:
                features_idx=[1,3,5,7]
            elif i==1:
                features_idx=[2,3,5]
            elif i==2:
                features_idx=[6,7,11]
            elif i==3:
                features_idx=[8,9,10]
            elif i==4:
                features_idx=[12,13,14]
            if num_label==3:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 4, figsize=(15, 15), sharey=True)
                mlp_disp1 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx, ax=ax1
                ,kind='both',target=0)
                mlp_disp2 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx, ax=ax2
                ,kind='both',target=1)
                mlp_disp3 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx, ax=ax3
                ,kind='both',target=2)
                ax1[0].get_legend().remove()
                ax1[0].set_title('LDL [mmol/L]', size=15)
                ax1[0].set_ylabel('Normoglycemia\nPartial Dependence', size=15)
                ax1[0].set(xlabel=None)
                ax1[0].set(xticklabels=[])
                ax2[0].get_legend().remove()
                ax2[0].set_ylabel('Prediabetes\nPartial Dependence', size=15)
                ax2[0].set(xlabel=None)
                ax2[0].set(xticklabels=[])
                ax3[0].get_legend().remove()
                ax3[0].set_ylabel('T2DM\nPartial Dependence', size=15)
                ax3[0].set(xlabel=None)
                ax1[1].get_legend().remove()
                ax1[1].set_title('HbA1c [%]', size=15)
                ax1[1].set(xlabel=None)
                ax1[1].set(ylabel=None)
                ax1[1].set(xticklabels=[])
                ax2[1].get_legend().remove()
                ax2[1].set(xlabel=None)
                ax2[1].set(ylabel=None)
                ax2[1].set(xticklabels=[])
                ax3[1].get_legend().remove()
                ax3[1].set(xlabel=None)
                ax3[1].set(ylabel=None)
                ax1[2].get_legend().remove()
                ax1[2].set_title('sBP [mmHg]', size=15)
                ax1[2].set(xlabel=None)
                ax1[2].set(ylabel=None)
                ax1[2].set(xticklabels=[])
                ax2[2].get_legend().remove()
                ax2[2].set(xlabel=None)
                ax2[2].set(ylabel=None)
                ax2[2].set(xticklabels=[])
                ax3[2].get_legend().remove()
                ax3[2].set(xlabel=None)
                ax3[2].set(ylabel=None)
                ax1[3].get_legend().remove()
                ax1[3].set_title('BMI [kg/m^2]', size=15)
                ax1[3].set(xlabel=None)
                ax1[3].set(ylabel=None)
                ax1[3].set(xticklabels=[])
                ax2[3].get_legend().remove()
                ax2[3].set(xlabel=None)
                ax2[3].set(ylabel=None)
                ax2[3].set(xticklabels=[])
                ax3[3].get_legend().remove()
                ax3[3].set(xlabel=None)
                ax3[3].set(ylabel=None)
                #ax1.set_title("Normoglycemia")
                #ax2.set_title("Prediabetes")
                #ax3.set_title("T2DM")
                plt.show()
            elif num_label==2:
                fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 15), sharey=True)
                mlp_disp1 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx[0:2], ax=ax1
                ,kind='both')
                mlp_disp2 = PartialDependenceDisplay.from_estimator(
                    best_clf[idx], X_test,features_idx[2:4], ax=ax2
                ,kind='both')
                ax1[0].get_legend().remove()
                ax1[0].set_title('LDL [mmol/L]', size=15)
                ax1[0].set(xlabel=None)
                ax1[0].set_ylabel('Partial Dependence', size=15)
                ax2[0].set_title('sBP [mmHg]', size=15)
                ax2[0].get_legend().remove()
                ax2[0].set_ylabel('Partial Dependence', size=15)
                ax2[0].set(xlabel=None)
                ax1[1].get_legend().remove()
                ax1[1].set_title('HbA1c [%]', size=15)
                ax1[1].set(ylabel=None)
                ax1[1].set(xlabel=None)
                ax2[1].set_title('BMI [kg/m^2]', size=15)
                ax2[1].get_legend().remove()
                ax2[1].set(ylabel=None)
                ax2[1].set(xlabel=None)
                plt.show()

        for j in range(len(predictions_train)):
            if y_train.iloc[j]!=predictions_train[j] and y_train.iloc[j]==0:# and X_train.iloc[j]['fbs']>=5.6 and X_train.iloc[j]['hba1c']>=5.7:
                print(predictions_train[j], y_train.iloc[j])
                shap.plots.waterfall(shap_values[j])

    if save:
        import pickle
        with open('xgboost.pkl', 'wb') as file:
            pickle.dump(best_clf[idx], file)


def train_bagging(X_train, y_train, X_test, y_test, sampling=None, save=False):
    idx=0
    single_clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf = BaggingClassifier()
    num_label=np.unique(y_train).shape[0]
    min_label=np.min(y_train)
    pipeline=make_pipeline(clf)
    param_grid = {
    'baggingclassifier__random_state': [42],
#    'baggingclassifier__n_estimators': [5],
#    'baggingclassifier__max_samples': [1.0],
#    'baggingclassifier__max_features': [1.0],
#    'baggingclassifier__estimator': [single_clf]
    'baggingclassifier__n_estimators': [5, 7, 10, 50, 100],
    'baggingclassifier__max_samples': [0.5, 0.7, 1.0, 1.25, 1.5],
    'baggingclassifier__max_features': [0.5, 0.7, 1.0, 1.25, 1.5],
    'baggingclassifier__estimator': [single_clf]
    }
    if sampling=='under':
        idx=1
        # Instantiate the RandomUnderSampler
        undersampler = RandomUnderSampler(random_state=42)
        # Create a pipeline including the undersampling and classifier
        pipeline = make_pipeline(undersampler, clf)
    elif sampling=='over':
        idx=1
        # Instantiate the RandomUnderSampler
        oversampler = RandomOverSampler(random_state=42)
        # Create a pipeline including the undersampling and classifier
        pipeline = make_pipeline(oversampler, clf)
    
    # Create GridSearchCV instance
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', verbose=3, n_jobs=-1, refit='f1_macro')

    # Fit the GridSearchCV to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best estimator
    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    predictions_train = best_clf.predict(X_train)
    predictions_test = best_clf.predict(X_test)

    # Evaluate the best model
    f1_train = sk.metrics.f1_score(y_train, predictions_train, average='macro')
    f1_test = sk.metrics.f1_score(y_test, predictions_test, average='macro')
    print(f"Best Model - F1 train: {f1_train:.2f} - F1 test: {f1_test:.2f}")
    print("Best Parameters:", best_params)
    print("TRAINING")
    print("Classification Report:")
    print(classification_report(y_train, predictions_train))
    average_specificity = 0
    for class_label in range(num_label):
        spec = specificity(y_train, predictions_train, class_label)
        average_specificity += spec
    average_specificity /= num_label
    print(f"Average Specificity: {average_specificity}")
    print("TESTING")
    print("Classification Report:")
    print(classification_report(y_test, predictions_test))
    average_specificity = 0
    for class_label in range(num_label):
        spec = specificity(y_test, predictions_test, class_label)
        average_specificity += spec
    average_specificity /= num_label
    print(f"Average Specificity: {average_specificity}")

    if save:
        import pickle
        with open('bagging.pkl', 'wb') as file:
            pickle.dump(best_clf[idx], file)