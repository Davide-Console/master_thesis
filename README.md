# Exploring Prediabetes Pathways: Using Machine Learning and Counterfactual Explanations for Type 2 Diabetes Prediction and Prevention

Console D., Lenatti M., Simeone D., Keshavjee K., Guergachi A., Mongelli M., Paglialonga A., “Exploring Prediabetes Pathways Using Explainable AI on Data from Electronic Medical Records,” Proceedings of the 34th Medical Informatics Europe Conference (EFMI MIE 2024), Aug 25-29, 2024, Athens, Greece. Studies in Health Technology and Informatics, 2024. In press.

## Introduction
Type 2 Diabetes Mellitus (T2DM) is a chronic metabolic disorder characterized by hyperglycemia due to impaired insulin secretion. Early detection and personalized intervention are crucial to reducing the risk of T2DM and associated healthcare costs. Prediabetes (PD) is a reversible state that precedes T2DM, making it an important focus for early intervention.

This work leverages Machine Learning (ML) and counterfactual explanations to predict and prevent transitions between normoglycemia (NG), PD, and T2DM using Electronic Medical Record (EMR) data.

## Materials & Methods

### Dataset Extraction
Data was extracted from the Canadian Primary Care Sentinel Surveillance Network (CPCSSN). Features included were blood exams, glycemic biomarkers, general health indicators and presence of comorbidities.

### Dataset Characterization
Univariate and bivariate analyses were conducted, including the Shapiro-Wilk test and Spearman’s correlation matrix. The separability of the target variable was assessed using boxplots and statistical tests.

### Model Training and Evaluation
Four models were tested: Decision Tree, Random Forest, XGBoost, and Bagging of Logistic Regressions. Models were trained using stratified 5-fold cross-validation on scaled data. Total population, subgroup of patients with NG and subgroup of patients with PD were separately analysed. XAI techniques, such as feature importance and partial dependence plots (PDPs), were used to understand model predictions.

### Counterfactual Explainability
Counterfactual explanations were generated using [DiCE](https://github.com/interpretml/DiCE) (Diverse Counterfactual Explanations) to determine the minimal changes needed for a patient with PD to regress to NG and prevent T2DM. Random and genetic search methods were tested.

## Results

### Dataset Characterization
Non-Gaussian distributions were observed for all features. Spearman’s correlation matrix showed low inter-correlation of features, except for total cholesterol and LDL. Discriminable distributions were found between the target classes, particularly for transitions ending in T2DM.

The similarity in feature ranges between PD and NG patients indicates the complexity of predicting PD. High correlation was found between some features, like FBS and HbA1c. LDL showed counter-intuitive decreases, likely due to medication.

### Model Performance
The XGB model performed best on the total population and CurrentState=PD subgroup. For the CurrentState = NG subgroup, no model achieved satisfactory performance due to the imbalance of the dataset. Glycemic biomarkers, BMI, and LDL were identified as the most important features, as can be seen from the feature importance and PDP graphs.

| Population          | Model | F1Macro | Sensitivity | Specificity |
|---------------------|-------|---------|-------------|-------------|
| Total population    | XGB   | 83%     | 86%         | 90%         |
| CurrentState = PD   | XGB   | 81%     | 76%         | 86%         |
| CurrentState = NG   | DT    | 58%     | 13%         | 99%         |

<p align="center"> FI PDP </p>

### Counterfactual Explainability
Counterfactuals for transitions from PD to T2DM highlighted the importance of improving glycemic biomarkers and BMI to reduce T2DM risk. The random method generated fewer counterfactuals per record compared to the genetic method, which changed more features but produced fewer outliers.

| Metric                   | Random Method     | Genetic Method     |
|--------------------------|-------------------|--------------------|
| Availability [%]         | 100               | 100                |
| Mean number of CFs       | 10/10             | 10/10              |
| Features changed per CF  | 1.63 (0.49)       | 6.80 (0.78)        |
| Outliers on all features | 49%               | 10%                |

<p align="center"> SP </p>

## Conclusions
This work contributes to understanding transitions between glycemic states using ML on primary care data, focusing on prediabetes for early intervention. Counterfactual explanations provide actionable insights for personalized prevention plans.





</br>
</br>

<details>
  <summary>Instructions</summary>
  This work is not reproducible due to fact that the CPCSSN is not a public database and therefore our dataset cannot be legally shared.
</details>
