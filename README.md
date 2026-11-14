# Credit Card Fraud Detection — 3MTT Capstone

This repository contains a credit-card fraud classification project using the anonymised European card transaction dataset.

## Improved, executed workflow

The improved notebook is **[Credit_Card_Fraud_Detection_Improved_Executed.ipynb](Credit_Card_Fraud_Detection_Improved_Executed.ipynb)**. It includes the complete model-training workflow, held-out test evaluation, feature-importance analysis and optional SHAP explanations.

The workflow:
- inspects missing values, class balance and exact duplicates;
- removes exact duplicate records before splitting;
- uses stratified train, validation and test partitions;
- fits preprocessing only within training pipelines;
- compares Logistic Regression with SMOTE against a class-weighted Random Forest;
- selects the model using validation Average Precision (PR-AUC);
- tunes a decision threshold on validation data using F2;
- evaluates once on the untouched test set;
- saves the selected model and threshold and demonstrates inference;
- analyses permutation feature importance on a separate validation sample;
- provides optional SHAP TreeExplainer explanations for the selected Random Forest.

## Dataset

Expected filename: `creditcard.csv`.

The dataset is **not included in this repository**. Download the public dataset separately and place it in the notebook's expected location. In Google Colab, upload it to `/content/creditcard.csv`, or edit `DATA_PATH` in the setup cell to point to your local copy.

## Verified execution results

Execution used 284,807 original records (492 fraud cases), with no missing values. The workflow identified and removed 1,081 exact duplicate rows, leaving 283,726 records (473 fraud cases).

After stratified splitting:
- Training: 170,235 records (284 fraud cases)
- Validation: 56,745 records (94 fraud cases)
- Test: 56,746 records (95 fraud cases)

The validation-selected model was **Random Forest (class-weighted)**. The validation-selected threshold was **0.193788** (maximising F2).

On the untouched test set, at that threshold:
- Average Precision (PR-AUC): **0.8529**
- ROC-AUC: **0.9707**
- Precision: **0.8864**
- Recall: **0.8211**
- F1: **0.8525**
- F2: **0.8333**
- Confusion matrix (TN, FP, FN, TP): **56,641; 10; 17; 78**

For comparison, at the default 0.50 threshold, test precision was 0.9595 and recall was 0.7474. The lower selected threshold detected more fraud cases at the cost of more false alerts.

## Model analysis

Permutation importance was computed using Average Precision on a reproducible 10,000-row validation sample (3 repeats), not on the final test set. The most influential features in this run were:
1. `V4` — mean AP decrease **0.084781**
2. `V14` — **0.023661**
3. `V11` — **0.017915**
4. `V9` — **0.016212**
5. `V18` — **0.015895**

These are model-specific predictive importance estimates, not causal findings. `V1`–`V28` are anonymised PCA components, so their underlying real-world meanings cannot be inferred from the available dataset.

The notebook also includes a SHAP summary plot calculated with TreeExplainer on 500 validation transactions. SHAP explains the fitted model's predictions; it does not establish causal relationships.

## Reproducibility and limitations

- The notebook contains 15 executed code cells and no execution errors.
- The model artifact is saved by the notebook as `credit_card_fraud_model.joblib`; it is not committed here.
- The dataset and saved model binary are not committed.
- Results are specific to this dataset, split, model configuration and threshold. They do not establish production readiness or guarantee performance on new transaction populations.
- The decision threshold should be selected with the operational costs of false negatives and false positives in mind.
