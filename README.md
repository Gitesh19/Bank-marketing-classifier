# Bank Marketing – Term Deposit Subscription Classifier

**BITS Pilani WILP · M.Tech (AIML) · Machine Learning · Assignment 2**
**Student:** Gitesh Sawant

🔗 **Live Streamlit app:** <PASTE-YOUR-STREAMLIT-URL-HERE>
🔗 **GitHub repository:** <PASTE-YOUR-GITHUB-URL-HERE>

---

## a. Problem statement

A Portuguese bank runs direct-marketing phone campaigns to sell term deposits.
Every call costs agent time, so the bank wants to know **in advance which clients
are likely to subscribe** so that it can prioritise them.

This is a **binary classification** problem: given a client's demographic,
loan-history, campaign-contact and macro-economic attributes, predict whether
the client will subscribe to a term deposit (`y = yes` → 1, `y = no` → 0).

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours,
Gaussian Naive Bayes and Support Vector Machine) are trained on the same data
and compared using Accuracy, AUC, Precision, Recall, F1 and MCC. A Streamlit
web app lets a user upload test data, pick a model and inspect the metrics,
confusion matrix and classification report.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository – *Bank Marketing* (Moro, Cortez & Rita, 2014) – file `bank-additional-full.csv` |
| Link | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Instances | **41,188** (≥ 500 ✔) |
| Input features used | **19** (≥ 12 ✔) – 10 numeric + 9 categorical |
| Target | `y` – did the client subscribe to a term deposit? (`yes` / `no`) |
| Class balance | 36,548 *no* (88.7 %) vs 4,640 *yes* (11.3 %) → **imbalanced** |
| Missing values | none (unknown categories are encoded as the literal `"unknown"`) |

**Features**

| Group | Columns |
|---|---|
| Client data | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Last contact | `contact`, `month`, `day_of_week` |
| Campaign | `campaign`, `pdays`, `previous`, `poutcome` |
| Macro-economic context | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` |

The original file has a 20th feature, `duration` (call length in seconds). It
was **deliberately dropped** because it is only known after the call is over
and almost perfectly reveals the outcome (data leakage); the UCI page itself
recommends removing it for a realistic model.

**Pre-processing** (done inside each saved `sklearn.Pipeline`, so the app never
has to repeat it): `StandardScaler` on numeric columns, `OneHotEncoder` on
categorical columns, 80 / 20 stratified train–test split (`random_state = 42`).
The 20 % hold-out split (8,238 rows) is saved as `test_data.csv`.

## c. GitHub repository link

<PASTE-YOUR-GITHUB-URL-HERE>

```
bank-marketing-classifier/
├── app.py               # Streamlit web application
├── train_models.py      # trains the 5 models, saves pipelines + metrics
├── requirements.txt
├── README.md
├── test_data.csv        # 8,238-row hold-out set used by the app
├── data/
│   └── bank-additional-full.csv   # full raw dataset (41,188 rows)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.json     # test-set metrics of all five models
```

## d. Models used – comparison table

All five models were trained on the identical 32,950-row training split and
evaluated on the identical 8,238-row test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8350 | **0.8009** | 0.3679 | **0.6466** | 0.4689 | 0.4011 |
| Decision Tree | 0.8314 | 0.7921 | 0.3599 | 0.6379 | 0.4602 | 0.3906 |
| KNN | **0.8949** | 0.7553 | **0.5708** | 0.2694 | 0.3660 | 0.3434 |
| Naive Bayes | 0.8580 | 0.7764 | 0.3910 | 0.4677 | 0.4259 | 0.3474 |
| Support Vector Machine | 0.8577 | 0.7828 | 0.4140 | 0.6325 | **0.5004** | **0.4346** |

Hyper-parameters: LR `max_iter=2000, class_weight="balanced"` · DT
`max_depth=8, min_samples_leaf=20, class_weight="balanced"` · KNN
`n_neighbors=15, weights="distance"` · GaussianNB `var_smoothing=1e-3` · SVM
`kernel="rbf", C=1, gamma="scale", class_weight="balanced", probability=True`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best **AUC (0.80)** and best **Recall (0.65)**: with balanced class weights it catches almost two-thirds of the real subscribers. Precision is low (0.37) because it also flags many non-subscribers, but for a marketing use-case (cheap false positive, expensive missed customer) this trade-off is attractive. Fast to train and easy to interpret. |
| Decision Tree | Very close to Logistic Regression on every metric (AUC 0.79, Recall 0.64) with a depth-8 tree. Its splits are mainly on `nr.employed`, `euribor3m`, `pdays` and `poutcome`, which matches domain intuition (economic climate and previous campaign success). Slightly lower MCC indicates a little more over-fitting than LR. |
| KNN | **Highest Accuracy (0.89)** and highest Precision (0.57) but these numbers are misleading: it has the **lowest Recall (0.27)**, AUC and MCC. The model mostly predicts the majority class "no", so the accuracy is close to the 88.7 % baseline. Distance-based voting in a 50-dimension one-hot space is dominated by the many "no" neighbours; it also has the largest model file (stores the whole training set). |
| Naive Bayes | Middle of the pack (AUC 0.78, F1 0.43). Its independence assumption is violated (e.g. `euribor3m`, `emp.var.rate` and `nr.employed` are strongly correlated), and Gaussian likelihoods are a poor fit for one-hot columns, which caps its performance. It is however the fastest model to train (< 1 s) and needs no scaling. |
| Support Vector Machine | **Best F1 (0.50) and best MCC (0.43)** – the most balanced trade-off between precision (0.41) and recall (0.63). The RBF kernel captures non-linear interactions that LR misses. Cost: by far the slowest model (minutes to train on 33 k rows because `probability=True` runs internal cross-validation). |
| **Overall, Winner for your dataset?** | **Support Vector Machine** is the overall winner: it gives the highest F1 and MCC, the two metrics that are most meaningful on an imbalanced dataset, while keeping recall almost as high as Logistic Regression. If training time or interpretability matter more, **Logistic Regression** is the practical runner-up with the best AUC and recall. Plain accuracy should not be used to choose a model here – KNN "wins" on accuracy only by predicting "no" for nearly everyone. |

## Streamlit app features

* 📂 **CSV upload** of test data (or use the bundled `test_data.csv`)
* 🔽 **Model selection dropdown** for the five classifiers
* 📊 **Accuracy, AUC, Precision, Recall, F1, MCC** of the chosen model
* 🟦 **Confusion matrix** heat-map, **ROC curve** and full **classification report**
* 🏁 optional side-by-side comparison of all five models on the uploaded data
* ⬇️ download of predictions as CSV

Models are loaded from `model/*.joblib`; **no re-training** occurs in the app.

## How to run locally

```bash
git clone <PASTE-YOUR-GITHUB-URL-HERE>
cd bank-marketing-classifier
pip install -r requirements.txt
python train_models.py        # optional – models are already committed
streamlit run app.py
```

## Reference

Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the
Success of Bank Telemarketing.* Decision Support Systems, 62, 22-31.
