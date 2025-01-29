#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.duplicated().sum()


# In[5]:


#cek nol value

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
missing = df.applymap(lambda x:x == 0)
missing_value = missing.sum()
print('Nol value:\n', missing_value)


# In[6]:


#cek NaN

check_nan = df.isna().any().any()
total_nan = df.isna().sum()
with_nan = total_nan[total_nan >0]
print('Column with NaN:\n', with_nan)


# In[7]:


cols_to_drop = ['Unnamed: 0',
                'id',
                'member_id',
                'emp_title',
                'url','desc',
                'purpose',
                'title',
                'zip_code',
                'delinq_2yrs',
                'inq_last_6mths',
                'mths_since_last_delinq',
                'mths_since_last_record',
                'pub_rec',
                'collections_12_mths_ex_med',
                'annual_inc_joint','dti_joint',
                'verification_status_joint',
                'acc_now_delinq',
                'open_acc_6m',
                'open_il_6m',
                'open_il_12m',
                'open_il_24m',
                'mths_since_rcnt_il',
                'total_bal_il',
                'il_util',
                'open_rv_12m',
                'open_rv_24m',
                'max_bal_bc',
                'all_util',
                'inq_fi',
                'total_cu_tl',
                'inq_last_12m',
                'total_rev_hi_lim',
                'mths_since_last_major_derog',
                'next_pymnt_d',
                'issue_d',
                'recoveries',
                'collection_recovery_fee',
                'tot_coll_amt',
                'tot_cur_bal',
                'last_credit_pull_d',
                'total_rec_late_fee']
df = df.drop(cols_to_drop, axis=1)


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.head()


# In[11]:


df.isnull().sum()


# In[13]:


verif_counts = df['verification_status'].value_counts(normalize=True) * 100
loan_counts = df['loan_status'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

#verif status visualization
verif_counts.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Percentage of Verification Status')
axes[0].set_xlabel('Verification Status')
axes[0].set_ylabel('Percentage (%)')

#loan status visualization
loan_counts.plot(kind='bar', ax=axes[1], color='skyblue')
axes[1].set_title('Percentage of Loan Status')
axes[1].set_xlabel('Loan Status')
axes[1].set_ylabel('Percentage (%)')

plt.tight_layout()
plt.show()


# In[31]:


plt.figure(figsize=(6,4))
sns.histplot(df['loan_amnt'], bins=30, kde=True, color='blue')
plt.title('Distribution of Loan Amounts')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()


# In[14]:


#emp_length

df['emp_length'] = df['emp_length'].str.extract(r'(\d+)') #extract number
df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce') #change to numeric


# In[15]:


med_emp = df['emp_length'].median()
df['emp_length'].fillna(med_emp, inplace=True)


# In[16]:


df.dropna(subset=['earliest_cr_line','last_pymnt_d'], inplace=True)


# In[17]:


#handling missing value
med_revol = df['revol_util'].median()
df['revol_util'].fillna(med_revol, inplace=True)


# In[18]:


df.drop(columns=['addr_state'], inplace=True)


# In[19]:


df.isnull().sum()


# In[20]:


#term

df['term_num'] = df['term'].str.extract(r'(\d+)', expand=False).astype(int) #mengambil angka
df.drop(columns=['term'], inplace=True)


# In[21]:


#convert to date time

from datetime import datetime

df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
df['earliest_cr_line_mnth'] = (datetime.now() - df['earliest_cr_line']).dt.days // 30

df.drop(['last_pymnt_d','earliest_cr_line'], axis=1, inplace=True)


# In[22]:


from sklearn.preprocessing import LabelEncoder

le_grade = LabelEncoder() #for grade
df['grade'] = le_grade.fit_transform(df['grade'])

le_sub_grade = LabelEncoder()
df['sub_grade'] = le_sub_grade.fit_transform(df['sub_grade'])


# In[23]:


df =  pd.get_dummies(df, columns=['home_ownership', 'pymnt_plan','initial_list_status','application_type','verification_status'], dtype=int)


# In[24]:


# loan status

label_map = {  #change to bad and good
    'Fully Paid':'Good',
    'Charged Off':'Bad',
    'Current':'Good',
    'Default':'Bad',
    'Late (31-120 days)':'Bad',
    'In Grace Period':'Good',
    'Late (16-30 days)':'Bad',
    'Does not meet the credit policy. Status:Fully Paid':'Bad',
    'Does not meet the credit policy. Status:Charged Off':'Bad'
}
#map the loan
df['loan_status'] = df['loan_status'].map(label_map)

df['loan_status'] = df['loan_status'].map({
    'Good':1,
    'Bad':0
})


# In[30]:


plt.figure(figsize=(8,5))
sns.countplot(x='grade', hue='loan_status', data=df, palette='coolwarm')
plt.title('Distribution Loan Status on Grade')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.legend(title='Loan Status')
plt.show()


# In[25]:


#correlation of variable

correlation_matrix = df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=False, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[26]:


#cek outlier

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot {col}')
    plt.show()


# In[36]:


from scipy.stats.mstats import winsorize

for col in numerical_cols:
    df[col] = winsorize(df[col], limits=[0.01,0.01]) #1% triming up and down

df['loan_status'].value_counts()

loan_status_counts = df['loan_status'].value_counts(normalize=True) * 100
loan_status_counts.index = ['Good(1)','Bad(0)']
plt.figure(figsize=(8,6))
plt.pie(loan_status_counts, labels=loan_status_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of Loan Status')
plt.axis('equal')
plt.show()


# ## Model and Evaluation

# In[28]:


df.fillna(df.median(), inplace=True) #isi missing value jika masih ada

X = df.drop('loan_status', axis=1)
y = df['loan_status']  #target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)
}

#hyperparameter grids
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10]
    },
    'Random Forest': {
        'n_estimators': [100,200],
        'max_depth': [None, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
}

#Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, model in models.items():
    if model_name in ['Logistic Regression', 'Random Forest']:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        param_grid={f'model__{k}': v for k, v in param_grids[model_name].items()}
    else:
        # no scaling for xgboost
        pipeline = model
        param_grid = param_grids[model_name]
        
    #grid search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1
    )
    if model_name == 'XGBoost':
        grid_search.fit(X_train, y_train)
    else:
        grid_search.fit(X_train_scaled, y_train)
        
    #best model
    best_model = grid_search.best_estimator_
    
    #predicition
    if model_name == 'XGBoost':
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
    else:
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = best_model.predict(X_test_scaled)
    
    #evaluation
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    #results
    results[model_name] = {
        'Best Parameters': grid_search.best_params_,
        'ROC-AUC': roc_auc,
        'Classification Report': report
    }

for model_name, metrics in results.items():
    print(f'Model: {model_name}')
    print(f"Best Parameters: {metrics['Best Parameters']}")
    print(f"ROC-AUC: {metrics['ROC-AUC']}")
    print(f"Classification Report:\n{metrics['Classification Report']}")


# In[32]:


pip install shap


# In[33]:


import shap


# In[34]:


xgb_model = XGBClassifier(
    learning_rate=0.3,
    max_depth=5,
    n_estimators=200,
    random_state=42
)
xgb_model.fit(X_train, y_train)


# In[35]:


explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)


# In[ ]:




