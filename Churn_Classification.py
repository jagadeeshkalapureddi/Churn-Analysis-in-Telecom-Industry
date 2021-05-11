# ### `IMPORT REQUIRED PACKAGES(LIBRARIES)`

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from matplotlib.cm import get_cmap
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
# import statsmodels.api as sm
# from sklearn.metrics import mean_squared_error
# from statsmodels.stats.stattools import durbin_watson
# from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[203]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# ### `READ THE DATASET & MEMORY OPTIMIZATION`

# In[204]:


print('*' * 80)
print('churn Dataset')
df = import_data('churn.csv')
print('*' * 80)


# ### `DATA UNDERSTANDING`

# In[205]:


print('First Five Records of the Dataset')
df.head()


# `Change the Index with Car Names & Check for the Shape`

# In[206]:


df = df.set_index('customerID')
print("The Data Frame having the Rows of '{}' and Columns of '{}'".format (df.shape[0],df.shape[1]))
df.head()


# `Check for the Detailed Information of the Dataset`

# In[207]:


df.info()


# `Check for the Null values in the Dataset`

# In[208]:


df.isnull().sum()


# In[209]:


df['SeniorCitizen'] = np.where(df['SeniorCitizen'] > 0.5, 1, 0)


# ### Senior Citizen

# In[210]:


df['SeniorCitizen'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[211]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['SeniorCitizen'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('SeniorCitizen', fontsize = 20)
plt.title('Barplot_ '+ 'SeniorCitizen', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### Gender

# In[212]:


df['gender'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[213]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['gender'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('gender', fontsize = 20)
plt.title('Barplot_ '+ 'gender', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### Partner

# In[214]:


df['Partner'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[215]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['Partner'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('Partner', fontsize = 20)
plt.title('Barplot_ '+ 'Partner', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### Dependents

# In[216]:


df['Dependents'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[217]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['Dependents'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('Dependents', fontsize = 20)
plt.title('Barplot_ '+ 'Dependents', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### CallService

# In[218]:


df['CallService'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[219]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['CallService'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('CallService', fontsize = 20)
plt.title('Barplot_ '+ 'CallService', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### MultipleConnections

# In[220]:


df['MultipleConnections'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[221]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['MultipleConnections'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('MultipleConnections', fontsize = 20)
plt.title('Barplot_ '+ 'MultipleConnections', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### InternetConnection

# In[222]:


df['InternetConnection'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[223]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['InternetConnection'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('InternetConnection', fontsize = 20)
plt.title('Barplot_ '+ 'InternetConnection', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### OnlineSecurity

# In[224]:


df['OnlineSecurity'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[225]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['OnlineSecurity'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('OnlineSecurity', fontsize = 20)
plt.title('Barplot_ '+ 'OnlineSecurity', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### OnlineBackup

# In[226]:


df['OnlineBackup'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[227]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['OnlineBackup'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('OnlineBackup', fontsize = 20)
plt.title('Barplot_ '+ 'OnlineBackup', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### DeviceProtectionService

# In[228]:


df['DeviceProtectionService'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[229]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['DeviceProtectionService'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('DeviceProtectionService', fontsize = 20)
plt.title('Barplot_ '+ 'DeviceProtectionService', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### TechnicalHelp

# In[230]:


df['TechnicalHelp'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[231]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['TechnicalHelp'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('TechnicalHelp', fontsize = 20)
plt.title('Barplot_ '+ 'TechnicalHelp', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### OnlineTV

# In[232]:


df['TechnicalHelp'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[233]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['TechnicalHelp'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('TechnicalHelp', fontsize = 20)
plt.title('Barplot_ '+ 'TechnicalHelp', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### OnlineMovies

# In[234]:


df['OnlineMovies'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[235]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['OnlineMovies'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('OnlineMovies', fontsize = 20)
plt.title('Barplot_ '+ 'OnlineMovies', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### Agreement

# In[236]:


df['Agreement'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[237]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['Agreement'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('Agreement', fontsize = 20)
plt.title('Barplot_ '+ 'Agreement', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### BillingMethod

# In[238]:


df['BillingMethod'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[239]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['BillingMethod'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('BillingMethod', fontsize = 20)
plt.title('Barplot_ '+ 'BillingMethod', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### PaymentMethod

# In[240]:


df['PaymentMethod'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[241]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['PaymentMethod'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('PaymentMethod', fontsize = 20)
plt.title('Barplot_ '+ 'PaymentMethod', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### Churn

# In[242]:


df['Churn'].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'


# In[243]:


name = "Dark2"
cmap = get_cmap(name)
colors = cmap.colors
df['Churn'].value_counts().plot(kind = 'bar', color = colors)
plt.xlabel('Churn', fontsize = 20)
plt.title('Barplot_ '+ 'Churn', fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1, top=1.2)


# ### `Feature Engineering`

# In[244]:


df_categorical = df.select_dtypes(include=['category'])
df_categorical.head()


# In[245]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()


# In[246]:


df = df.drop(df_categorical.columns,axis=1)
df = pd.concat([df,df_categorical],axis=1)
df.head()


# ### `Decision Tree - Model Building`

# In[247]:


# Function For Summary of classification model

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='darkblue', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkorange', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def get_summary(y_test,predicted):
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test,predicted)
    TP = conf_mat[0,0:1]
    FP = conf_mat[0,1:2]
    FN = conf_mat[1,0:1]
    TN = conf_mat[1,1:2]
    
    accuracy = (TP+TN)/((FN+FP)+(TP+TN))
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall =  TP / (TP + FN)
    fScore = (2 * recall * precision) / (recall + precision)
    auc = roc_auc_score(y_test, predicted)

    print("Confusion Matrix:\n",conf_mat)
    print("Accuracy:",accuracy)
    print("Sensitivity :",sensitivity)
    print("Specificity :",specificity)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-score:",fScore)
    print("AUC:",auc)
    print('\n')
    print("ROC curve:")
    fpr, tpr, thresholds = roc_curve(y_test, predicted)
    plot_roc_curve(fpr, tpr)


# #### `Split the data as 'x' and 'y'`

# In[248]:


x = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'CallService', 'MultipleConnections', 
        'InternetConnection', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtectionService', 'TechnicalHelp', 
        'OnlineTV', 'OnlineMovies', 'Agreement', 'BillingMethod', 'PaymentMethod', 'MonthlyServiceCharges', 
        'TotalAmount']]
y = df['Churn']


# `Check for the first five records of the Dataset of 'x' Variable`

# In[249]:


x.head()


# `Check for the first five records of the Dataset of 'y' Variable`

# In[250]:


y.head()


# In[251]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=100)

x_train.head()


# In[252]:


from sklearn.tree import DecisionTreeClassifier
dt_default = DecisionTreeClassifier()
dt_default.fit(x_train,y_train)


# In[253]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
y_pred_default = dt_default.predict(x_test)
print(classification_report(y_test,y_pred_default))


# In[254]:


get_summary(y_test, y_pred_default)


# In[255]:


# Importing required packages for visualization
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus,graphviz

# Putting features
features = list(df.columns[0:-1])
features


# In[256]:


# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(dt_default, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[257]:


# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(1, 50)}

# instantiate the model
dtree = DecisionTreeClassifier()

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# In[258]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[259]:


# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[260]:


# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(1, 200, 10)}

# instantiate the model
dtree = DecisionTreeClassifier()

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# In[261]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[262]:


# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[263]:


# GridSearchCV to find optimal min_samples_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(1, 200, 10)}

# instantiate the model
dtree = DecisionTreeClassifier()

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# In[264]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[265]:


# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[266]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(1, 15, 5),
    'min_samples_leaf': range(1, 25, 10),
    'min_samples_split': range(1, 25, 10),
    'criterion': ['gini', "entropy"]
}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1)

# Fit the grid search to the data
grid_search.fit(x_train,y_train)


# In[267]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.head(3)


# In[268]:


# printing the optimal accuracy score and hyperparameters
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[269]:


# model with optimal hyperparameters
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=11, 
                                  #min_samples_leaf=1,
                                  min_samples_split=11)
clf_gini.fit(x_train, y_train)


# In[270]:


y_pred_p = clf_gini.predict(x_test)
print(classification_report(y_test, y_pred_p))


# In[271]:


# accuracy score
clf_gini.score(x_test,y_test)


# In[272]:


# plotting the tree
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[274]:


get_summary(y_test, y_pred_p)


# In[275]:


#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, y_pred_p)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

ppv = cm1[0,0]/(cm1[0,0]+cm1[1,1])
print('Positive Predictive Value :', ppv)

npv = cm1[1,1]/(cm1[1,0]+cm1[0,1])
print('Negetive Predictive Value :', npv)


# ### `Random Forest - Model Building`

# #### `Split the data as 'x' and 'y'`

# In[276]:


x = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'CallService', 'MultipleConnections', 
        'InternetConnection', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtectionService', 'TechnicalHelp', 
        'OnlineTV', 'OnlineMovies', 'Agreement', 'BillingMethod', 'PaymentMethod', 'MonthlyServiceCharges', 
        'TotalAmount']]
y = df['Churn']


# `Check for the first five records of the Dataset of 'x' Variable`

# In[277]:


x.head()


# `Check for the first five records of the Dataset of 'y' Variable`

# In[278]:


y.head()


# In[279]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=100)

x_train.head()


# In[280]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)


# In[281]:


y_pred=rf.predict(x_test)


# In[282]:


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[283]:


# classification metrics
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_pred))


# In[284]:


#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, y_pred)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

ppv = cm1[0,0]/(cm1[0,0]+cm1[1,1])
print('Positive Predictive Value :', ppv)

npv = cm1[1,1]/(cm1[1,0]+cm1[0,1])
print('Negetive Predictive Value :', npv)


# ### `Pruning`

# In[285]:


from sklearn.model_selection import GridSearchCV

param_grid = [
{'n_estimators': [10, 20,30,40,50], 'max_features': [5, 10,15,20,25], 
 'max_depth': [10,20,30,40,50, None], 'bootstrap': [True, False]}
]

grid_search_forest = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_forest.fit(x_train, y_train)


# In[286]:


grid_search_forest.best_estimator_


# In[292]:


rcf = RandomForestClassifier(bootstrap=False, max_depth=40, max_features=10,
                       n_estimators=40)
rcf.fit(x_train, y_train)


# In[293]:


y_predd = rcf.predict(x_test)


# In[294]:


get_summary(y_test, y_predd)


# In[295]:


# classification metrics
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_predd))


# In[296]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# # Boosting

# ### `Ada Boost Classifier`

# In[297]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
# base estimator
tree = DecisionTreeClassifier()

# adaboost with the tree as base estimator
adaboost_model_1 = AdaBoostClassifier(
    base_estimator=tree,
    #n_estimators=600,
    learning_rate=0.5,
    algorithm="SAMME")


# In[298]:


# fit
adaboost_model_1.fit(x_train, y_train)


# In[299]:


predictions = adaboost_model_1.predict(x_test)


# In[300]:


confusion_matrix(y_test, predictions)


# In[301]:


print(classification_report(y_test, predictions))


# In[302]:


get_summary(y_test, predictions)


# ## Gradient Boosting for Classification

# In[310]:


# parameter grid
param_grid = {"learning_rate": [0.2, 0.6, 0.9],
              "subsample": [0.3, 0.6, 0.9]
             }
GBC = GradientBoostingClassifier()


# In[311]:


# run grid search
folds = 3
grid_search_GBC = GridSearchCV(GBC, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               return_train_score=True,                         
                               verbose = 1)

grid_search_GBC.fit(x_train, y_train)


# In[312]:


predictions = grid_search_GBC.predict(x_test)
confusion_matrix(y_test, predictions)


# In[313]:


print(classification_report(y_test, predictions))


# In[314]:


get_summary(y_test, predictions)


# ## XG Boost

# In[315]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train,y_train)


# In[316]:


predictions = model.predict(x_test)


# In[317]:


confusion_matrix(y_test, predictions)


# In[318]:


print(classification_report(y_test, predictions))


# In[319]:


get_summary(y_test, predictions)


# # CatBoost for Classification

# In[320]:


from catboost import CatBoostClassifier
cat_model = CatBoostClassifier()
cat_model.fit(x_train, y_train)


# In[321]:


predictions =cat_model.predict(x_test)
confusion_matrix(y_test, predictions)


# In[322]:


print(classification_report(y_test, predictions))


# In[323]:


get_summary(y_test, predictions)


# ## Light GBM

# In[324]:


from lightgbm import LGBMClassifier
model = LGBMClassifier()
model.fit(x_train, y_train)


# In[325]:


predictions =model.predict(x_test)
confusion_matrix(y_test, predictions)


# In[326]:


print(classification_report(y_test, predictions))


# In[327]:


get_summary(y_test, predictions)

