#!/usr/bin/env python
# coding: utf-8

# ## Breast cancer cell classification using Weighted K-Nearest Neighbor classifier. Use the dataset of file wisc_bc_data.csv and following settings to design the classifier:
# -	Min-max feature normalization.
# -	Randomly select 100 healthy and 100 cancerous cell samples to construct the training dataset. Use rest of the samples to estimate the accuracy of the classifier.
# -	Calculate the accuracies for K = 9, 11, 13, 15, 17 and 19

# In[1]:


import pandas as pd
import numpy as np
#Read the dataset using pandas
df = pd.read_csv('wisc_bc_data.csv')


# In[2]:


#Printing first 5 rows of the train dataset.
df.head(5)


# ### Data Dictionary
# - radius_mean
# - texture_mean
# - perimeter_mean
# - area_mean
# - smoothness_mean
# - compactness_mean
# - concavity_mean
# - concave points_mean
# - symmetry_mean
# - fractal_dimension_mean
# - radius_se
# - texture_se
# - perimeter_se
# - area_se
# - smoothness_se
# - compactness_se
# - concavity_se
# - concave points_se
# - symmetry_se
# - fractal_dimension_se
# - radius_worst
# - texture_worst
# - perimeter_worst
# - area_worst
# - smoothness_worst
# - compactness_worst
# - concavity_worst
# - concave points_worst
# - symmetry_worst
# - fractal_dimension_worst

# In[3]:


df.shape


# ###  Totaal rows = 569, Total columns = 33

# In[4]:


#check if all the columns have numerical values
df.info()


# In[5]:


#check if there are any null cells in dataset
df.isnull().sum()

#dropping useless column ID as it is not helful in training of our model.
df = df.drop('id', axis = True)


# In[6]:


#preprocessing
#map diagnosis to numerical values 1,0 for M & B respectively
def diagnosis_mapping(diagnosis): 
    if diagnosis == 'M': 
        return 1
    else: 
        return 0
    
df['diagnosis'] = df['diagnosis'].apply(diagnosis_mapping) 


# In[7]:


#dividing dataset into 2 on the basis of label
df_M = df[df['diagnosis'] == 1]
df_B = df[df['diagnosis'] == 0]


# In[8]:


#randomly selecting 100 cancerous cell samples for training
df_M_train = df_M.sample(n=100)
df_M_train


# In[9]:


#randomly selecting 100 healthy cell samples for training
df_B_train = df_B.sample(n=100)
df_B_train


# In[10]:


#joining the two dataframes to form whole training data
df_train = pd.concat([df_B_train, df_M_train], ignore_index=True)
df_train


# In[11]:


#removing training data from whole dataset to obtain the test data
df_test = pd.concat([df, df_train])
df_test.drop_duplicates(keep=False)


# In[12]:


#extracting only featues from whole train sample
X_train = np.array(df_train.iloc[:, 1:]) 

#extracting only lables from whole train sample
y_train = np.array(df_train['diagnosis'])


# In[13]:


#extracting only featues from whole test sample 
X_test = np.array(df_test.iloc[:, 1:]) 

#extracting only lables from whole test sample
y_test = np.array(df_test['diagnosis']) 


# In[14]:


from sklearn import preprocessing
#Min-max feature normalization [0,1] using sklearn
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)


# In[15]:


scores = {}
scores_list = []
#importing KNN model from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
list_K = [9,11,13,15,17,19] #making a list of ks given in the question
for k in list_K:  
    knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance') #weighted KNN using parameter 'weights'='distance'
    knn.fit(X_train,y_train) #training the model
    y_pred = knn.predict(X_test) #testing the model
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))
scores #stored dictionary of k with their accuracies as key


# In[16]:


#ploting k and acuracies
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (10, 6)) 
plt.plot(list_K,scores_list) 
plt.xlabel('Value of K for Weighted KNN') 
plt.ylabel('Testing Accuracy') 
plt.show() 

