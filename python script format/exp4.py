#!/usr/bin/env python
# coding: utf-8

# ## Classify the mushrooms into poisonous and non-poisonous class using the Decision Tree classifier. Use the dataset named “mushrooms.csv” and use the following classifier settings to evaluate the performance of classifier:
# -	Use 80% of the data sample to train the classifier and rest of them to evaluate the classifier performance. 
# -	Calculate the accuracy, sensitivity, specificity, true positive rate and false positive rate.

# ## 1. Importing dataset

# In[1]:


import pandas as pd
complete_df = pd.read_csv('mushroom_csv.csv')


# ## 2. Exploratory data analysis
# Printing first 5 rows of the train dataset.

# In[2]:


complete_df.head(5)


# ### Data Dictionary
# - cap-shape
# - cap-surface
# - cap-color
# - bruises%3F
# - odor
# - gill-attachment
# - gill-spacing
# - gill-size
# - gill-color
# - stalk-shape
# - stalk-root
# - stalk-surface-above-ring
# - stalk-surface-below-ring
# - stalk-color-above-ring
# - stalk-color-below-ring
# - veil-type
# - veil-color
# - ring-number
# - ring-type
# - spore-print-color
# - population
# - habitat
# - class

# In[3]:


complete_df.shape


# In[4]:


complete_df.info()


# In[5]:


complete_df.isnull().sum()


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots


# In[7]:


def bar_chart(feature):
    poisonus = complete_df[complete_df['class']=='p'][feature].value_counts()
    not_poisonus = complete_df[complete_df['class']=='e'][feature].value_counts()
    df = pd.DataFrame([poisonus,not_poisonus])
    df.index = ['poisonus','not_poisonus']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# ## 3. Feature engineering

# In[8]:


complete = [complete_df]


# In[9]:


#cap-shape
complete_df['cap-shape'].value_counts()


# In[10]:


cap_shape_mapping = {"x": 0, "f": 1, "k": 2, 
                 "b": 3, "s": 4, "c": 5 }
for dataset in complete:
    dataset['cap-shape'] = dataset['cap-shape'].map(cap_shape_mapping)


# In[11]:


bar_chart('cap-shape')


# In[12]:


complete_df['cap-surface'].value_counts()


# In[13]:


cap_surface_mapping = {"y": 0, "s": 1, "f": 2, 
                 "g": 3 }
for dataset in complete:
    dataset['cap-surface'] = dataset['cap-surface'].map(cap_surface_mapping)


# In[14]:


bar_chart('cap-surface')


# In[15]:


complete_df['cap-color'].value_counts()


# In[16]:


cap_color_mapping = {"n": 0, "g": 1, "e": 2, 
                 "y": 3,"w": 4, "b": 5, "p": 6, 
                 "c": 7,"u": 8, "r": 9 }
for dataset in complete:
    dataset['cap-color'] = dataset['cap-color'].map(cap_color_mapping)


# In[17]:


bar_chart('cap-color')


# In[18]:


#bruises%3F
complete_df['bruises%3F'].value_counts()


# In[19]:


bruises_mapping = {"f": 0, "t": 1}
for dataset in complete:
    dataset['bruises%3F'] = dataset['bruises%3F'].map(bruises_mapping)


# In[20]:


bar_chart('bruises%3F')


# In[21]:


#odor
complete_df['odor'].value_counts()


# In[22]:


odor_mapping = {"n": 0, "f": 1, "y": 2, 
                 "s": 3,"a": 4, "l": 5, "p": 6, 
                 "c": 7,"m": 8 }
for dataset in complete:
    dataset['odor'] = dataset['odor'].map(odor_mapping)


# In[23]:


bar_chart('odor')


# In[24]:


#gill-attachment
complete_df['gill-attachment'].value_counts()


# In[25]:


gill_attachment_mapping = {"f": 0, "a": 1}
for dataset in complete:
    dataset['gill-attachment'] = dataset['gill-attachment'].map(gill_attachment_mapping)
bar_chart('gill-attachment')


# In[26]:


#gill-spacing
complete_df['gill-spacing'].value_counts()


# In[27]:


gill_spacing_mapping = {"c": 0, "w": 1}
for dataset in complete:
    dataset['gill-spacing'] = dataset['gill-spacing'].map(gill_spacing_mapping)
bar_chart('gill-spacing')


# In[28]:


#gill-size
complete_df['gill-size'].value_counts()


# In[29]:


gill_size_mapping = {"b": 0, "n": 1}
for dataset in complete:
    dataset['gill-size'] = dataset['gill-size'].map(gill_size_mapping)
bar_chart('gill-size')


# In[30]:


#gill-color
complete_df['gill-color'].value_counts()


# In[31]:


gill_color_mapping = {"b": 0, "p": 1, "w": 2, 
                 "n": 3,"g": 4, "h": 5, "u": 6, 
                 "k": 7,"e": 8,"y":9 , "o":10, "r":11}
for dataset in complete:
    dataset['gill-color'] = dataset['gill-color'].map(gill_color_mapping)
bar_chart('gill-color')


# In[32]:


#stalk-shape
complete_df['stalk-shape'].value_counts()


# In[33]:


stalk_shape_mapping = {"t": 0, "e": 1}
for dataset in complete:
    dataset['stalk-shape'] = dataset['stalk-shape'].map(stalk_shape_mapping)
bar_chart('stalk-shape')


# In[34]:


#stalk-root
complete_df['stalk-root'].value_counts()


# In[35]:


stalk_root_mapping = {"b": 0, "e": 1, "c": 2, "r": 3}
for dataset in complete:
    dataset['stalk-root'] = dataset['stalk-root'].map(stalk_root_mapping)
bar_chart('stalk-root')


# In[36]:


complete_df["stalk-root"].fillna(complete_df.groupby("stalk-shape")["stalk-root"].transform("median"), inplace=True)
complete_df.groupby("stalk-shape")["stalk-root"].transform("median")


# In[37]:


complete_df.isnull().sum()


# In[38]:


#stalk-surface-above-ring
complete_df['stalk-surface-above-ring'].value_counts()


# In[39]:


above_ring_mapping = {"s": 0, "k": 1, "f": 2, "y": 3}
for dataset in complete:
    dataset['stalk-surface-above-ring'] = dataset['stalk-surface-above-ring'].map(above_ring_mapping)
bar_chart('stalk-surface-above-ring')


# In[40]:


#stalk-surface-below-ring
complete_df['stalk-surface-below-ring'].value_counts()


# In[41]:


below_ring_mapping = {"s": 0, "k": 1, "f": 2, "y": 3}
for dataset in complete:
    dataset['stalk-surface-below-ring'] = dataset['stalk-surface-below-ring'].map(below_ring_mapping)
bar_chart('stalk-surface-below-ring')


# In[42]:


#stalk-color-above-ring
complete_df['stalk-color-above-ring'].value_counts()


# In[43]:


c_above_ring_mapping = {"w": 0, "p": 1, "g": 2, "n": 3, "b":4, "o":5 ,"e":6, "c":7, "y":8}
for dataset in complete:
    dataset['stalk-color-above-ring'] = dataset['stalk-color-above-ring'].map(c_above_ring_mapping)
bar_chart('stalk-color-above-ring')


# In[44]:


#stalk-color-below-ring
complete_df['stalk-color-below-ring'].value_counts()


# In[45]:


c_below_ring_mapping = {"w": 0, "p": 1, "g": 2, "n": 3, "b":4, "o":5 ,"e":6, "c":7, "y":8}
for dataset in complete:
    dataset['stalk-color-below-ring'] = dataset['stalk-color-below-ring'].map(c_below_ring_mapping)
bar_chart('stalk-color-below-ring')


# In[46]:


#veil-type
complete_df['veil-type'].value_counts()


# As both types of mushrooms have save veil_type, this data field is useless and should be removed

# In[47]:


complete_df.drop('veil-type', axis=1, inplace=True)


# In[48]:


#veil-color
complete_df['veil-color'].value_counts()


# In[49]:


veil_color_mapping = {"w": 0, "n": 1, "o": 2, "y": 3}
for dataset in complete:
    dataset['veil-color'] = dataset['veil-color'].map(veil_color_mapping)
bar_chart('veil-color')


# In[50]:


#ring-number
complete_df['ring-number'].value_counts()


# In[51]:


ring_number_mapping = {"o": 0, "t": 1, "n": 2}
for dataset in complete:
    dataset['ring-number'] = dataset['ring-number'].map(ring_number_mapping)
bar_chart('ring-number')


# In[52]:


#ring-type
complete_df['ring-type'].value_counts()


# In[53]:


ring_type_mapping = {"p": 0, "e": 1, "l": 2, "f": 3, "n":4}
for dataset in complete:
    dataset['ring-type'] = dataset['ring-type'].map(ring_type_mapping)
bar_chart('ring-type')


# In[54]:


#spore-print-color
complete_df['spore-print-color'].value_counts()


# In[55]:


spore_print_color_mapping = {"w": 0, "n": 1, "k": 2, "h": 3, "r":4, "y":5 ,"u":6, "o":7, "b":8}
for dataset in complete:
    dataset['spore-print-color'] = dataset['spore-print-color'].map(spore_print_color_mapping)
bar_chart('spore-print-color')


# In[56]:


#population
complete_df['population'].value_counts()


# In[57]:


population_mapping = {"v": 0, "y": 1, "s": 2, "n": 3, "a":4, "c":5}
for dataset in complete:
    dataset['population'] = dataset['population'].map(population_mapping)
bar_chart('population')


# In[58]:


#habitat
complete_df['habitat'].value_counts()


# In[59]:


habitat_mapping = {"d": 0, "g": 1, "p": 2, "l": 3, "u":4, "m":5, "w":6}
for dataset in complete:
    dataset['habitat'] = dataset['habitat'].map(habitat_mapping)
bar_chart('habitat')


# In[60]:


complete_df.head()


# In[61]:


complete_df.info()


# In[62]:


train_data = complete_df.drop('class', axis=1)
target = complete_df['class']

train_data.shape, target.shape


# In[63]:


train_data.head(10)


# ### 4. Modelling

# In[64]:


import numpy as np


# In[65]:


train_data.info()


# ## 5. Decision Tree Classifier

# In[66]:


from sklearn.model_selection import train_test_split
X = train_data # Features
y = target  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## 6. Training

# In[67]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[68]:


#visualizing decision tree
from sklearn import tree
fn= list(train_data.columns)
cn=['p', 'e']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dt,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('decisionTree.png')


# # 7. Testing

# ### Visualizing Important Features

# In[69]:


feature_imp = pd.Series(dt.feature_importances_,index = list(train_data.columns)).sort_values(ascending=False)
feature_imp


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[71]:


y_pred = dt.predict(X_test)


# # 8. Confusion Matrix

# In[72]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(y_test, y_pred) 
  
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(y_test, y_pred))
print('Report : ')
print(classification_report(y_test, y_pred))

