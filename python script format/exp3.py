#!/usr/bin/env python
# coding: utf-8

# ## Write a program to generate the 100 sample point which follows the following linear equation: y=5x+3. Assume that the  x variable take the values in range [0, 10].
# - NOTE: Use the values of (y ) ̂ and x to estimate the value of slope and intercept.
# 

# In[1]:


import numpy as np
#creating a function which takes low bound, high bund and n as input and outputs n random integers between the bounds
def gen_random_points(low_bound, high_bound, n):
    X = np.random.uniform(low_bound, high_bound, n)
    y = 5*X + 3
    return (X, y)
#generating 100 random points between [0,10]
X, y = gen_random_points(0,10,100)


# In[8]:


#visualizing the obtained points on scatterplot
import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.ylabel('y')
plt.xlabel('X')
plt.show()


# ##  Now add the additive white Gaussian noise N(0,1) with zero mean and unity variance to y to obtain (y ) ̂=y+N(0,1). Now estimate the values of slope and intercept using the linear regression analysis. 

# In[3]:


#Adding AWGN with mean=0 and varience = 1 to the output y
noise = np.random.normal(0, 1, y.shape)
y_ = noise + y


# In[4]:


#plotting y'
import matplotlib.pyplot as plt
plt.scatter(X,y_)
plt.ylabel('y_')
plt.xlabel('X')
plt.show()


# In[5]:


#applying simple lenier regression to estimate slope and intercept
def estimate_coef(x, y):
    n = np.size(x) 
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x
    # calculating regression coefficients 
    slope = SS_xy / SS_xx 
    intercept = m_y - slope*m_x 
  
    return intercept,slope
intercept, slope = estimate_coef(X,y_)


# In[6]:


#printing the result
print("The estimated equation found using LR is:", "Y =",slope,"*X + ",intercept)


# In[7]:


#plotting the estimated line and train data
plt.scatter(X, y_, color = "r",marker = "o", s = 10) 
y_pred = intercept + slope*X 
plt.plot(X, y_pred, color = "g") 
plt.xlabel('X') 
plt.ylabel('y') 
plt.show() 

