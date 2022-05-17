#!/usr/bin/env python
# coding: utf-8

# # Google COLAB and Python

# A. Why Pyhton?
# 
# Some Common Uses for Python in Research
# 
# B. Understanding the python environment:
# 
# Navigating the Python
# 
# Understanding Toolbox Components
# 
# Executing Commands
# 
# Help and Documentation
# 
# C. Using Python
# 
# Lists, arrays, dataframes - and their apporpriate packages
# 

# Why and What Python?
# 

# Python is an open source programming language that was made to be easy-to-read and powerful. A Dutch programmer named Guido van Rossum made Python in 1991. He named it after the television show Monty Python's Flying Circus. Many Python examples and tutorials include jokes from the show
# 
# 

# The Python programming language in particular has seen a surge in popularity across the sciences, for reasons which include its readability, modularity, and large standard library. The use of Python as a scientific programming language began to increase with the development of numerical libraries for optimized operations on large arrays in the late 1990s, in which an important development was the merging of the competing Numeric and Numarray packages in 2006 to form NumPy (Oliphant, 2007). As Python and NumPy have gained traction in a given scientific domain, we have seen the emergence of domain-specific ecosystems of open-source Python software developed by scientists. It became clear to us in 2007 that we were on the cusp of an emerging Python in neuroscience ecosystem, particularly in computational neuroscience and neuroimaging, but also in electrophysiological data analysis and in psychophysics.

# Currently has grown into an interactive system and high level programming language for general
# scientific and technical computation
# 
# Common Uses for Python in Research
# • Data Acquisition
# • Multi-platform, Multi Format data importing
# • Analysis Tools (Existing,Custom)/Statistics
# • Graphing
# 

# Data Acuisition:
# 
# A framework for bringing live, measured data into computers via aquisition hardware
# 
# ![data-acquisition-system.png](https://github.com/BlohmLab/NSCI801-QuantNeuro/blob/master/Figures/data-acquisition-system.png?raw=1)

# Multi-platform, Multi Format data importing
# 
# • Data can be loaded into Python from almost any format and platform (numpy, pandas etc.)
# • Binary data files (eg. REX, PLEXON etc.)
# • Ascii Text (eg. Eyelink I, II)
# • Analog/Digital Data files
# 
# 
# ![importing_data.png](https://github.com/BlohmLab/NSCI801-QuantNeuro/blob/master/Figures/importing_data2.png?raw=1)

# Analysis Tools (Existing,Custom)/Statistics
# 
# • A considerable library of analysis tools exist for data analysis (sklearn etc.)
# • Provides a framework for the design, creation, and implementation of any custom analysis tool
#   imaginable
# 
# A considerable variety of statistical tests available including: Parametric and non-parametric, and various others (Scipy, sklearn etc)
# 
# 

# Graphing
# 
# • A Comprehensive array of plotting options available from 2 to 4 dimensions (matplotlib)
# • Full control of formatting, axes, and other visual representational elements
# 
# ![Mpl_screenshot_figures_and_code.png](https://github.com/BlohmLab/NSCI801-QuantNeuro/blob/master/Figures/Mpl_screenshot_figures_and_code.png?raw=1)

# Machine Learning is the hottest trend in modern neuroscience (and science in general). Machine learning patents grew at a 34% rate between 2013 and 2017 and this is only set to increase in the future. And Python is the primary programming language used for much of the research and development in Machine Learning. So much so that Python is the top programming language for Machine Learning according to Github. However, while it is clear that Python is the most popular, this article focuses on the all-important question of “Why is Python the Best-Suited Programming Language for Machine Learning?”

# Since most of the students are new to Python, they could run into the following practical problems easily.
# 
# 1) Do not know how to install and set up the Python running environment;
# 
# 2) Do not know how to find the solutions effectively when facing the problems;
# 
# 3) Do not know how to collaborate with others when trying to finish the group tasks；
# 
# 4) Do not know how to handle version control, which may lead the code to chaotic.
# 
# The problems mentioned above are the main pain points for Python beginners.

# Your own machine and Anaconda 
# 
# ![alt text](https://github.com/BlohmLab/NSCI801-QuantNeuro/blob/master/Figures/Screenshot_anaconda.png?raw=1)

# Getting Started
# 
# To start working with Colab you first need to log in to your google account, then go to this link https://colab.research.google.com.

# Opening Jupyter Notebook in Colab:
# On opening the website you will see a pop-up containing following tabs –

# ![Screenshot%202019-12-24%2013.54.17.png](https://github.com/BlohmLab/NSCI801-QuantNeuro/blob/master/Figures/Screenshot%202019-12-24%2013.54.17.png?raw=1)

# EXAMPLES: Contain a number of Jupyter notebooks of various examples.
# RECENT: Jupyter notebook you have recently worked with.
# GOOGLE DRIVE: Jupyter notebook in your google drive.
# GITHUB: You can add Jupyter notebook from your GitHub but you first need to connect Colab with GitHub.
# UPLOAD: Upload from your local directory.

# On creating a new notebook, it will create a Jupyter notebook with Untitled.ipynb and save it to your google drive in a folder named Colab Notebooks. Now as it is essentially a Jupyter notebook, all commands of Jupyter notebooks will work here. 
# 
# ![notebook_figure.png](https://github.com/BlohmLab/NSCI801-QuantNeuro/blob/master/Figures/notebook_figure.png?raw=1)

# You can change runtime environment by selecting the “Runtime” dropdown menu. Select “Change runtime type”. 
# 
# ![runtime_enviornment.png](https://github.com/BlohmLab/NSCI801-QuantNeuro/blob/master/Figures/runtime_enviornment.png?raw=1)

# Click the “Runtime” dropdown menu. Select “Change runtime type”. Now select anything(GPU, CPU, None) you want in the “Hardware accelerator” dropdown menu.
# 
# ![notebook_settings.png](https://github.com/BlohmLab/NSCI801-QuantNeuro/blob/master/Figures/notebook_settings.png?raw=1)
# 

# Install Python packages –
# Use can use pip to install any package. For example:
# 
# 
# "pip install pandas"
# 
# Commonly Used Packages
# 
# 1) os
# 2) numpy
# 3) scipy
# 4) pandas
# 

# ## Variables and Expressions

# Solving equations using variables
# 
# Expression language
# 
# Expressions typed by the user are interpreted and evaluated by the python
# 
# Variables are names used to store values 
# 
# Variable names allow stored values to be retrieved for calculations or permanently saved  
# 
# Variable = Expression
# 
# **Variable Names are Case Sensitive!
# 

# HELP DOCUMENTATION!!!!
# 
# numpy: https://numpy.org
# scipy: https://www.scipy.org/scipylib/index.html
# pandas: https://pandas.pydata.org
# scikit-learn: https://scikit-learn.org/stable/
# 
# THIS NEEDS TO BECOME YOUR BEST FRIEND!

# ## Lists

# Lists:
# 
# In Python programming, a list is created by placing all the items (elements) inside a square bracket [ ], separated by commas.
# 
# It can have any number of items and they may be of different types (integer, float, string etc.).

# In[1]:


# empty list
my_list = []
# list of integers
my_list = [1, 2, 3]
# list with mixed datatypes
my_list = [1, "Hello", 3.4]


# Also, a list can even have another list as an item. This is called nested list.

# In[2]:


# nested list
my_list = ["mouse", [8, 4, 6], ['a']]


# How to access elements from a list?
# 
# There are various ways in which we can access the elements of a list.
# 
# List Index
# 
# We can use the index operator [] to access an item in a list. Index starts from 0. So, a list having 5 elements will have index from 0 to 4.
# 
# Trying to access an element other that this will raise an IndexError. The index must be an integer. We can't use float or other types, this will result into TypeError.
# 
# Nested list are accessed using nested indexing.

# ![python-list-index.png](attachment:python-list-index.png)

# In[3]:


my_list = ['p','r','o','b','e']
# Output: p
print(my_list[0])
# Output: o
print(my_list[2])
# Output: e
print(my_list[4])
# Error! Only integer can be used for indexing
# my_list[4.0]
# Nested List
n_list = ["Happy", [2,0,1,5]]
# Nested indexing
# Output: a
print(n_list[0][1])    
# Output: 5
print(n_list[1][3])


# Negative indexing
# Python allows negative indexing for its sequences. The index of -1 refers to the last item, -2 to the second last item and so on.

# In[4]:


my_list = ['p','r','o','b','e']
# Output: e
print(my_list[-1])
# Output: p
print(my_list[-5])


# List are mutable, meaning, their elements can be changed unlike string or tuple.
# 
# We can use assignment operator (=) to change an item or a range of items.
# 
# Finally, we can even add to lists using append, and extend

# In[5]:


# mistake values
odd = [2, 4, 6, 8]
# change the 1st item    
odd[0] = 1            
# Output: [1, 4, 6, 8]
print(odd)
# change 2nd to 4th items
odd[1:4] = [3, 5, 7]  
# Output: [1, 3, 5, 7]
print(odd)   

odd = [1, 3, 5]
odd.append(7)
# Output: [1, 3, 5, 7]
print(odd)
odd.extend([9, 11, 13])
# Output: [1, 3, 5, 7, 9, 11, 13]
print(odd)


# We can also use + operator to combine two lists. This is also called concatenation.
# 
# The * operator repeats a list for the given number of times.

# In[6]:


odd = [1, 3, 5]
# Output: [1, 3, 5, 9, 7, 5]
print(odd + [9, 7, 5])
#Output: ["re", "re", "re"]
print(["re"] * 3)


# Furthermore, we can insert one item at a desired location by using the method insert() or insert multiple items by squeezing it into an empty slice of a list.

# In[7]:


odd = [1, 9]
odd.insert(1,3)
# Output: [1, 3, 9] 
print(odd)
odd[2:2] = [5, 7]
# Output: [1, 3, 5, 7, 9]
print(odd)


# Python List Methods:
# 
# append() - Add an element to the end of the list
# extend() - Add all elements of a list to the another list
# insert() - Insert an item at the defined index
# remove() - Removes an item from the list
# pop() - Removes and returns an element at the given index
# clear() - Removes all items from the list
# index() - Returns the index of the first matched item
# count() - Returns the count of number of items passed as an argument
# sort() - Sort items in a list in ascending order
# reverse() - Reverse the order of items in the list
# copy() - Returns a shallow copy of the list

# ## Array

# If you create arrays using the array module, all elements of the array must be of the same numeric type.
# 
# Create a NumPy Array
# 
# Simplest way to create an array in Numpy is to use Python List 
# 
# To convert python list to a numpy array by using the object np.array.
# 
# but before we do we need to import the package
# 
# import numpy as np

# In[8]:


import numpy as np

myPythonList = [1,9,8,3]

numpy_array_from_list = np.array(myPythonList)

print(numpy_array_from_list)


# Mathematical Operations on an Array
# You could perform mathematical operations like additions, subtraction, division and multiplication on an array. The syntax is the array name followed by the operation (+.-,*,/) followed by the operand

# In[ ]:


numpy_array_from_list + 10

#This operation adds 10 to each element of the numpy array.


# Shape of Array
# You can check the shape of the array with the object shape preceded by the name of the array. In the same way, you can check the type with dtypes.

# In[ ]:


a  = np.array([1,2,3])
print(a.shape)
print(a.dtype)


# 2 Dimension Array and 3 Dimension Array
# 
# You can add a dimension with a ","coma
# 
# Note that it has to be within the bracket []
# 

# In[ ]:


### 3 dimension
d = np.array([[[1, 2,3],[4, 5, 6]],[[7, 8,9],[10, 11, 12]]])
print(d.shape)


# What is np.zeros and np.ones?
# 
# You can create a matrix full of zeroes or ones using np.zeros and np.one commands respectively.

# In[ ]:


#ones
A = np.ones((2,3))
#zeros
B = np.zeros((3,4))


# Reshape Data
# 
# In some occasions, you need to reshape the data from wide to long. You can use the reshape function for this. The syntax is

# In[ ]:


e  = np.array([(1,2,3), (4,5,6)])
print(e)
e.reshape(3,2)


# EXTRACTING INFORMATION FROM ARRAYS IS BASICALLY IDENTICAL TO LISTS!!!!

# The key difference between an array and a list is, arrays are designed to handle vectorized operations while a python list is not.
# 
# That means, if you apply a function it is performed on every item in the array, rather than on the whole array object.
# 
# Let’s suppose you want to add the number 2 to every item in the list. The intuitive way to do it is something like this:

# In[ ]:


#e is an array defined above
D = e+15
print("the array is ", D)
#odd is a list from above
#odd +15


# You can always convert an array back to a python list using 

# In[ ]:


odd.tolist()
print(odd)


# Array Operations like Mean, min, max, etc....

# In[ ]:


print("Mean value is: ", D.mean()) # alternatively could be np.mean(D)
print("Max value is: ", D.max()) # alternatively could be np.max(D)
print("Min value is: ", D.min()) # alternatively could be np.min(D)


# Reshaping and Flattening Multidimensional arrays
# Reshaping is changing the arrangement of items so that shape of the array changes while maintaining the same number of dimensions.

# In[ ]:


D.reshape(1,-1)
D.flatten()


# ## Dataframe

# What Are Pandas Data Frames?
# 
# Before you start, let’s have a brief recap of what DataFrames are.
# Those who are familiar with R know the data frame as a way to store data in rectangular grids that can easily be overviewed. Each row of these grids corresponds to measurements or values of an instance, while each column is a vector containing data for a specific variable. This means that a data frame’s rows do not need to contain, but can contain, the same type of values: they can be numeric, character, logical, etc.
# 

# Creating DataFrames
# 
# Obviously, making your DataFrames is your first step in almost anything that you want to do when it comes to data munging in Python. Sometimes, you will want to start from scratch, but you can also convert other data structures, such as lists or NumPy arrays, to Pandas DataFrames. In this section, you’ll only cover the latter. 

# In[ ]:


import pandas as pd

data = np.array([[1,2],[3,4]])
col = ['Col1','Col2']

df = pd.DataFrame(data=data,columns=col)

print(df)


# Fundamental DataFrame Operations
# 
# Now that you have put your data in a more convenient Pandas DataFrame structure, it’s time to get to the real work!
# This first section will guide you through the first steps of working with DataFrames in Python. It will cover the basic operations that you can do on your newly created DataFrame: adding, selecting, deleting, renaming, … You name it!
# 
# Selecting an Index or Column 
# 

# In[ ]:


# Using `iloc[]`
print('using iloc ', df.iloc[0][0]) #can't use boolean

# Using `loc[]`
print('using loc ', df.loc[0]['Col2']) #CAN use boolean

# Using `at[]`
print('using at ',df.at[0,'Col2'])

# Using `iat[]`
print('using iat ', df.iat[0,0])


# How do we select just the data, columns or Index

# In[ ]:


#get values
vals = df.values
print("these are the values ", vals)

#get columns
cls = df.columns.values
print("these are the values ", cls)

#get columns
idx = df.index.values
print("these are the values ", idx)




# From here all the same operations done on lists and arrays are possible!!!

# In[ ]:




