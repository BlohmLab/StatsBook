#!/usr/bin/env python
# coding: utf-8

# # Advanced Python
# ## Quick Review

# In[1]:


# pip install pandas
import os
import pandas as pd
import numpy as np
#from numpy import zeros


# LISTS
A = ['yep', 0 ,True,3]
A_1 = [0,1,2,3,4]

print('Lists comprise any data type A = ')
print(A)

#indexing A ########## NOT LIKE MATLAB indexing starts at 0
print('the first element in A')
print(A[0])

#checking the length of a list: ### function len
print('The length of A')
print(len(A))

# get the mean of a list (IF IT HAS NUMBERS)
print('Mean of A_1')
print(np.mean(A_1))
#print(np.mean(A))




# In[ ]:


#numpy arrays
B = np.array([[2,3,1,0],[ 0,1,2,3]])
print('Numpy array B = ')
print(B)

#indexing B ########## NOT LIKE MATLAB 
print('The first position in the array:')
print(B[0])

#checking the shape of a numpy array/matrix: ### function shape
print('The shape of B')
print(B.shape)

# get the mean of a array
print('The mean of B along axis 0')
print(np.mean(B,axis=0))
print('The mean of B along axis 1')
print(np.mean(B,axis=1))




# In[3]:


import pandas as pd

path_2_data = '/Users/joe/Desktop/rois.csv'


data = pd.read_csv(path_2_data)
data = pd.read_csv('/Users/joe/Desktop/rois.csv')

## see inside dataFrame
print(data)

## Get values
print('Get data values')
print(data.values)

## Get headers
print('Get header values')
print(data.columns.values)




# In[ ]:


#construct data values
data_values = data.values

#get data for plotting
x_values = data_values[:,0]
y_values = data_values[:,1]



# In[ ]:


#import relevant packages
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x_values,y_values,'g')
ax.set_xlabel('Volumes')
ax.set_ylabel('au')
plt.show()


# In[ ]:


import scipy.stats as ss

fig, ax = plt.subplots()

zscored_data = ss.zscore(data_values[:,1::],axis=0)
correlation_co = np.corrcoef(zscored_data.transpose())

ax.imshow(correlation_co)
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()


# ## Loops

# Loops the simplest examples:
# 
# For loops are for iterating through “elements”. Most of the time these “elements” will be in well-known data types: lists, strings or dictionaries. Sometimes they can also be range() objects.
# 
# Let’s take the simplest example first: a list!

# In[6]:


thislist = ["apple", "banana", "cherry", "peach", "orange"]


# Once it’s created, we can go through the elements and print them one by one – by using this very basic for loop:

# In[7]:


for i in thislist:
    print(i)


# Wonderful!
# But how does this actually become useful? Take another example, a list of numbers:
# 

# In[8]:


numbers = [10, 15, 21, 19, 102]


# In[9]:


for i in numbers:
    print(i)


# Say that we want to square each of the numbers in this list!
# 
# Note: Unfortunately the numbers * numbers formula won’t work… I know, it might sound logical at first but when you will get more into Python, you will see that in fact, it is not logical at all.
# 
# We have to do this:

# In[14]:


for i in numbers:
    print(i * i)

print(i)


# What happened here step by step is:
# 
# 1 - We set a list (numbers) with five elements.
# 2 - We took the first element (well actually, because of zero-based indexing, it’s the 0th element) of the list (1) and stored it into the i variable.
# 3 - Then we executed the print(i * i) function, which returned the squared value of 1 which is also 1.
# 4 - Then we started the whole process over…
# 5 - We took the next element – we assigned it to the i variable.
# 6 - We executed print(i * i) again and we got the squared value of the second element: 25.
# 7 - And we continued this process until we got the last element’s squared value.
# 

# What if we want to create a new list with the squares?????

# In[ ]:


New_numbers = []

for i in numbers:
    New_numbers.append(i*i) # APPEND for LISTS

print(New_numbers)


# ## Iterating through strings

# Okay, going forward!
# As I mentioned earlier, you can use other sequences than lists too. Let’s try a string:

# In[ ]:


my_list = "Hello World!"
my_list2 = ["H","e","l"]
for i in my_list:
    print(i)


# Remember, strings are basically handled as sequences of characters, thus the for loop will work with them pretty much as it did with lists.
# 
# 

# ## Iterating through range() objects

# In[15]:


my_list = range(0,10)
for i in range(0,10):
    print(i)


# ![Python-For-Loops-range-explanation-1024x435.png](attachment:Python-For-Loops-range-explanation-1024x435.png)

# ## Iterating through interables using enumerate

# Lets use our old list of numbers

# In[16]:


numbers = [10, 15, 21, 19, 102]

for num,i in enumerate(numbers):
    print("the location of the element " +str(i)+" is " +str(num))


# In[ ]:


#Alternatively 

numbers = [10, 15, 21, 19, 102]
numbers_length = len(numbers)
for i in range(0,numbers_length):
    print(i, numbers[i])


# ## Conditional statements

# ![1-Python-if-statement-logic-1024x499.png](attachment:1-Python-if-statement-logic-1024x499.png)

# We have two values: a = 10 and b = 20. We compare these two values: a == b. This comparison has either a True or a False output. 

# In[17]:


a = 10
b = 20
a == b


# In[ ]:


a = 10
b = 20
not a == b


# We can go even further and set a condition: if a == b is True then we print 'yes'. If it’s False then we print 'no'. And that’s it, this is the logic of the Python if statements. 

# In[ ]:


a = 10
b = 20
if a == b:
    print('yes')
else:
    print('no')


# Now that you understand the basics, it’s time to make your conditions more complex 

# In[18]:


a = 10
b = 20
c = 30
if (a + b) / c == 1 or c - b - a == 0:
    print('yes')
else:
    print('no')


# Lets add one more level of complexity

# ![10-Python-if-statement-condition-sequence-logic-1024x473.png](attachment:10-Python-if-statement-condition-sequence-logic-1024x473.png)

# In[ ]:


a = 10
b = 11
c = 10
if a == b:
    print('first condition is true')
elif a == c:
    print('second condition is true')
else:
    print('nothing is true. existence is pain.')


# Now that you understand the basics, it’s time to make your conditions more complex 
# 
# 

# ## Functions
# 

# Functions are an essential part of the Python programming language: you might have already encountered and used some of the many fantastic functions that are built-in in the Python language or that come with its library ecosystem. However, as a Data Scientist, you’ll constantly need to write your own functions to solve problems that your data poses to you.

# How To Define A Function: User-Defined Functions (UDFs)
# 
# The four steps to defining a function in Python are the following:
# 1 - Use the keyword def to declare the function and follow this up with the function name.
# 
# 2 - Add parameters to the function: they should be within the parentheses of the function. End your line with a colon.
# 
# 3 - Add statements that the functions should execute.
# 
# 4 - End your function with a return statement if the function should output something. Without the return statement, your function will return an object None.

# In[20]:


def hello():
  print("Hello World") 
  return 

hello()


# Of course, your functions will get more complex as you go along: They often include loops, flow control, … and more to it to make it more complex.

# In[23]:


def hello():
  name = input("Enter your name: ")
  #name = str(input("Enter your name: "))
  if name:
    print ("Hello " + str(name))
  else:
    print("Hello World") 
  return 

hello()


# The return Statement
# 
# Note that as you’re printing something in your UDF hello(), you don’t really need to return it. There won’t be any difference between the function above and this one:

# However, if you want to continue to work with the result of your function and try out some operations on it, you will need to use the return statement to actually return a value, such as a String, an integer, …. Consider the following scenario, where hello() returns a String "hello", while the function hello_noreturn() returns None:

# In[29]:


def hello():
  print("Hello World") 
  return("hello")
my_list = []
my_list = hello()

print(my_list)

def hello_noreturn():
  print("Hello World")
  


# In[28]:


# Multiply the output of `hello()` with 2 
hello() * 2


# In[30]:


# (Try to) multiply the output of `hello_noreturn()` with 2 
hello_noreturn() * 2


# The second function gives you an error because you can’t perform any operations with a None. You’ll get a TypeError that says that you can’t do the multiplication operation for NoneType (the None that is the result of hello_noreturn()) and int (2).
# 

# Tip functions immediately exit when they come across a return statement, even if it means that they won’t return any value.

# In[32]:


def run():
  for x in range(10):
     if x == 2:
       return
     print("Run!")
  
run()


# Another thing that is worth mentioning when you’re working with the return statement is the fact that you can use it to return multiple values. To do this, you make use of tuples.
# Remember that this data structure is very similar to that of a list: it can contain multiple values. However, tuples are immutable, which means that you can’t modify any amounts that are stored in it! You construct it with the help of double parentheses (). You can unpack tuples into multiple variables with the help of the comma and the assignment operator.

# In[33]:


# Define `plus()`
def plus(a,b):
  sum = a + b
  return (sum, a)

# Call `plus()` and unpack variables 
sum, a = plus(3,4)

# Print `sum()`
print(sum)


# Function Arguments in Python
# 
# Arguments are the things which are given to any function or method call, while the function or method code refers to the arguments by their parameter names. 

# In[34]:


# Define `plus()` function
def plus(a,b=2):
  return a + b
  
# Call `plus()` with only `a` parameter
plus(a=1)

# Call `plus()` with `a` and `b` parameters
plus(a=1, b=3)

