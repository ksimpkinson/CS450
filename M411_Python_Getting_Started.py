# -*- coding: utf-8 -*-
"""
Edited on Sep 1 12:03:44 2017

@author: chilton
"""

# include these lines in your script but ignore them for now
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


### Start Here ###

# ARRAYS
# declare an array with four numbers, 10, 20, 30, 40
array1 = np.array([10, 20, 30, 40])
# print the elements of the array
print('array1 = ', array1)
# here is another way to display the elements of array 1,
# emphasizing that the first element of array1 has index 0,
# the second element in array1 has index 1, ...
print('the first element of array1 is %d' % array1[0])
print('the second element of array1 is %d' % array1[1])
print('the third element of array1 is %d' % array1[2])
print('the fourth element of array1 is %d' % array1[3])
print("\n")


# np.linspace can be used to create an array of evenly spaced numbers between
# a first (smallest) and last (largest) number
# array2 is a list of numbers starting at 0, ending at 5, and there are 11
# of them, so the spacing between each adjacent pair is 0.5
array2 = np.linspace(0, 5, 11, endpoint=True)
print('array2 = ', array2)
print('the length of array2 is ', len(array2))
print('\n')


# np.arange is similar to np.linspace, except instead of inputing the length
# of the array, you input the desired spacing. also, the last number is excluded.
# array3 is a array of numbers starting at 1, with equal spacing of 0.1, ending
# at 3.1 - 0.1 = 3.0
# this time, I will display array3 using the repr() function
array3 = np.arange(1, 3.1, 0.1)
print('array3 = ' + repr(array3))
print('the length of array3 is ' + repr(len(array3)))
print('\n')


# this shows another way to create an array, this time an array of integers
# form 0 to 9
array4 = np.array([i for i in range(10)])
print('array4 = ', array4)
print('the length of array4 is ', len(array4))
print('\n')


# now we create a 2 dimensional array (like a matrix), with 4 rows and 2 columns
# notice that the elements of array5 are very small, and essentially random
# this is because python is grabbing an available section of your computers
# membory and allocaitng it to array5, without initializing the values
# so array 5 has values equal to whatever was in the section of memory that was
# allocated to it
# then we raise e to the power of the numbers in array5
nrow = 3;
ncol = 2;
array5 = np.ndarray((nrow, ncol))
print('array5 = ')
print(array5)
print('\n')
print('exp(array5) = ')
print(np.exp(array5))
print('\n')


# array6 starts out as an empty array, then we add (append) 1,2,3,
# then we append 4,5,6
array6 = np.ndarray(0)
print('array6 = ')
print(array6)
array6 = np.append(array6,[1,2,3])
print('array6 = ')
print(array6)
array6 = np.append(array6,[4,5,6])
print('array6 = ')
print(array6)
print('\n')


# FOR loops
# range(1, 11) is an array of the integers 1, 2, ..., 10
# this FOR loop sums these intgers one at a time
sum = 0
for x in range(1, 11):
    sum = sum + x
    print('x = ', x)
    print('sum = ', sum)
print('\n')


# here we use a for loop to compute 5!
# x takes on the values 1, 2, 3, 4, 5
# note: this is not a good use of a FOR loop
# there is a built in function for calculationg factorials
fac = 1
for x in range(1, 6):
    fac = fac * x
    print('%d factorial = %d' % (x, fac))
from scipy.special import factorial
print('5 factorial = ', factorial(5))
print('\n')


# here we compute the sum of the squared differences between the numbers
# 1.0, 2.0, 3.0, 4.0, 5.0 and their square roots, 1., 1.41421, 1.73205, 2., 2.23607
# in other owrds, (1-1)^2 + (2 - 1.41421)^2 + ... + (5 - 1.23607)^2
a = np.linspace(1, 5, 5, endpoint=True)
b = np.sqrt(a)
sse = 0
for i in range(5):
    sse = sse + np.power(a[i]-b[i], 2)
print('sse = %f' % sse)
print('\n')


# WHILE loops
# here we add the numbers 0 + 1 + 2 + 3 + ... until the sum exceeds 100
# we then report the number of additions required
sum = 0
x = 0
while (sum < 100):
    x = x + 1
    sum = sum + x
    print('sum = %d' % (sum))
print('sum = %d required %d additions' % (sum, x))
print('\n')


# compute x! for x = 1, 2, 3, ... until x! > 100000
fac = 1
x = 0
while (fac < 100000):
    x = x + 1
    fac = fac * x
print('%d factorial = %d' % (x, fac))
print('%d factorial = %d' % (x-1, fac/x))
print('\n')


# start with x = 1 and divide by 2, then divide by 2 again, ...
# while x > 10^(-4) or equivalently, until x < 10^(-4)
# report the number of divisions by 2 that occured
x = 1
k = 0
while (x > np.power(10.0,-4)):
    x = x/2.0
    k = k + 1
print('x = %f   k = %d' % (x, k))
print('\n')


# DEFINE your own function
# create a funciton that computes the euclidean norm (or 2-norm) of a vector
# this shows two different ways to print the number, the first is the default
# the second allows you to control the numer of decimal places that are printed
def norm2(ar):
    return np.sqrt(np.dot(ar,ar))

a = np.random.randn(5)
print('a = ')
print(a)
print("length of a =", norm2(a))
print('length of a = %16.14f' % norm2(a))
print('\n')


# this functions inputs an array x and outputs another array where the function
# e^(-x) cos(x) is applied to each element of x
def f(x):
    return np.exp(-x)*np.cos(x)

x = np.linspace(0, 1, 11, endpoint=True)
y = f(x)
print('x = ')
print(x)
print('y = ')
print(y)
print('\n')


# create graphs using MATPLOTLIB
# these two plots illustrate some of the basic plotting tools needed for
# plotting functions of 1 variable, including solid lines, dahsed lines,
# plotting points, using a legend, labeling axes, plot title, etc.
# plot the previous vectors x (horizontal axis) and y (vertical axis)
# in PythonAnywhere, you need to save the figure to a file
fig = plt.figure()
plt.plot(x, y, 'g--')
plt.title(r"$f(x) = e^{-x} \cos(x)$", fontsize=16, color='darkblue')
plt.xlabel(r'\em{x}', fontsize=14, color='darkblue')
plt.ylabel(r'\em{y}', fontsize=14, color='darkblue')
fig.savefig("fig1.png")

# plot sin and cosine functions on interval [0, 2 pi]
fig = plt.figure()
t = np.linspace(0, 2*np.pi, 100, endpoint=True)
plt.plot(t, np.sin(t), 'b-', label='sin(x)')
plt.plot(t, np.cos(t), 'ro', label='cos(x)')
plt.legend(loc = 'lower left')
plt.xlabel('x')
plt.ylabel('y')
fig.savefig("fig2.png")
