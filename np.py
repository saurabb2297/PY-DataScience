import numpy as np

#creation of array
np1 = np.array([1,2,3,4,5])
print(np1,type(np1))

#creation of matrix
np2 = np.array([[1,2,3],[4,5,6]])
print(np2,np2.shape,type(np2))#type will be same as its still array , shape gives rows,cols
print(np2.dtype) #to get the datatype of elements in matrix
#on storing different data types in mumpy array all will get converted to string by default during initialization
#later modifying element to different data type will result in error
np2[0,0] = 7
print(np2)
#np2[0,1] = 'str' #error
#print(np2)


np3 = np.arange(0,10,2) #excludes last value i.e 10 , 3rd parametre specifies interval
print(np3,type(np3),np3.shape)

np4 =  np.linspace(0,10,20) #includes last value , 3rd parameter specifies number of elements
print(np4,type(np4),np4.shape)

np5 = np.random.rand(5,5) #matrix with random number b/w 0 and 1
print(np5)
np6 = np.random.randn(5,5) #also includes negative num
print(np6)

print(np6[0:3,1:3]) #accesing range of rows and columns from matrix

mat = np.arange(9).reshape(3,3)
print(mat)
print(np.diag(mat,k=0)) #extracts diagonal from given matrix
print(np.diag(mat,k=1)) #k is used to specify which diagonal upper diaogonal or lower 1,-1 or default i.e 0

np7 = np.zeros((2,2),dtype=int) #creating an array or matrix filled with zeroes with specified shape and dtype
print(np7)

np8 = np.ones((2,2),dtype=str)
print(np8)


