import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv('Movie_regression.csv' , header = 0)
print(df.head())
print(df.info()) #to get summary of data  types and count of each variable

#Now from info it is evedent that there are some null entries , so how to treat missing values
# best way is to fill mean avg value of all other preent data in place of missing data

mn = df['Time_taken'].mean() #finding out mean for time taken column
df['Time_taken'].fillna(value = mn,inplace = True)
#replacing null values with mean , to get the changes reflected in df inplace must be true

print(df.info())


#now we will see creation of dummy variable to convert categorical variable into numeric variables
#for ex in our data set there are two categorical var 3d_available and genre
#number of variable required to convert categorical variable is 1 less than the no. of categories

df = pd.get_dummies(df,columns=['3D_available','Genre'],drop_first=True)
#if drop_first will be false then same no. of variables will be created as the number of categories

print(df.head())


#now we will divide df into two i.e dependent and independent variables
#we already know that in our df only collection is dependent others are independent

x = df.loc[:,df.columns != 'Collection'] #as we need all rows and column except collection column
y = df['Collection']
print(type(x),type(y))
print(x.shape,y.shape)
print(x.head())

#now we will agin split our data in 80:20 ratio 80% data for learning and 20% for testing
#for this we will use sklearn module

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state=0)
#random_state helps in getting same test data each time which helps us to evaluate performance
print(x_train.shape,x_test.shape)


#creation of decision tree

rtree = tree.DecisionTreeRegressor(max_depth=3)
#function to create regression decision tree
#instead of max_depth min_samples_split can also be used  , the min no. of samples required for splitting
#min_samples_leaf which specifies minimum no. of samples required at each leaf node
#all these 3 are helpful in prunning the tree or minimising tree growth , multiple conditions can also be used
rtree.fit(x_train,y_train)

#now our regression tree is ready so we will predict values now
y_train_pred = rtree.predict(x_train)
y_test_pred =   rtree.predict(x_test)

#now we have both created regression decision tree and predicted values
#we will evaluate the performance of our model

from sklearn.metrics import mean_squared_error,r2_score
print(mean_squared_error(y_test,y_test_pred)) #comparing the predicted and original values
print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))
#signifacnt valuation is when we are findin r2_score for test data as train data we have used for training
#if r2_score > 0.8 for test data excellent 0.2 to 0.8 is ok


#plotting decision tree

dot_data = tree.export_graphviz(rtree,out_file=None,feature_names=x_train.columns,filled=True)
#to fill color and specifying feature name instead of default name as x0,x1.......

from IPython.display import  Image
import pydotplus

grph = pydotplus.graph_from_dot_data(dot_data)
Image(grph.create_png())
#allthough we are using correct way the image of tree will not be visible using pycharm and some error
#will be there i.e due to absence of graphviz app ,will run in conda environment
