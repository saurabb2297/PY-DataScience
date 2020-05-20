import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data1 = pd.read_csv('Customer.csv',header = 0)
print(data1.head(7))
data2 = pd.read_csv('Customer.csv',header =0 ,index_col = 0) #default index is 0,1,2.... to specicfy col as index
print(data2.head())

print(data1.describe()) #used to get stastics for the available numerical dats

print(data1.iloc[0]) # to get all data present at specified index

print(data2.loc["CG-12520"]) # to get data present at specified index value

#iloc can be used irrespective of we have used default indexing or specified col as index


#in pycharm we need to use matplotlib function show to visualise
plt.show(sb.distplot(data2.Age)) #histogram
plt.show(sb.distplot(data2.Age,kde=False,color='red')) #to remove kde line
print(help(sb.distplot)) # to check tha info and arguements of function

#seaborn comes with embedded datset in it namely iris , we will use that
iris = sb.load_dataset('iris')
print(iris.head())
print(iris.shape)

#scatter plot

plt.show(sb.jointplot('sepal_length','sepal_width',iris)) #(x,y,data)

#pair plot gives plot between all possible pairs of data

plt.show(sb.pairplot(iris))