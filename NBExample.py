# For handling datbases
import pandas as pd
# For plotting graphs
from matplotlib import pyplot as plt
# For implementing Naive bayes 
from sklearn.naive_bayes import GaussianNB
df= pd.read_csv('D:/NC00486885/TECHM/nbExample.csv')
#print(df.head())

plt.xlabel('Features')
plt.ylabel('Survived')
#plt.show()

X=df.loc[:,'Age']
Y=df.loc[:,'Survived']
plt.scatter(X,Y,color='green',label='Age')

X=df.loc[:,'Year']
Y=df.loc[:,'Survived']
plt.scatter(X,Y,color='blue',label='Year')

X=df.loc[:,'Nodes']
Y=df.loc[:,'Survived']
plt.scatter(X,Y,color='magenta',label='Nodes')

plt.legend(loc=4, prop={'size':5})
plt.show()

X=df.loc[:,'Age':'Nodes']
Y=df.loc[:,'Survived']

clf=GaussianNB()

#train the model
clf.fit(X,Y)

#prediction=
print(clf.predict([[33,61,1],[35,65,2]]))
#print(prediction)