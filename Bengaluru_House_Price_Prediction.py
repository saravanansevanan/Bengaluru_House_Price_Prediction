#!/usr/bin/env python
# coding: utf-8

# # Predicting home prices in Bengaluru

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df1 = pd.read_csv("Bengaluru_House_Data.csv")
df1.head()


# In[5]:


df1.shape


# In[13]:


df1.groupby("area_type")["area_type"].agg("count")


# In[30]:


gr = df1.groupby("area_type")
gr


# In[45]:


df2 = df1.drop(["area_type","availability","balcony","society"], axis="columns")
df2.head()


# In[51]:


gr2= df2.groupby("size")['size'].agg('count')
gr2


# In[58]:


df2.isnull().sum()


# In[63]:


df3=df2.dropna()
df3.isnull().sum()


# In[64]:


df3.shape


# In[66]:


df3['size'].unique()


# In[67]:


df3['bhk'] = df3['size'].apply(lambda x : int(x.split(' ')[0]))
df3.head()


# In[68]:


df3['bhk'].unique()


# In[69]:


df3[df3.bhk>20]


# In[81]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
    


# In[84]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[133]:


def convert_sqrt_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return((float(tokens[0]))+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    


# In[134]:


convert_sqrt_to_num('2100 - 2850')


# In[135]:


df3.head()


# In[142]:


df3["total_sqft"].head(31)


# In[144]:


df4 = df3.copy()

df4.head()


# In[117]:


df4.loc[30]


# In[146]:


df4.head(20)


# In[147]:


#Feature Engineering and dimensionality reduction
df4.head()


# In[149]:


df4=df4.drop('sqft', axis = "columns")
df4.head()


# In[151]:


df5 = df4.copy()
df5.head()


# In[152]:


#price per square foot
df5["price_per_sqft"]=df5["price"]*100000/df5['total_sqft']
df5.head()


# In[153]:


len(df5['location'].unique())


# In[155]:


df5['location'] = df5['location'].apply(lambda x : x.strip())
df5.head()


# In[160]:


location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[172]:


location_stats_less_than_10 = location_stats[location_stats<10]
#len(location_stats_less_than_10)
len(location_stats[location_stats<10])


# In[171]:


len(df5['location'].unique())


# In[174]:


df5['location'] = df5['location'].apply(lambda x : 'others' if x in location_stats_less_than_10 else x)
len(df5['location'].unique())


# In[180]:


df5[df5['location']=='others'].head()


# In[181]:


#outliers removals
df5.head()


# In[182]:


df5[df5['total_sqft']/df5['bhk']<300].head()


# In[184]:


len(df5["location"])


# In[186]:


df6 = df5[~(df5['total_sqft']/df5['bhk']<300)]
df6.shape


# In[187]:


df6['price_per_sqft'].describe()


# In[192]:


#removing price per sqft(pps) outliers by location
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf["price_per_sqft"])
        sd = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf["price_per_sqft"]>(m-sd)) & (subdf["price_per_sqft"]<=(m+sd))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[199]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df['location']==location) & (df['bhk']==2)]
    bhk3 = df[(df['location']==location) & (df['bhk']==3)]
    plt.scatter(bhk2.total_sqft,bhk2.price, color = 'blue')
    plt.scatter(bhk3.total_sqft,bhk3.price, color = 'green')
    plt.xlabel("Total Sqare feet area")
    plt.ylabel('Price')
    plt.title(location)
   
plot_scatter_chart(df7,"Hebbal")


# In[200]:


#Plotting histograms
plt.hist(df7['price_per_sqft'], rwidth = .8)
plt.xlabel('Price_per_sqft')
plt.ylabel('Counts')


# In[201]:


df7.groupby('bath')['bath'].agg('count')


# In[202]:


df7['bath'].unique()


# In[204]:


df7[df7['bath']>10]


# In[205]:


plt.hist(df7['bath'], rwidth=.8)
plt.xlabel('No. of bathrooms')
plt.ylabel("counts")


# In[207]:


df7[df7['bath']>(df7['bhk']+2)]


# In[209]:


df8 = df7[df7['bath']<(df7['bhk']+2)]
df8.shape


# In[211]:


df9 = df8.drop(['size','price_per_sqft'],axis = 'columns')
df9.head()


# In[214]:


#Model building
dummies = pd.get_dummies(df9['location'])
dummies.head()


# In[218]:


df10 = pd.concat([df9,dummies.drop('others',axis='columns')],axis = 'columns')
df10.head()


# In[220]:


df11 = df10.drop('location',axis='columns')
df11.head()


# In[221]:


df11.shape


# In[222]:


X = df11.drop('price', axis = "columns")
X.head()


# In[223]:


y = df11['price']
y.head()


# In[224]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10) 


# In[226]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)


# In[228]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X,y, cv=cv)


# In[248]:


#to find out which algorithms gives best score we use grid search cv
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

def find_best_model_using_gridsearchcv(X,y):
#class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    algos ={
        'linear_regression' : {
            'model' : LinearRegression(),
            'params':{
                'normalize' : [True,False]             
            }
        },
        'lasso':{
            'model':Lasso(),
            'params': {
                'alpha':[1,2],
                'selection':['random', 'cyclic']
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'params':{
                'criterion':['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores=[]
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'],config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model':algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[238]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'copy_X' : [True, False],
                'fit_intercept' : [True, False],
                'n_jobs' : [1,2,3],
                'positive' : [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[249]:


#predicting price
def predict_price(location,sqrt,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqrt
    x[1]=bath
    x[2]=bhk
    if loc_index >= 0:
        x[loc_index]=1
    return lr.predict([x])[0]
predict_price('1st Phase JP Nagar',1000,2,2)
    


# In[250]:


predict_price('1st Phase JP Nagar',1000,3,3)


# In[252]:


predict_price('Indira Nagar',1000,2,2)


# In[253]:


predict_price('Indira Nagar',1000,3,3)


# In[254]:


import pickle
with open("bangalore_home_price_model.pickle", 'wb') as f:
    pickle.dump(lr,f)


# In[255]:


import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))


# In[ ]:




