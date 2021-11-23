# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:52:17 2020

@author: farah
"""
#add all libraries needing to be used 
import pandas as pd 
import numpy as np
import scipy as sp 
from scipy import stats
import matplotlib as mp
import datetime as dt
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# PART 1 
#importing excel csv file from computer 
kordsa_data = pd.read_csv('C:/Users/farah/Desktop/FALL 2020-21/OMIS 3020/data_raw_data.csv')

# a) Analyze the data types for each column and impute missing data for each column accordingly 
print(kordsa_data)
print(kordsa_data.shape)
print(kordsa_data.info())


#the data info shows the rows indexed upto 1923 but in the raw data there are only 1851 rows with data 
#removing the data with more than 4 columns of missings data 
kordsa_data= kordsa_data.dropna(thresh=4)
print(kordsa_data)

# b) Which data imputation strategy (e.g., mean, most frequent, etc.) should be used for each column?

#for data imputation 'Order ID' should be 'most frequent'
imp_order_id=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_order_id.fit(kordsa_data[['Order ID']])
kordsa_data['Order ID']=imp_order_id.fit_transform(kordsa_data[['Order ID']]).ravel()

kordsa_data['Order ID']=kordsa_data['Order ID'].astype('int')

#for data imputation 'Product ID' should be 'most frequent' 
imp_product_id=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_product_id.fit(kordsa_data[['Product ID']])
kordsa_data['Product ID']=imp_product_id.fit_transform(kordsa_data[['Product ID']]).ravel()

#for data imputation 'Amount Kg' should be 'mean' --> then converting float to int dt
imp_amount=SimpleImputer(missing_values=np.nan, strategy='mean')
imp_amount.fit(kordsa_data[['Amount (kg)']])
kordsa_data['Amount (kg)']=imp_amount.fit_transform(kordsa_data[['Amount (kg)']]).ravel()

kordsa_data['Amount (kg)']=kordsa_data['Amount (kg)'].astype('int')

#for data imputation 'Order Confirm Date' should be 'most_frequent' 
imp_confirm_date=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_confirm_date.fit(kordsa_data[['Order Confirm Date']])
kordsa_data['Order Confirm Date']=imp_confirm_date.fit_transform(kordsa_data[['Order Confirm Date']]).ravel()

#for data imputation 'Required Delivery Date' should be 'constant'  
imp_delivery_date=SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=('2020-12-31'))
imp_delivery_date.fit(kordsa_data[['Required Delivery Date']])
kordsa_data['Required Delivery Date']=imp_delivery_date.fit_transform(kordsa_data[['Required Delivery Date']]).ravel()

#print imputed values dataframe
print(kordsa_data)

# PART 2
print(kordsa_data.info())
# the dtypes for Order Confirm Date and Required Delivery Date are 'objects' 

#converting Dates from 'object' data type to 'datetime' data types
kordsa_data['Order Confirm Date'] = pd.to_datetime(kordsa_data['Order Confirm Date'])
kordsa_data['Required Delivery Date'] = pd.to_datetime(kordsa_data['Required Delivery Date'])

# a) Add three new series to your data frame that show the demand lead time 
#    for each order in days, weeks, and months, respectively. 

kordsa_data['DL Days']= (kordsa_data['Required Delivery Date']-kordsa_data['Order Confirm Date']).dt.days
print(kordsa_data['DL Days'])

kordsa_data['DL Weeks']= (kordsa_data['Required Delivery Date']) - (kordsa_data['Order Confirm Date'])
kordsa_data['DL Weeks']= kordsa_data['DL Weeks']/np.timedelta64(1,'W')
print(kordsa_data['DL Weeks'])

kordsa_data['DL Months']= (kordsa_data['Required Delivery Date'])-(kordsa_data['Order Confirm Date'])
kordsa_data['DL Months']= kordsa_data['DL Months']/np.timedelta64(1,'M')
print(kordsa_data['DL Months'])

#removing negative discrepancies in orginal data and some negative calculation discrepancies 
kordsa_data= kordsa_data[(kordsa_data['DL Days']>0) & (kordsa_data['DL Weeks']>0) & (kordsa_data['DL Months']>0)]
print(kordsa_data.info())

#rounding calculated data to integer values for better understanding 
kordsa_data['DL Weeks']= kordsa_data['DL Weeks'].astype(int).round()
kordsa_data['DL Months']= kordsa_data['DL Months'].astype(int).round()
#type for DL Days (timedelta64), DL Years (int64), DL Months (float64)

print(kordsa_data)

#PART 3 
# a) Create a pivot table that uses the demand lead time in months as index 
#    and sum up the amount requested for each order 
DLMonths_vs_Amount= pd.pivot_table(kordsa_data, values='Amount (kg)', index= 'DL Months', aggfunc = np.sum)
print (DLMonths_vs_Amount)

# b) Create a pie chart based on the pivot table, 
#    the pie chart should show the percentage of the 
#    amount with respect to lead time in months. 

label = DLMonths_vs_Amount.index
value = DLMonths_vs_Amount['Amount (kg)']
percent = 100.*value/value.sum()
patches, texts,  = plt.pie(value, startangle=90, radius=1.2)
labels = ['{0} Months - {1:1.2f} %'.format(i,j) for i,j in zip(label, percent)]
plt.legend(patches, labels, loc='left center', bbox_to_anchor=(-0.1, 1.),fontsize=8)
plt.title('Amount Ordered vs Demand Lead time in Months')
plt.show()


#PART 4
# a) Create a pivot table that uses the product ID and sum up the amount requested for each order

ID_vs_Amount= pd.pivot_table(kordsa_data, values="Amount (kg)", index= "Product ID", aggfunc = np.sum)
print(ID_vs_Amount)

# b) Create a pie chart based on the pivot table you created for the previous question

label2 = ID_vs_Amount.index
value2 = ID_vs_Amount['Amount (kg)']
percent2 = 100.*value2/value2.sum()
patches2, texts2,  = plt.pie(value2, startangle=90, radius=1.2)
labels2 = ['Product ID ({0}) - {1:1.2f} %'.format(i,j) for i,j in zip(label2, percent2)]
plt.legend(patches2, labels2, loc='left center', bbox_to_anchor=(-0.1, 1.),fontsize=8)
plt.title('Amount Ordered vs Product ID')
plt.show()





