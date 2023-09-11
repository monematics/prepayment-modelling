#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
import glob
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf


# In[2]:


#header = ['NAME','ADDRESS','PHONE','EMAIL']
# with pipe delimter data
with open("sample_orig_2020.txt", mode='r') as text_pipe, open("output.csv", 'w', newline= '') as file_comma:
    reader_pipe = csv.reader(text_pipe, delimiter = '|')
    writer_delim = csv.writer(file_comma, delimiter = ',')
    #writer_delim.writerow(header) # add header
    for record_index, row in enumerate(reader_pipe, 1): # loop to read file
        row.insert(0, record_index)
        writer_delim.writerow(row) # print value in each line


# In[3]:


var_names=["Credit Score",
"First Payment Date",
"First Time Homebuyer Flag",
"Maturity Date",
"Metropolitan Statistical Area (MSA) Or Metropolitan Division",
"Mortgage Insurance Percentage (MI %)",
"Number of Units",
"Occupancy Status",
"Original Combined Loan-to-Value (CLTV)",
"Original Debt-to-Income (DTI) Ratio",
"Original UPB",
"Original Loan-to-Value (LTV)",
"Original Interest Rate",
"Channel",
"Prepayment Penalty Mortgage (PPM) Flag",
"Amortization Type (Formerly Product Type)",
"Property State",
"Property Type",
"Postal Code",
"Loan Sequence Number",
"Loan Purpose",
"Original Loan Term",
"Number of Borrowers",
"Seller Name",
"Servicer Name",
"Super Conforming Flag",
"Pre-HARP Loan Sequence Number",
"Program Indicator",
"HARP Indicator",
"Property Valuation Method",
"Interest Only (I/O) Indicator"]


# In[4]:


data=pd.read_csv("output.csv", names=var_names)


# In[5]:


data = data.set_index("Loan Sequence Number")


# In[6]:


data


# In[7]:


data=data.drop(columns=["First Time Homebuyer Flag",
                   "Metropolitan Statistical Area (MSA) Or Metropolitan Division",
                   "Mortgage Insurance Percentage (MI %)",
                   "Number of Units",
                   "Original Combined Loan-to-Value (CLTV)",
                   "Channel",
                   "Prepayment Penalty Mortgage (PPM) Flag",
                   "Property Type",
                   "Amortization Type (Formerly Product Type)",
                   "Postal Code",
                   "Number of Borrowers",
                "Seller Name",
"Servicer Name",
"Super Conforming Flag",
"Pre-HARP Loan Sequence Number",
"Program Indicator",
"HARP Indicator",
"Property Valuation Method",
"Interest Only (I/O) Indicator"])


# In[8]:


data


# In[9]:


#header = ['NAME','ADDRESS','PHONE','EMAIL']
 
# with pipe delimter data
with open("sample_svcg_2020.txt", mode='r') as text_pipe, open("output_svcg.csv", 'w', newline= '') as file_comma:
    reader_pipe = csv.reader(text_pipe, delimiter = '|')
    writer_delim = csv.writer(file_comma, delimiter = ',')
 
    #writer_delim.writerow(header) # add header
 
    for record_index, row in enumerate(reader_pipe, 1): # loop to read file
        row.insert(0, record_index)
        writer_delim.writerow(row) # print value in each line


# In[10]:


var_names2=["LOAN SEQUENCE NUMBER",
           "MONTHLY REPORTING PERIOD",
           "CURRENT ACTUAL UPB",
           "CURRENT LOAN DELINQUENCY STATUS",
           "LOAN AGE",
           "REMAINING MONTHS TO LEGAL MATURITY",
           "DEFECT SETTLEMENT DATE",
           "MODIFICATION FLAG",
           "ZERO BALANCE CODE",
           "ZERO BALANCE EFFECTIVE DATE",
           "CURRENT INTEREST RATE",
           "CURRENT DEFERRED UPB",
           "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)",
           "MI RECOVERIES",
           "NET SALE PROCEEDS",
           "NON MI RECOVERIES",
           "EXPENSES",
           "LEGAL COSTS",
           "MAINTENANCE AND PRESERVATION COSTS",
           "TAXES AND INSURANCE",
           "MISCELLANEOUS EXPENSES",
           "ACTUAL LOSS CALCULATION",
           "MODIFICATION COST",
           "STEP MODIFICATION FLAG",
           "DEFERRED PAYMENT PLAN",
           "ESTIMATED LOAN TO VALUE (ELTV)",
           "ZERO BALANCE REMOVAL UPB",
           "DELINQUENT ACCRUED INTEREST",
           "DELINQUENCY DUE TO DISASTER",
           "BORROWER ASSISTANCE STATUS CODE",
           "CURRENT MONTH MODIFICATION COST",
           "INTEREST BEARING UPB",
          ]


# In[11]:


data_svcg=pd.read_csv("output_svcg.csv",low_memory=False, names=var_names2)


# In[12]:


data_svcg = data_svcg.set_index("LOAN SEQUENCE NUMBER")


# In[13]:


data_svcg=data_svcg.drop(columns=["CURRENT LOAN DELINQUENCY STATUS",
                                  "DEFECT SETTLEMENT DATE",
                                  "MODIFICATION FLAG",
                                  "CURRENT DEFERRED UPB",
                                  "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)",
                                  "MI RECOVERIES",
                                  "MODIFICATION COST",
                                  "STEP MODIFICATION FLAG",
                                  "DEFERRED PAYMENT PLAN",
                                  "ZERO BALANCE REMOVAL UPB",
                                  "DELINQUENT ACCRUED INTEREST",
                                  "DELINQUENCY DUE TO DISASTER",
                                  "BORROWER ASSISTANCE STATUS CODE",
                                  "CURRENT MONTH MODIFICATION COST",
                                  "INTEREST BEARING UPB",
                                  "MISCELLANEOUS EXPENSES",
                                  "TAXES AND INSURANCE",
                                  "MAINTENANCE AND PRESERVATION COSTS",
                                  "LEGAL COSTS",
                                  "EXPENSES",
                                  "NON MI RECOVERIES",
                                  "NET SALE PROCEEDS",
                                  "ZERO BALANCE EFFECTIVE DATE",
                                  "ACTUAL LOSS CALCULATION"
                                 ])


# In[14]:


data_svcg= data_svcg.fillna(0)


# In[15]:


data_svcg


# In[16]:


data2=data_svcg.iloc[0:785489,:]


# In[17]:


import math
#regions sin_t cos_t (current i -market i)
data_loan=set(data.index)
svcg_loan=set(data_svcg.index)
ind=0
subsize=785489
csarr=np.zeros(subsize)
dti=np.zeros(subsize)
upb=np.zeros(subsize)
oltv=np.zeros(subsize)
oi=np.zeros(subsize)
olt=np.zeros(subsize)

os_i=np.zeros(subsize)
os_p=np.zeros(subsize)
os_s=np.zeros(subsize)

lp_c=np.zeros(subsize)
lp_n=np.zeros(subsize)
lp_p=np.zeros(subsize)

ne_states =["ME","NH","VT","MA","RI","CT","NY","NJ","PA"]
#Maine, New Hampshire, Vermont, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, and Pennsylvania
#Ohio, Michigan, Indiana, Wisconsin, Illinois, Minnesota, Iowa, Missouri, North Dakota, South Dakota, Nebraska, and Kansas.
mw_states = ["OH","MI","IN","WI","IL","MN","IA","MO","ND","SD","NE","KS"]
#Delaware, Maryland, Virginia, West Virginia, Kentucky, North Carolina, South Carolina, Tennessee, Georgia, Florida, Alabama, Mississippi, Arkansas, Louisiana, Texas, and Oklahoma,Washington, DC
#Montana, Idaho, Wyoming, Colorado, New Mexico, Arizona, Utah, Nevada, California, Oregon, Washington, Alaska, and Hawaii.
w_states = ["MT","ID","WY","CO","NM","AZ","UT","NV","CA","OR","WA","AK","HI"]

midwest=np.zeros(subsize)
northeast=np.zeros(subsize)
west=np.zeros(subsize)
south=np.zeros(subsize)

sin_t=np.zeros(subsize)
cos_t=np.zeros(subsize)

for i in data2.index:
    csarr[ind]=int(data.loc[i,"Credit Score"])
    dti[ind]=int(data.loc[i,"Original Debt-to-Income (DTI) Ratio"])
    upb[ind]=int(data.loc[i,"Original UPB"])
    oltv[ind]=int(data.loc[i,"Original Loan-to-Value (LTV)"])
    oi[ind] = float(data.loc[i,"Original Interest Rate"])
    olt[ind] = float(data.loc[i,"Original Loan Term"])
    
    os_i[ind]=1 if data.loc[i,"Occupancy Status"] =='I' else 0
    os_p[ind]=1 if data.loc[i,"Occupancy Status"] =='P' else 0
    os_s[ind]=1 if data.loc[i,"Occupancy Status"] =='S' else 0
    
    lp_c[ind]=1 if data.loc[i,"Loan Purpose"] =='C' else 0
    lp_n[ind]=1 if data.loc[i,"Loan Purpose"] =='N' else 0
    lp_p[ind]=1 if data.loc[i,"Loan Purpose"] =='P' else 0
    
    northeast[ind]=1 if data.loc[i,"Property State"] in ne_states else 0
    midwest[ind]=1 if data.loc[i,"Property State"] in mw_states else 0
    west[ind]=1 if data.loc[i,"Property State"] in w_states else 0
    south[ind]=1 if data.loc[i,"Property State"] not in (ne_states + mw_states + w_states) else 0
    
    sin_t[ind] = math.sin((data_svcg.iloc[ind,0])%100)
    cos_t[ind] = math.cos((data_svcg.iloc[ind,0])%100)
    if ind%100==0:
        print(ind)
    ind+=1


# In[18]:


pd.options.mode.chained_assignment = None  # default='warn'
data2.loc[:,'cs']=csarr
data2.loc[:,'dti']=dti
data2.loc[:,'oupb']=upb
data2.loc[:,'oltv']=oltv
data2.loc[:,'oi']=oi
data2.loc[:,'olt']=olt

data2.loc[:,'os_i']=os_i
data2.loc[:,'os_p']=os_p
data2.loc[:,'os_s']=os_s

data2.loc[:,'lp_c']=lp_c
data2.loc[:,'lp_n']=lp_n
data2.loc[:,'lp_p']=lp_p

data2.loc[:,'northeast']=northeast
data2.loc[:,'midwest']=midwest
data2.loc[:,'west']=west
data2.loc[:,'south']=south

data2.loc[:,'sin_t']=sin_t
data2.loc[:,'cos_t']=cos_t


# In[ ]:





# In[ ]:





# In[19]:


data2


# In[20]:


#engineering on y
prev_loan=0
prev_p=0
p_act=np.zeros(subsize)
p_exp=np.zeros(subsize)
cpr=np.zeros(subsize)
ind2=0
labels=np.zeros(subsize)
for i in data2.index:
    cur_loan=i
    cur_p=data2.iloc[ind2,1]
    if cur_loan==prev_loan:
        p_act[ind2] = prev_p - cur_p
        r=data2.iloc[ind2,5]/100
        p0=data2.iloc[ind2,9]
        n=data2.iloc[ind2,12]
        p_exp[ind2] = (r*p0)/(1-(1+r)**(-n))-r*prev_p
        cpr[ind2] = (p_act[ind2]-p_exp[ind2])/prev_p
    else:
      labels[ind2]=1
    prev_p=cur_p
    prev_loan=cur_loan
    ind2+=1


# In[21]:


data2.loc[:,'p_act']=p_act
data2.loc[:,'labels']=labels
data2.loc[:,'p_exp']=p_exp
data2.loc[:,'cpr']=cpr


# In[22]:


data2[0:20]


# In[23]:


final_data = data2.drop(columns=["MONTHLY REPORTING PERIOD","REMAINING MONTHS TO LEGAL MATURITY","ZERO BALANCE CODE","p_act","labels","p_exp","oltv"])


# In[24]:


final_data


# In[25]:


final_data.dtypes


# In[26]:


final_data.to_csv('final_data_mbs.csv')  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# What version of Python do you have?
import sys

import tensorflow.keras


import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")

gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")


# In[ ]:


x=data_svcg.iloc[:,0:-1]


# In[ ]:


y=data_svcg["ESTIMATED LOAN TO VALUE (ELTV)"]


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


model = Sequential()
model.add(Dense(4096, input_dim=6, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(x, y, epochs=150, batch_size=10)


# In[ ]:


import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# In[ ]:




