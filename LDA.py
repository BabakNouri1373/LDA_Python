#!/usr/bin/env python
# coding: utf-8

# In[93]:


#------------------------------------------------------
#             Import Libary                 
#------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.pyplot as mp
import numpy as np
import pandas as pd
#------------------------------------------------------


# In[94]:


#------------------------------------------------------
#             Import DataSet                
#------------------------------------------------------
Example_DataSet= pd.read_csv('C:\\Users\\babak_Nouri\\Desktop\\Example_DataSet.txt')
SDS = pd.DataFrame(data=Example_DataSet)
SDS_Plot = pd.DataFrame(data=Example_DataSet)
SDS.head(12)
#------------------------------------------------------


# In[95]:


SDS_P = SDS[SDS["Label"] == 1]
SDS_P


# In[96]:


SDS_NP = SDS[SDS["Label"] == 0]
SDS_NP


# In[97]:


#------------------------------------------------------
#             Drop Label Column                
#------------------------------------------------------
SDS_NP.drop('Label', axis= 1, inplace= True)
SDS_P.drop('Label' , axis= 1, inplace= True)
SDS.drop('Label'   , axis= 1, inplace= True)
#------------------------------------------------------


# In[98]:


#------------------------------------------------------
#      calculate Dimention_Count  AND   Label_Count
#------------------------------------------------------
Dimention_Count = len(SDS.T.values)
print(Dimention_Count)
Df = pd.DataFrame()
Df['Pass'] = ['0', '1']
Label_Count=len(Df.values)
print(Label_Count)
#------------------------------------------------------


# In[99]:


#------------------------------------------------------
#      calculate  mean  OF vectors
#------------------------------------------------------
n_features   = Dimention_Count
n_classes    = Label_Count
classes      = [SDS,SDS_P,SDS_NP]
mean_vectors = []
for i in range(n_classes+1):
    mean_vectors.append(np.mean(classes[i]))
    print('--------------------------------')
    print('Mean Vector class :',i,'\n')
    print(mean_vectors[i])
print('--------------------------------')
#------------------------------------------------------


# In[100]:


#------------------------------------------------------
#     calculate  Between-class scatter matrix          
#------------------------------------------------------
Smean_SDS    =pd.DataFrame(data=mean_vectors[0])
Smean_SDS_P  =pd.DataFrame(data=mean_vectors[1])
Smean_SDS_NP =pd.DataFrame(data=mean_vectors[2])
SMean_List=[Smean_SDS_P,Smean_SDS_NP]
S_B_List=[]
SDS_P_Sample_Count =len(SDS_P.values)
SDS_NP_Sample_Count=len(SDS_NP.values)
N=[SDS_P_Sample_Count,SDS_NP_Sample_Count]
for i in range(0,n_classes):
    print('----------------------------------------------------------------------------')
    S_B_I=N[i]*np.dot(SMean_List[i]-Smean_SDS,(SMean_List[i]-Smean_SDS).T)
    S_B_List.append(S_B_I)
    print('----------------------------------------------------------------------------')
    print('S[',i,']:\n\n', S_B_List[i],'\n\n')
print('----------------------------------------------------------------------------')


S_B=0
for i in range(0,n_classes):
    S_B+=S_B_List[i]
print('Between-class scatter matrix :\n\n',S_B,'\n\n')
#------------------------------------------------------


# In[102]:


#------------------------------------------------------
#     calculate  within-class Scatter Matrix          
#------------------------------------------------------
Wmean_SDS    =mean_vectors[0]
Wmean_SDS_P  =mean_vectors[1]
Wmean_SDS_NP =mean_vectors[2]
WMean_List=[Wmean_SDS_P,Wmean_SDS_NP]
vectors_List =[SDS_P,SDS_NP]
S_W_List=[]


for i in range(0,n_classes):
    print('----------------------------------------------------------------------------')
    d_I=vectors_List[i]-WMean_List[i].T
    S_I=np.dot(d_I.T,d_I)
    S_W_List.append(S_I)
    print('----------------------------------------------------------------------------')
    print('S[',i,']:\n\n', S_W_List[i],'\n\n')
print('----------------------------------------------------------------------------')


S_W=0
for i in range(0,n_classes):
    S_W+=S_W_List[i]
print('within-class Scatter Matrix :\n\n',S_W,'\n\n')
#------------------------------------------------------


# In[103]:


#------------------------------------------------------
#     calculate  Scatter Matrix          
#------------------------------------------------------
S_W=pd.DataFrame(data=S_W)
S_B=pd.DataFrame(data=S_B)
w=np.linalg.inv(S_W).dot(S_B)
print(w)
#------------------------------------------------------


# In[104]:


#------------------------------------------------------
#     calculate  eign_vals AND  eign_vecs          
#------------------------------------------------------
import matplotlib.pyplot as plt
eign_vals, eign_vecs = np.linalg.eig(w)
print('\nEignvalues \n%s' % eign_vals,'\n\n eign_vecs\n')
print(eign_vecs)
plt.bar(range(0,2), eign_vals, alpha = 0.5, align= 'center', label= 'indivisual eign_vals')
plt.show()
#------------------------------------------------------


# In[105]:


#------------------------------------------------------
#     calculate  eign_vals AND  eign_vecs          
#------------------------------------------------------
eig_pairs = [(np.abs(eign_vals[i]), eign_vecs[:,i]) for i in range(len(eign_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])
#------------------------------------------------------


# In[111]:


#------------------------------------------------------
#       calculate  Percent OF  eign_vals      
#------------------------------------------------------
print('Variance explained:\n')
eigv_sum = sum(eign_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
#------------------------------------------------------


# In[113]:


#------------------------------------------------------
#               Scatter Matrix
#------------------------------------------------------
W = np.hstack((eig_pairs[0][1].reshape(2,1), eig_pairs[1][1].reshape(2,1)))
print('Matrix W:\n', W.real)
#------------------------------------------------------


# In[115]:


#------------------------------------------------------
#          calculate  New Space=DataSet * eign_vecs 
#------------------------------------------------------
eign_vecs = pd.DataFrame(data=eign_vecs)
SDS = pd.DataFrame(data=SDS)
SDS_lda = np.dot(SDS,eign_vecs)
assert SDS_lda.shape == (11,2), "The matrix is not 11x2 dimensional."
SDS_lda = pd.DataFrame(data=SDS_lda)
print(SDS_lda)
#------------------------------------------------------


# In[116]:


#------------------------------------------------------
#        Show New Space             
#------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(SDS_Plot['Label'])

plt.xlabel('N1')
plt.ylabel('N2')
plt.xlim(-8,8)
plt.ylim(-8,8) 
plt.scatter(
    SDS_lda[0],
    SDS_lda[1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='r'
)
#------------------------------------------------------


# In[ ]:




