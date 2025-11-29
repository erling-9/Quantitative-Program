# In[1]:


from datetime import datetime
from lunarcalendar.festival import festivals
import os
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
# from pingouin import ttest
import matplotlib.pyplot as plt
import sys 
from lightgbm import LGBMClassifier
import xgboost as xgb
import sklearn as sk
from sklearn import svm
from progressbar import *
import copy

import joblib
import pickle
import json

pb = ProgressBar()


# In[2]:


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)
pd.options.display.float_format = '{:,.4f}'.format


# In[3]:

datasourceDir = '…/datasource/'


# ### Define the XYZ meta model

# In[4]:


class XYZ_Meta():
    
    def __init__(self):
        super(XYZ_Meta,self).__init__()
        '''
            Global parameters
        '''
        self.raw_X = None
        self.raw_Y = None
        self.raw_W = None
        self.transaction_cost_Y = 0.0002
        self.transaction_cost_X = 0
        # self.ml_support = ML_Supportor()
        self.kernel = "rbf"
        self.Lambda = [50, 5]
        
        '''
            Parameters on LightGBM
        '''
        self.transaction_cost_lgbm_filter = 0.002
        self.lgbm_learning_rate = 0.005
        self.lgbm_min_child_samples = 50
        self.lgbm_reg_lambda = 0.02
        self.lgbm_unbalance_ratio = 5
        
        '''
            Parameters on KSVM
        '''
        self.transaction_cost_ksvm_filter = 0.002
        self.ksvm_gamma = 0.4
        self.ksvm_unbalance_ratio = 5

    '''
        Linear combination of average return and sharpe ratio
    '''
    
    def utility_function(self,sequence,transaction_cost):
        sequence = np.array(sequence)
        n=len(sequence)
        n1 = np.sum(sequence != 0 )
        #print("total days/trading days are\n")
        #print(n,n1)
        ave_ann_return = (np.mean(sequence) - transaction_cost*n1/n)*252 # average annual net return
        ave_daily_return_per_trade = (np.sum(sequence)/(n1+0.001)-transaction_cost) # average daily net return
        ave_ann_return_per_trade=ave_daily_return_per_trade*252
        ##this one measures the quality of trade...
        sequence_non_null = [i for i in sequence if i != 0]
        
        if len(sequence_non_null)==0:
            # print("\nNULL SEQUENCE DETECTED!")
            sharpe_ratio_trade = 0
        else:
            sharpe_ratio_trade = ave_daily_return_per_trade/(np.std(sequence_non_null)+0.0001)*np.sqrt(252)
            
        cut = 3
        cut_2 = 5
        if sharpe_ratio_trade < cut:
            uti = sharpe_ratio_trade + self.Lambda[0]*ave_ann_return+self.Lambda[1]*ave_daily_return_per_trade*252
        elif sharpe_ratio_trade < cut_2:
            uti = cut+(sharpe_ratio_trade-cut)**0.6 + self.Lambda[0]*ave_ann_return + \
                self.Lambda[1]*ave_daily_return_per_trade*252
        else:
            uti = cut+(cut_2-cut)**0.6 +(sharpe_ratio_trade-cut_2)**0.3 + self.Lambda[0]*ave_ann_return + \
                self.Lambda[1]*ave_daily_return_per_trade*252

        #print("uti is\n",uti,sharpe_ratio_trade,ave_ann_return,ave_daily_return_per_trade)
        return uti, sharpe_ratio_trade,ave_ann_return,ave_ann_return_per_trade
    
    def one_dimensional_filter(self, W, Y):
        nquantiles = 100
        grid_up = pd.Series(list(range(int(0.1*nquantiles), int(0.9*nquantiles))))/nquantiles
        grid_down = pd.Series(list(range(int(0.1*nquantiles), int(0.9*nquantiles))))/nquantiles
        c_up = np.quantile(W, grid_up)
        c_down = np.quantile(W, grid_down)
        #print("c_up is\n",c_up)
        #print("c_down is\n",c_down)
        utility_up = []
        utility_down = []
        sig_up = []
        sig_down = []
        for x in c_up:
            Y_filter = Y.copy()
            Y_filter.loc[W < x] = 0 # W < x
            
            # calculate utility
            uti_temp,sharpe_ratio_trade,ave_ann_return, \
                ave_ann_return_per_trade = self.utility_function(Y_filter, self.transaction_cost_Y)
            utility_up.append(uti_temp)
        
        for x in c_down:
            Y_filter = Y.copy()
            #print("x is",x)
            Y_filter.loc[W > x] = 0 # W > x

            # calculate utility
            uti_temp,sharpe_ratio_trade,ave_ann_return, \
                ave_ann_return_per_trade = self.utility_function(Y_filter,self.transaction_cost_Y)
            utility_down.append(uti_temp)

        i_up = np.argmax(utility_up)
        i_down = np.argmax(utility_down)
        i_up_m = np.argmin(utility_up)
        i_down_m = np.argmin(utility_down)
        
        utility_max = np.max(utility_up + utility_down)
        utility_min = np.min(utility_up + utility_down)
        if utility_max < abs(utility_min):
            pass
            #print("shorting Y|W is probably better")
#             print("")

        if np.max(utility_up) > np.max(utility_down):
#             print("model cut-off-up\n")
            Y_filter = Y
            Y_filter.loc[W < c_up[i_up]] = 0
            
            #Y_filter = Y.loc[W>c_up[i_up]]
            uti_temp, sharpe_ratio_trade, ave_ann_return, \
                ave_ann_return_per_trade = self.utility_function(Y_filter,self.transaction_cost_Y)
            
            return c_up[i_up],grid_up[i_up],1,sharpe_ratio_trade,ave_ann_return,ave_ann_return_per_trade,Y_filter
        
        else:
#             print("model cut-off-down\n")
            Y_filter = Y
            Y_filter.loc[W > c_down[i_down]] = 0
            
            #Y_filter = Y.loc[W < c_down[i_down]]
            uti_temp, sharpe_ratio_trade, ave_ann_return, \
                ave_ann_return_per_trade = self.utility_function(Y_filter,self.transaction_cost_Y)
            
            return c_down[i_down],grid_down[i_down],-1,sharpe_ratio_trade, \
                    ave_ann_return,ave_ann_return_per_trade,Y_filter

    '''
        XGBoost
    '''
    def XGBoost_filter_Weighted(self,Y,W,unbalance_ratio):
        '''
            Define XGBoost as the (supervised) binary classifier
        '''
        #,reg_lambda = 0.0001
        xgboost = xgb.XGBClassifier(objective = 'binary:logistic', learning_rate = 0.001, reg_lambda = 0.0,
                                    eval_metric = ['logloss', 'auc', 'error'])
        label = np.sign(Y - self.transaction_cost_Y)
        #unbalance_ratio = 3
        ##give more weights on negative - focus on positive and kill more negative Ys
        weight = abs(np.multiply(Y-self.transaction_cost_Y, (label - (unbalance_ratio-1)/(1+unbalance_ratio))))
        
        '''
            Fit the model: Build a gradient boosting model from the training set (X, y).
        '''
        xgboost.fit(W, label, sample_weight = weight.values.reshape(-1))
        
        return xgboost

    '''
        LGBM
    '''
    def LGBM_filter_Weighted(self,Y,W,unbalance_ratio):
        
        '''
            Define LGBM as the (supervised) binary classifier
        '''
#         lgbm = LGBMClassifier(learning_rate = 0.005, min_child_samples = 50, reg_lambda = 0.02)
        lgbm = LGBMClassifier(learning_rate = self.lgbm_learning_rate, 
                              min_child_samples = self.lgbm_min_child_samples, 
                              reg_lambda = self.lgbm_reg_lambda)
        
        label = np.sign(Y - self.transaction_cost_Y)
        # give more weights on negative - focus on positive and kill more negative Ys
#         weight = abs(np.multiply(Y-self.transaction_cost_Y, (label - (unbalance_ratio-1)/(1+unbalance_ratio))))
        weight = abs(np.multiply(Y-self.transaction_cost_Y, 
                                 (label - (self.lgbm_unbalance_ratio-1)/(1+self.lgbm_unbalance_ratio))))
        
        '''
            Fit the model: Build a gradient boosting model from the training set (X, y).
        '''
        lgbm.fit(W,label,sample_weight = weight.values.reshape(-1))
        return lgbm
    
    # add transaction cost
    def LGBM_filter_Weighted_L(self,Y,W,unbalance_ratio,transaction_cost):
        '''
            Define LGBM as the (supervised) binary classifier
        '''
        lgbm = LGBMClassifier(learning_rate = 0.005, min_child_samples = 50, reg_lambda = 0.5)
        
        label = np.sign(Y - transaction_cost) 
        # give more weights on negative - focus on positive and kill more negative Ys
        weight = abs(np.multiply(Y-transaction_cost, (label - (unbalance_ratio-1)/(1+unbalance_ratio))))
        
        '''
            Fit the model: Build a gradient boosting model from the training set (X, y).
        '''
        lgbm.fit(W,label,sample_weight = weight.values.reshape(-1))
        return lgbm
    
    '''
        Kernel SVM: Fit the SVM model according to the given training data.
    '''
    def Kernel_SVM_Weighted(self,Y,W,unbalance_ratio):
        '''
            Define SVM as the (supervised) binary classifier
        '''
        
#         clf = svm.SVC(kernel=self.kernel, gamma=0.4, probability=True, random_state=42)
        clf = svm.SVC(kernel=self.kernel, gamma=self.ksvm_gamma, probability=True, random_state=42)
        label = np.sign(Y-self.transaction_cost_Y) # compared with NO-trading
        
#         unbalance_ratio = 3
        #give more weights on negative - focus on positive and kill more negative Ys
        unbalance_ratio = self.ksvm_unbalance_ratio
        weight = abs(np.multiply(Y-self.transaction_cost_Y, (label - (unbalance_ratio-1)/(1+unbalance_ratio))))
        
        clf.fit(W, label, sample_weight= weight)
        return clf

    ###Generate two dimensional hotmap
    def test_2d(self,W,Y,sty,name,precision):
        #XYZ_Model = XYZ_Meta()
        h_m = self.multi_dimensional_grid_mean_2d(Y,W,sty)
        #print(h_m)
        fig, ax = plt.subplots()
        im = ax.imshow(h_m)
        n,p = h_m.shape
        col_names = W.columns.values.tolist()
        ax.set_xlabel(col_names[1])
        ax.set_ylabel(col_names[0])
        for i in range(n):
            for j in range(p):
                if sty == "number":
                    ax.text(j, i, int(h_m[i, j]/precision),ha="center", va="center", color="w")
                elif sty == "sum":
                    ax.text(j, i, int(h_m[i, j]*precision)/10,ha="center", va="center", color="w")
                elif sty == "mean":
                    ax.text(j, i, int(h_m[i, j]*precision)/100,ha="center", va="center", color="w")
        fig.tight_layout()
        plt.show()
        #if sty == "sum":
        plt.savefig("hot_figure_"+sty+"_"+name+".png") 

    '''
        Summarize the average return and sharpe ratio of a given strategy defined by a sequence
    '''
    def summarize(self, sequence):
        mean_r = np.mean(sequence)
        sequence1 = sequence.loc[sequence.values.reshape(-1) != 0]
        annual_mean_r = (np.sum(sequence) - self.transaction_cost_Y *len(sequence1))/len(sequence)*252
        if len(sequence1)>0:
            mean_r_trade = np.mean(sequence1)
            sharpe_per_trade = (np.mean(sequence1)-self.transaction_cost_Y)/(np.std(sequence1)+0.0001)*np.sqrt(252)
            win_rate = np.sum(sequence1>self.transaction_cost_Y)/len(sequence1)
        else:
            mean_r_trade = 0
            sharpe_per_trade = 0
            win_rate = 0
            
        percent_of_trade = len(sequence1)/len(sequence) 


    def summarizePrint(self, sequence):
        mean_r = np.mean(sequence)
        sequence1 = sequence.loc[sequence.values.reshape(-1) != 0]
        annual_mean_r = (np.sum(sequence) - self.transaction_cost_Y *len(sequence1))/len(sequence)*252
        if len(sequence1)>0:
            mean_r_trade = np.mean(sequence1)
            sharpe_per_trade = (np.mean(sequence1)-self.transaction_cost_Y)/(np.std(sequence1)+0.0001)*np.sqrt(252)
            win_rate = np.sum(sequence1>self.transaction_cost_Y)/len(sequence1)
        else:
            mean_r_trade = 0
            sharpe_per_trade = 0
            win_rate = 0
            
        percent_of_trade = len(sequence1)/len(sequence) 
        
        print("annual mean_return %.5f"%annual_mean_r)
        print("mean_return_per_trade %.5f"%mean_r_trade)
        print("sharpe_per_trade %.5f"%sharpe_per_trade)
        print("percent of trade %.5f"%percent_of_trade)
        print("winning rate %.5f"%win_rate)

    def summarizeReturn(self, sequence):
        mean_r = np.mean(sequence)
        sequence1 = sequence.loc[sequence.values.reshape(-1) != 0]
        annual_mean_r = (np.sum(sequence) - self.transaction_cost_Y *len(sequence1))/len(sequence)*252
        if len(sequence1)>0:
            mean_r_trade = np.mean(sequence1)
            sharpe_per_trade = (np.mean(sequence1)-self.transaction_cost_Y)/(np.std(sequence1)+0.0001)*np.sqrt(252)
            win_rate = np.sum(sequence1>self.transaction_cost_Y)/len(sequence1)
        else:
            mean_r_trade = 0
            sharpe_per_trade = 0
            win_rate = 0
            
        percent_of_trade = len(sequence1)/len(sequence) 
        return annual_mean_r, mean_r_trade, sharpe_per_trade, percent_of_trade, win_rate
    
    def summarize_portfolio(self, sequence, loading):        
        mean_r = np.mean(sequence)        
        sequence1 = sequence.loc[sequence.values.reshape(-1) != 0]        
        annual_mean_r = (np.sum(sequence) - self.transaction_cost_Y *np.sum(loading))/len(sequence)*252        
        if len(sequence1) > 0:            
            mean_r_trade = np.sum(sequence1)/np.sum(loading)
            # sharpe ratio
            sharpe_mean = (np.sum(sequence1)/np.sum(loading) - 
                           self.transaction_cost_Y*np.mean(loading)*len(sequence)/len(sequence1))
            sharpe_per_trade = sharpe_mean/(np.std(sequence1)+0.0001)*np.sqrt(252)            
            win_rate = np.sum(sequence - self.transaction_cost_Y*loading>0)/len(sequence1)        
        else:            
            mean_r_trade = 0            
            sharpe_per_trade = 0           
            win_rate = 0 

        percent_of_trade = len(sequence1)/len(sequence) 
            
        print("annual mean_return %.5f"%annual_mean_r)        
        print("mean_return_per_trade %.5f"%mean_r_trade)        
        print("sharpe_per_trade %.5f"%sharpe_per_trade)        
        print("percent of trade %.5f"%percent_of_trade)        
        print("winning rate %.5f"%win_rate)
        


# In[5]:


def test_model(W, Y, model, unbalance_ratio, XYZ_Model):
    # XYZ_Model = XYZ_Meta()
#     print("testing model " + model)
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(W, Y, test_size = 0.1, shuffle = False)
    ## test light gbm
    if model == "LGBM":
        model_temp = XYZ_Model.LGBM_filter_Weighted(y_train, X_train, unbalance_ratio)
    # print(W_all-1)
    # print(lgbm_model)
        pred_proba = model_temp.predict_proba(X_train)
    elif model == "KSVM":
        model_temp = XYZ_Model.Kernel_SVM_Weighted(y_train, X_train, unbalance_ratio)
        pred_proba = model_temp.predict_proba(X_train)
    elif model == "XGBoost":
        model_temp = XYZ_Model.XGBoost_filter_Weighted(y_train, X_train, unbalance_ratio)
        pred_proba = model_temp.predict_proba(X_train)
    else:
        print("model does not exist")
        return None
    
    '''
        Optimize the cutoff
    '''
    # print(pred_proba[:,0],np.std(pred_proba[:,0]))
    pos_proba = pred_proba[:, 1]
        
    cut_off, trade_percent, sig, sharpe_ratio_trade, ave_ann_return, ave_ann_return_per_trade, Y_filter = \
        XYZ_Model.one_dimensional_filter(pos_proba, y_train)
#     print("The result in the training sample is")
    XYZ_Model.summarize(Y_filter)
    
    ##now look at out of sample
#     print('--> The learned cutoff is', cut_off, 'with sign', sig)
        
    pos_proba_test_all = model_temp.predict_proba(X_test)
    pos_proba_test = pos_proba_test_all[:,1]
    
    if sig == 1:
        y_test.loc[pos_proba_test < cut_off] = 0
        y_test1 = y_test.loc[pos_proba_test >= cut_off]
    else:
        y_test.loc[pos_proba_test > cut_off] = 0
        y_test1 = y_test.loc[pos_proba_test <= cut_off]
#     print("The result in the testing sample is")
#     XYZ_Model.summarize(y_test)
    return y_test, model_temp, cut_off


# In[6]:


def test_model_two_sides(W, Y, model, unbalance_ratio, XYZ_Model):
    
    y_test_long, model_long, cutoff_long = test_model(W, Y, model, unbalance_ratio, XYZ_Model) # Long strategy
    y_test_short, model_short, cutoff_short = test_model(W, -Y, model, unbalance_ratio, XYZ_Model) # Short strategy
    
    y_test_all = y_test_long + y_test_short # Combine long and short strategies
#     print("--> The over all strategy in testing sample:")
    XYZ_Model.summarize(y_test_all)
    
    y_test1 = y_test_all.loc[y_test_all.values.reshape(-1) != 0]
    
    return y_test_all, model_long, model_short, cutoff_long, cutoff_short


# ----------------------------

# ### Input data and construct filters

# In[7]:


xyzReturnsFinal = pd.read_csv("a50futures_filter_20mins_stop_20250615.csv")
xyzReturnsFinal = xyzReturnsFinal.sort_values(by = 'date_day', ascending = True)
numericalVars = [x for x in xyzReturnsFinal.columns if ('date' not in x) and ('Date' not in x)]
xyzReturnsFinal[numericalVars] = xyzReturnsFinal[numericalVars].astype(float)
print('Shape of dataframe <xyzReturnsFinal> is %s.'%str(xyzReturnsFinal.shape))
print('Sample period ranges from %s'%xyzReturnsFinal['date_day'].min() + ' to %s.'%xyzReturnsFinal['date_day'].max())
print('%s days in total.'%len(xyzReturnsFinal['date_day'].unique()))
xyzReturnsFinal.head(1)



startTime = '0400'
endTime = '0929'
print('Start Time is %s'%startTime + ', End Time is %s.'%endTime)

varY = 'y_a50f_' + startTime + '_' + endTime

varW = ['w_a50f_1500t1_' + startTime, 
        'w_usdrmb_1500t1_' + startTime + '_raw',
        'w_CSI_1445t1_1500', 
        'w_CSI_1500t2_1500t1']

# parameters on XYZ meta model
unbalance_ratio = 5

# length of training data and testing data
datesList = list(xyzReturnsFinal['date_day'].unique())
datesList.sort()

dateLength = len(datesList)
windowLength = 1000
testingRatio = 0.1
trainingLength = windowLength*(1-testingRatio)
testingLength = windowLength*testingRatio
windowCnt = int(np.floor((dateLength-trainingLength)/testingLength))


# In[ ]:


# Initialize result dataframe
testAll_ensembler = pd.DataFrame()

testAll_lgbm = pd.DataFrame()
y_test_all_lgbm = pd.DataFrame()
testAll_ksvm = pd.DataFrame()
y_test_all_ksvm = pd.DataFrame()

for windowIndex in range(0, windowCnt, 1):
    dateSub = datesList[
        int(windowIndex * testingLength) : int(windowIndex * testingLength + trainingLength + testingLength)]
    dateTrain = datesList[
        int(windowIndex * testingLength) : int(windowIndex * testingLength + trainingLength)]
    dateTest = datesList[
        int(windowIndex * testingLength + trainingLength) : int(windowIndex * testingLength + trainingLength + testingLength)]
    
    if windowIndex%5 == 0:
        print('Sample period ranges from %s'%min(dateSub) + ' to %s'%max(dateSub) + ' with length %s.'%len(dateSub))
    
    xyzReturnsRolling = xyzReturnsFinal[xyzReturnsFinal['date_day'].isin(dateSub)]
    
    # remove rows with NaN
    xyzReturnsRolling = xyzReturnsRolling[(~xyzReturnsRolling[varY].isna())]
    if 'w_CSI_1445t1_1500' in varW or 'w_CSI_1500t2_1500t1' in varW:
        xyzReturnsRolling = xyzReturnsRolling[(~xyzReturnsRolling['w_CSI_1445t1_1500'].isna())] 
        xyzReturnsRolling = xyzReturnsRolling[(~xyzReturnsRolling['w_CSI_1500t2_1500t1'].isna())]  
    
    xyzReturnsRolling = xyzReturnsRolling.sort_values(by = 'date_day')
    
    # define filters and return
    W = xyzReturnsRolling[varW]
    Y = xyzReturnsRolling[varY]
    
    # Input variables on return and filters
    XYZ_Model = XYZ_Meta()

    '''
        Model fitting
    '''
    # --> lightGBM
    y_test_lgbm, model_long_lgbm, model_short_lgbm, cutoff_long_lgbm, cutoff_short_lgbm = test_model_two_sides(
        W, Y, 'LGBM', unbalance_ratio, XYZ_Model)
    
    if y_test_all_lgbm.shape[0] == 0:
        y_test_all_lgbm = y_test_lgbm.copy()
    else:
        y_test_all_lgbm = pd.concat([y_test_all_lgbm, y_test_lgbm], axis = 0)
        
    # --> KSVM
    XYZ_Model = XYZ_Meta()
    y_test_ksvm, model_long_ksvm, model_short_ksvm, cutoff_long_ksvm, cutoff_short_ksvm = test_model_two_sides(
        W, Y, 'KSVM', unbalance_ratio, XYZ_Model)
    
    if y_test_all_ksvm.shape[0] == 0:
        y_test_all_ksvm = y_test_ksvm.copy()
    else:
        y_test_all_ksvm = pd.concat([y_test_all_ksvm, y_test_ksvm], axis = 0)
    
    
    '''
        Testing dataframe: LightGBM
    '''
    testDf = xyzReturnsRolling[xyzReturnsRolling['date_day'].isin(dateTest)][varW + [varY, 'date_day']]
    testDf = testDf.sort_values(by = 'date_day')
    
    # predicted probability based on filters
    testDf_lgbm = testDf.copy()
    #testDf_lgbm['long_prob_lgbm'] = testDf_lgbm[varW].apply(
        #lambda x: model_long_lgbm.predict_proba(np.array(x).reshape(1, -1))[0, 1], axis = 1)
    #testDf_lgbm['short_prob_lgbm'] = testDf_lgbm[varW].apply(
        #lambda x: model_short_lgbm.predict_proba(np.array(x).reshape(1, -1))[0, 1], axis = 1)
    proba_long = model_long_lgbm.predict_proba(testDf_lgbm[varW])[:, 1]
    proba_short = model_short_lgbm.predict_proba(testDf_lgbm[varW])[:, 1]
    testDf_lgbm['long_prob_lgbm'] = proba_long
    testDf_lgbm['short_prob_lgbm'] = proba_short

    # optimal cutoffs
    testDf_lgbm['long_cutoff_lgbm'] = cutoff_long_lgbm
    testDf_lgbm['short_cutoff_lgbm'] = cutoff_short_lgbm

    # the trading decisions
    testDf_lgbm['long_decision_lgbm']  = testDf_lgbm['long_prob_lgbm'].apply(
        lambda x: 1 if x >= cutoff_long_lgbm else 0)
    testDf_lgbm['short_decision_lgbm'] = testDf_lgbm['short_prob_lgbm'].apply(
        lambda x: 1 if x >= cutoff_short_lgbm else 0)
    testDf_lgbm['decision_lgbm'] = testDf_lgbm[['long_decision_lgbm', 'short_decision_lgbm']].apply(
        lambda x: 0 if abs(x.iloc[0]) + abs(x.iloc[1]) > 1 else (1 if x.iloc[0] == 1 else (-1 if x.iloc[1] == 1 else 0)), axis = 1)
    testDf_lgbm['net_return_lgbm'] = testDf_lgbm[[varY, 'decision_lgbm']].apply(lambda x: x.iloc[0] * x.iloc[1], axis = 1)
    
    if testAll_lgbm.shape[0] == 0:
        testAll_lgbm = testDf_lgbm.copy()
    else:
        testAll_lgbm = pd.concat([testAll_lgbm, testDf_lgbm], axis = 0)
        
    '''
        KSVM
    '''
    # predicted probability based on filters
    testDf_ksvm = testDf.copy()
    #testDf_ksvm['long_prob_ksvm'] = testDf_ksvm[varW].apply(
        #lambda x: model_long_ksvm.predict_proba(np.array(x).reshape(1, -1))[0, 1], axis = 1)
    #testDf_ksvm['short_prob_ksvm'] = testDf_ksvm[varW].apply(
        #lambda x: model_short_ksvm.predict_proba(np.array(x).reshape(1, -1))[0, 1], axis = 1)

    proba_long_ksvm = model_long_ksvm.predict_proba(testDf_ksvm[varW])[:, 1]
    proba_short_ksvm = model_short_ksvm.predict_proba(testDf_ksvm[varW])[:, 1]
    testDf_ksvm['long_prob_ksvm'] = proba_long_ksvm
    testDf_ksvm['short_prob_ksvm'] = proba_short_ksvm
    
    # optimal cutoffs
    testDf_ksvm['long_cutoff_ksvm'] = cutoff_long_ksvm
    testDf_ksvm['short_cutoff_ksvm'] = cutoff_short_ksvm

    # the trading decisions
    testDf_ksvm['long_decision_ksvm']  = testDf_ksvm['long_prob_ksvm'].apply(
        lambda x: 1 if x >= cutoff_long_ksvm else 0)
    testDf_ksvm['short_decision_ksvm'] = testDf_ksvm['short_prob_ksvm'].apply(
        lambda x: 1 if x >= cutoff_short_ksvm else 0)
    testDf_ksvm['decision_ksvm'] = testDf_ksvm[['long_decision_ksvm', 'short_decision_ksvm']].apply(
        lambda x: 0 if abs(x.iloc[0]) + abs(x.iloc[1]) > 1 else (1 if x.iloc[0] == 1 else (-1 if x.iloc[1] == 1 else 0)), axis = 1)
    testDf_ksvm['net_return_ksvm'] = testDf_ksvm[[varY, 'decision_ksvm']].apply(lambda x: x.iloc[0] * x.iloc[1], axis = 1)
    
    if testAll_ksvm.shape[0] == 0:
        testAll_ksvm = testDf_ksvm.copy()
    else:
        testAll_ksvm = pd.concat([testAll_ksvm, testDf_ksvm], axis = 0)

# aggregate of lightGBM
annual_mean_r_test_lgbm, mean_r_trade_test_lgbm, sharpe_per_trade_test_lgbm, \
    percent_of_trade_test_lgbm, win_rate_test_lgbm = \
    XYZ_Model.summarizeReturn(testAll_lgbm['net_return_lgbm'])
    
testAll_lgbm = testAll_lgbm.sort_values(by = 'date_day')
testAll_lgbm['net_return_cumu_lgbm'] = testAll_lgbm['net_return_lgbm'].cumsum()

# aggregate of KSVM
annual_mean_r_test_ksvm, mean_r_trade_test_ksvm, sharpe_per_trade_test_ksvm, \
    percent_of_trade_test_ksvm, win_rate_test_ksvm = \
    XYZ_Model.summarizeReturn(testAll_ksvm['net_return_ksvm'])

testAll_ksvm = testAll_ksvm.sort_values(by = 'date_day')
testAll_ksvm['net_return_cumu_ksvm'] = testAll_ksvm['net_return_ksvm'].cumsum()

'''
    Ensembler
'''
colsKSVM = [x for x in testAll_ksvm.columns if x not in testAll_lgbm.columns]
testAll_ensembler = testAll_lgbm.merge(testAll_ksvm[
    ['date_day', varY] + colsKSVM
    ])

testAll_ensembler['net_return_ensembler'] = testAll_ensembler[
    [varY, 'decision_lgbm', 'decision_ksvm']].apply(
    lambda x: x.iloc[0]*x.iloc[1]*1.5 if abs(x.iloc[1] + x.iloc[2]) > 1 
    else (x.iloc[0] * x.iloc[1] + x.iloc[0] * x.iloc[2] if abs(x.iloc[1]) + abs(x.iloc[2]) == 1 else 0), axis = 1)

# cumulative return
testAll_ensembler['net_return_cumu_ensembler'] = testAll_ensembler['net_return_ensembler'].cumsum()

# trading decisions
testAll_ensembler['decision_ensembler'] = testAll_ensembler[
    [varY, 'decision_lgbm', 'decision_ksvm']].apply(
    lambda x: x.iloc[1]*1.5 if abs(x.iloc[1] + x.iloc[2]) > 1
    else (x.iloc[2] if x.iloc[2] != 0 else (x.iloc[1] if x.iloc[1] != 0 else 0)), axis = 1)

# aggregate of ensembler
annual_mean_r_test_ensembler, mean_r_trade_test_ensembler, sharpe_per_trade_test_ensembler, \
    percent_of_trade_test_ensembler, win_rate_test_ensembler = \
    XYZ_Model.summarizeReturn(testAll_ensembler['net_return_ensembler'])

tradeCnt = testAll_ensembler[testAll_ensembler['decision_ensembler'] != 0].shape[0]
leverageCnt = testAll_ensembler[testAll_ensembler['decision_ensembler'] == 1.5].shape[0]

print('\nStart at %s'%startTime + ' end at %s'%endTime)
print('\nFinal results of lightGBM:')
XYZ_Model.summarizePrint(y_test_all_lgbm)

print('\nFinal results of KSVM:')
XYZ_Model.summarizePrint(y_test_all_ksvm)

print('\nFinal results of ensembler:')
XYZ_Model.summarizePrint(testAll_ensembler['net_return_ensembler'])
print('\nPercentage of leverage is %.4f'%(leverageCnt/tradeCnt))

print("直接训练最新模型用于预测")

print("\n" + "="*60)
print("重新训练最新模型用于预测")
print("="*60)

# 使用最近900天数据训练新模型
training_days = 900
recent_dates = datesList[-training_days:]

print(f"使用最近 {training_days} 天数据训练新模型:")
print(f"训练期间: {recent_dates[0]} 到 {recent_dates[-1]}")

# 准备训练数据
xyzReturnsRecent = xyzReturnsFinal[xyzReturnsFinal['date_day'].isin(recent_dates)]

# 更严格的数据清洗
print("数据清洗...")
initial_count = len(xyzReturnsRecent)

# 1. 删除目标变量为 NaN 的行
xyzReturnsRecent = xyzReturnsRecent[~xyzReturnsRecent[varY].isna()]

# 2. 删除特征变量为 NaN 的行
for feature in varW:
    nan_count = xyzReturnsRecent[feature].isna().sum()
    if nan_count > 0:
        print(f"特征 {feature} 有 {nan_count} 个 NaN 值")
    xyzReturnsRecent = xyzReturnsRecent[~xyzReturnsRecent[feature].isna()]

# 3. 删除无穷大的值
for feature in varW + [varY]:
    inf_count = np.isinf(xyzReturnsRecent[feature]).sum()
    if inf_count > 0:
        print(f"特征 {feature} 有 {inf_count} 个无穷大值")
        xyzReturnsRecent = xyzReturnsRecent[~np.isinf(xyzReturnsRecent[feature])]

final_count = len(xyzReturnsRecent)
print(f"数据清洗完成: {initial_count} -> {final_count} 条记录")
print(f"删除了 {initial_count - final_count} 条包含NaN或无穷大的记录")

if final_count == 0:
    raise ValueError("清洗后没有可用的数据！")

# 检查数据质量
print("\n数据质量检查:")
print(f"目标变量 {varY} 的范围: [{xyzReturnsRecent[varY].min():.6f}, {xyzReturnsRecent[varY].max():.6f}]")
for feature in varW:
    print(f"特征 {feature} 的范围: [{xyzReturnsRecent[feature].min():.6f}, {xyzReturnsRecent[feature].max():.6f}]")

xyzReturnsRecent = xyzReturnsRecent.sort_values(by='date_day')

# 定义 filters 和 return
W_recent = xyzReturnsRecent[varW]
Y_recent = xyzReturnsRecent[varY]

# 再次检查是否有 NaN
print(f"\n最终数据检查:")
print(f"W_recent 是否有 NaN: {W_recent.isna().any().any()}")
print(f"Y_recent 是否有 NaN: {Y_recent.isna().any()}")
print(f"W_recent 形状: {W_recent.shape}")
print(f"Y_recent 形状: {Y_recent.shape}")

# 初始化模型
XYZ_Model_New = XYZ_Meta()

print("\n训练 LightGBM 模型...")
try:
    y_pred_lgbm, latest_model_long_lgbm, latest_model_short_lgbm, latest_cutoff_long_lgbm, latest_cutoff_short_lgbm = test_model_two_sides(
        W_recent, Y_recent, 'LGBM', unbalance_ratio, XYZ_Model_New)
    print("LightGBM 模型训练成功！")
except Exception as e:
    print(f"LightGBM 训练失败: {e}")
    # 可以在这里添加更详细的调试信息
    print("调试信息:")
    print(f"W_recent 数据类型: {W_recent.dtypes}")
    print(f"Y_recent 数据类型: {Y_recent.dtype}")
    raise

print("训练 KSVM 模型...")
# 重新初始化模型参数
XYZ_Model_New = XYZ_Meta()
try:
    y_pred_ksvm, latest_model_long_ksvm, latest_model_short_ksvm, latest_cutoff_long_ksvm, latest_cutoff_short_ksvm = test_model_two_sides(
        W_recent, Y_recent, 'KSVM', unbalance_ratio, XYZ_Model_New)
    print("KSVM 模型训练成功！")
except Exception as e:
    print(f"KSVM 训练失败: {e}")
    raise

print("最新模型训练完成！")

# 更新预测函数使用最新训练的模型
def predict_signal_latest(features):
    # 检查输入特征
    if np.isnan(features).any() or np.isinf(features).any():
        print("警告: 输入特征包含 NaN 或无穷大值")
        return 0
        
    # 使用最新训练的模型
    prob_long_lgbm = latest_model_long_lgbm.predict_proba(np.array(features).reshape(1, -1))[0, 1]
    prob_short_lgbm = latest_model_short_lgbm.predict_proba(np.array(features).reshape(1, -1))[0, 1]
    prob_long_ksvm = latest_model_long_ksvm.predict_proba(np.array(features).reshape(1, -1))[0, 1]
    prob_short_ksvm = latest_model_short_ksvm.predict_proba(np.array(features).reshape(1, -1))[0, 1]

    # 决策逻辑（使用最新的cutoff值）
    long_decision_lgbm = 1 if prob_long_lgbm >= latest_cutoff_long_lgbm else 0
    short_decision_lgbm = 1 if prob_short_lgbm >= latest_cutoff_short_lgbm else 0
    long_decision_ksvm = 1 if prob_long_ksvm >= latest_cutoff_long_ksvm else 0
    short_decision_ksvm = 1 if prob_short_ksvm >= latest_cutoff_short_ksvm else 0

    # ensembler 决策逻辑
    long_votes = long_decision_lgbm + long_decision_ksvm
    short_votes = short_decision_lgbm + short_decision_ksvm

    print(f"LightGBM - 多头: {long_decision_lgbm}, 空头: {short_decision_lgbm}")
    print(f"KSVM - 多头: {long_decision_ksvm}, 空头: {short_decision_ksvm}")

    decision = 0
    if long_votes > 1 and short_votes == 0:        # 两个模型都看多
        decision = 1.5
    elif long_votes == 0 and short_votes > 1:      # 两个模型都看空
        decision = -1.5
    elif long_votes == 1 and short_votes == 0:     # 只有一个模型看多
        decision = 1
    elif long_votes == 0 and short_votes == 1:     # 只有一个模型看空
        decision = -1
    else:                                          # 一多一空 或其他情况，不交易
        decision = 0

    return decision

# 使用最新模型进行预测
print(f"\n使用最新模型进行预测:")
print(f"模型基于数据: {recent_dates[0]} 到 {recent_dates[-1]}")
features_today = [0.006287,-0.000351,-0.001485,0.001853]

# 检查预测输入
if np.isnan(features_today).any() or np.isinf(features_today).any():
    print("警告: 预测特征包含 NaN 或无穷大值")
else:
    signal_today = predict_signal_latest(features_today)
    print(f"今日交易信号: {signal_today}")