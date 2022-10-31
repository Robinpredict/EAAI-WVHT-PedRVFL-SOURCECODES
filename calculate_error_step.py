# -*- coding: utf-8 -*-
"""
A machine learning framework to forecast wave conditions (SVR MLP)

Application of neural networks and support vector
machine for significant wave height prediction (SVR MLP)

@author: lenovo
"""

# from ForecastLib import TsMetric
import numpy as np
import pandas as pd
import numpy.ma as ma
import collections
import matplotlib.pyplot as plt
from scipy.stats import rankdata,wilcoxon,friedmanchisquare
#ewtmedianp[:,k]
import scikit_posthocs as sp # https://pypi.org/project/scikit-posthocs/
#import stac
# Helper functions for performing the statistical tests
import warnings
warnings.filterwarnings("ignore")
class TsMetric(object):
    def __init__(self):
        pass
    def RMSE(self,actual, pred):
        """
        RMSE = sqrt(1/n * sum_{i=1}^{n}{pred_i - actual_i} )
        input: actual and pred should be np.array
        output: RMSE
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        RMSE = np.sqrt( 1/len(actual) *np.linalg.norm(actual - pred,2)**2)
        return RMSE

    def MAE(self,actual, pred):
        '''
        MAE = 1/n * sum_{i=1}^{n}|pred_i - actual_i} |
        input: actual and pred should be np.array
        output: MAE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAE =  1/len(actual) *np.linalg.norm(actual - pred,1)
        return MAE


    def MASE(self,actual, pred, history):
        '''
        MASE = 1/n * sum_{i=1}^{n}|pred_i - actual_i} |/ sum_traning(|diff|)
        input: actual and pred should be np.array
        output: MASE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAE =  1/len(actual) *np.linalg.norm(actual - pred,1)
        Scale =  1/(len(history)-1) * np.linalg.norm(np.diff(history),1)
        MASE = MAE/Scale

        return MASE

    def MAPE(self,actual, pred):
        '''
        MAPE = 1/n * sum_{i=1}^{n} |pred_i - actual_i} |/|actual_i|
        input: actual and pred should be np.array
        output: MAPE

        '''
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        MAPE =  1/len(actual) *np.linalg.norm((actual - pred)/actual, 1)

        return MAPE

    def sMAPE(actual, pred):
        """
        1/n  *  SUM_{i=1 to n}  { ( |pred_i-actual_i|)   /  (0.5*|pred_i|+0.5*|actual_i|)}
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred)) & (actual.shape == pred.shape )
        sMAPE = 1/len(actual) * np.sum(2*np.abs(actual - pred)/(np.abs(actual)+np.abs(pred)))
        return sMAPE

    def RAE(actual, pred, compared):
        """
        INPUT: actual, pred, a prediction to be compared with\
        :return:
            l1(pred-actual)/ l1(compared - actual)
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred) == type(compared)) & (actual.shape == pred.shape ==compared.shape)
        nom = np.linalg.norm(actual-pred,1)
        denom = np.linalg.norm(actual - compared,1)
        return nom/denom

    def RSE(actual, pred, compared):
        """
        INPUT: actual, pred, a prediction to be compared with
        :return:
            l2(pred-actual)/ l2(compared - actual)
        """
        assert (type(actual) is np.ndarray) & (type(actual) == type(pred) == type(compared)) & (actual.shape == pred.shape ==compared.shape)
        nom = np.linalg.norm(actual-pred,2) ** 2
        denom = np.linalg.norm(actual - compared,2) ** 2
        return nom/denom

    def Corr(actual, pred):
        """
        :param actual: 
        :param pred: 
        return     np.dot(actual - mean(actual), pred -mean(pred)) / norm(actual-mean(acutal)*norm(pred- mean(pred))
        """
        nom = np.dot(actual -np.mean(actual), pred-np.mean(pred))
        denom =np.linalg.norm(actual-np.mean(actual)) * np.linalg.norm(pred- np.mean(pred))
        return nom/denom
def generate_scores(method, method_args, data, labels):
    pairwise_scores = method(data.T, **method_args) # Matrix for all pairwise comaprisons
    pairwise_scores.set_axis(labels, axis='columns', inplace=True) # Label the cols
    pairwise_scores.set_axis(labels, axis='rows', inplace=True) # Label the rows, note: same label as pairwise combinations
    return pairwise_scores
def remove_minmax(x):
    xp=x.copy()
    max_=x.max()
    min_=x.min()
    # print(xp.argmax())
    xp=np.delete(xp,xp.argmax())
    xp=np.delete(xp,xp.argmin())
    return xp
def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd
def get_prediction(name):
    #file_name = D:\Newbuilding price forecasting\Corrected results
    file_name = 'D:\\Newbuilding price forecasting (Word)\\Corrected results\\'+name+'.csv'
  
    dat = pd.read_csv(file_name)
    # aa=np.round(dat.values,2)
    dat = dat.fillna(method='ffill')
    # dat.values=aa
    return dat,dat.columns
def get_data(name):
    #file_name = 'C:\\Users\\lenovo\\Desktop\\FuzzyTimeSeries\\pyFTS-master\\pyFTS\\'+name+'.csv'
    file_name = name+'.csv'
    #D:\Multivarate paper program\monthly_data
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    # if 'AEMO' not in name:
    #     dat.values=np.round(dat.values,2)
    return dat#,dat.columns
def randomnn_err(prediction,name,rmse_,mase_,mape_):
    prediction=np.around(prediction,1)
    edrvfl_dt_rmse=np.zeros(prediction.shape[1])
    edrvfl_dt_mape=np.zeros(prediction.shape[1])
    edrvfl_dt_mase=np.zeros(prediction.shape[1])
    for k in range(prediction.shape[1]):
        # print(prediction[:,k].shape)
        edrvfl_dt_rmse[k]=metric.RMSE(target, np.delete(prediction[:,k],test_idx))
        edrvfl_dt_mape[k]=metric.MAPE(target, np.delete(prediction[:,k],test_idx))
        edrvfl_dt_mase[k]=metric.MASE(target, np.delete(prediction[:,k],test_idx),history)
    rmse_[name]=edrvfl_dt_rmse.mean()
    mase_[name]=edrvfl_dt_mase.mean()
    mape_[name]=edrvfl_dt_mape.mean()
    return rmse_,mase_,mape_
def ensemble(prediction):
    nl=5
    for k in range(10):
   
        colidx=[k+l*10 for l in range(nl) ]
        layer_l_pre=prediction[:,colidx]
        for j in range(nl):
            p=np.delete( layer_l_pre[:,j],test_idx)
            
    target
# loc='SA'
# year='2020'
# month='PRICE_AND_DEMAND_202010_'+loc+str(1)#'PRICE_AND_DEMAND_202001_QLD1'
# dataset='D:\\AEMO\\'+loc+'\\'+ year +'\\'+month

rmse_all=[]
mape_all=[]
mase_all=[]
step=2
fs=16
# best_l={'201946083h': 2, '201946080h': 9, '201946076h': 4, '201946001h': 6, '201846083h': 8, '201846080h': 2, '201846076h': 9, '201846001h': 4, '201746083h': 2, '201746080h': 5, '201746076h': 7, '201746001h': 6}
for year in ['2019','2018','2017']:
    axi=0
    # fig, axes = plt.subplots(4,1,figsize=(9,9))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = fs
    for station in  ['46083h','46080h','46076h','46001h']:
        features=[ 'WDIR', 'WSPD', 'GST','APD','WVHT']
        data=pd.read_csv('Ocean energy/'+station+year+'.txt',delim_whitespace=True)
        data=data[features]
        var_name=data.columns
        data=data.where(data!='99.0',np.nan)
        data=data.where(data!='99.00',np.nan)
        data1=pd.read_csv('Ocean energy/'+station+year+'.txt',delim_whitespace=True)
        data1=data1[features]
        data1=data1.where(data1!='99.0',np.nan)
        data1=data1.where(data1!='99.00',np.nan)
        print(station+year)
        data1=data1.fillna(method='ffill')
        # do not compute error on Nan index
        while data1.isnull().values.any():
            data1=data1.fillna(method='ffill')
        # print(data1.isnull().values.any())
        # print(data.isnull().values.any())
        # print(data.values[:])
        data_=data['WVHT'].values[1:].astype(np.float).reshape(-1,1)
        idx=data_!=np.nan
        validation_l,test_l=int(0.2*data_.shape[0]),int(0.2*data_.shape[0])
        # test_idx=idx[-test_l:].ravel()
        np_data=data_#df_data.loc[df_data['MM']==month]['WSPD'].values.astype(float).reshape(-1,1)
       
        train_l=len(np_data)-test_l-validation_l
        test_idx=np.argwhere(np.isnan(np_data[-test_l:].ravel()))
        target=np.delete(np_data[-test_l:].ravel(),test_idx)
        np_datan=data1['WVHT'].values[1:].astype(np.float).reshape(-1,1)
        Naive=np.delete(np_datan[-test_l-step:-step].ravel(),test_idx)
        # print(np.isnan(np.min(np_data[-test_l:])))
        # print(np.isnan(np.min(target)))
        pad_data=data1['WVHT'].values[1:].astype(np.float).reshape(-1,1)
        history=pad_data[:-test_l].ravel()
        prediction={}
        pre_loc='Ocean energy/Results/'
        rmse={}
        mape={}
        mase={}
        # prediction['Persistence']=np_data[-test_l-1:-1].ravel()
        models=['GSVR','MLPrelu','RPSOELM','LSTM']
        
        for model in models[:]:
            #ARIMA'+station+year+'step'+str(step)+'.csv'
            prediction[model]=np.delete(get_data(pre_loc+model+station+year+'step'+str(step)).values[:,1:].ravel(),test_idx)
        
        metric=TsMetric()
        prediction['RF']=np.delete(pd.read_csv(pre_loc+'RFBOAWVHT_multi_var'+station+year+'500'+'step'+str(step)).values[:,1:].ravel(),test_idx)
        models.append('RF')
        prediction['ARIMA']=np.delete(pd.read_csv(pre_loc+'R_test_ARIMA'+station+year+str(step)+'.csv').values[:,:].ravel(),test_idx)
        models.append('ARIMA')
        
        # prediction['ETS']=np.delete(pd.read_csv(pre_loc+'ETS'+station+year+'step'+str(step)+'.csv').values[:,1:].ravel(),test_idx)
        # models.append('ETS')
        prediction['ETS']=np.delete(pd.read_csv(pre_loc+'R_test_ETS'+station+year+str(step)+'.csv').values.ravel(),test_idx)
        models.append('ETS')
        
        prediction['ARFIMA']=np.delete(pd.read_csv(pre_loc+'R_test_ARFIMA'+station+year+str(step)+'.csv').values[-test_l:].ravel(),test_idx)
        models.append('ARFIMA')
        # prediction['Naive']=Naive.ravel()
        # models.append('Naive')
        # rf_loc=pre_loc+'RFBOAWVHT_multi_var'+station+year+'300'
        # rf_pres = pd.read_csv(rf_loc).values[:,1:]
        # prediction['RF']=rf_pres
        # models.append('RF')
        
        # elm_loc=pre_loc+'EnELMBOAWVHT_multi_var'+station+year+'300'
        # elm_pres = pd.read_csv(elm_loc).values[:,1:]
        # prediction['EELM']=elm_pres
        # models.append('EELM')
        # #MAPE
        # #MASE
        
        vals=pd.read_csv(pre_loc+'Nbeats'+station+year+'step'+str(step)+'.csv').values[:,1:]
        prediction['Nbeats']=np.delete(np.median(vals,axis=1),test_idx)
        models.append('Nbeats')
        
        rmse=dict(zip(models,[metric.RMSE(target,prediction[i]) for i in models]))
        mape=dict(zip(models,[metric.MAPE(target,prediction[i]) for i in models]))
        mase=dict(zip(models,[metric.MASE(target,prediction[i],history) for i in models]))
        error = collections.OrderedDict()
        
        # 'EnELMBOAWVHT_multi_var'
        eelm_loc=pre_loc+'EELMBOAWVHT_multi_var'+station+year+'500'+'step'+str(step)#+'.csv'
        # DF=pd.read_csv(eelm_loc)
        # DF.to_csv(eelm_loc+'.csv',index=False)
        eelm_pres = pd.read_csv(eelm_loc).values[:,1:]#[test_idx]
        rmse,mase,mape=randomnn_err(eelm_pres,'EELM',rmse,mase,mape)
        
        
        
        edrvfl_loc=pre_loc+'TgridedRVFLBOAWVHT_multi_var1'+station+year+'100'+'step'+str(step)+'.csv'
        # DF=pd.read_csv(edrvfl_loc)
        # DF.to_csv(edrvfl_loc+'.csv',index=False)
        edrvfl_pres = pd.read_csv(edrvfl_loc).values[:,1:]#[test_idx]
        rmse,mase,mape=randomnn_err(edrvfl_pres,'RVFL',rmse,mase,mape)
        # rmse,mase,mape=randomnn_err(edrvfl_pres,'RVFL',rmse,mase,mape)
        #edRVFLBOAWVHT_multi_var_FS_DT_Enh46001h201760
        # edrvfl_loc=pre_loc+'edRVFLBOAWVHT_multi_var_FS_LR_Enh'+station+year+'100'
        # edrvfl_pres = pd.read_csv(edrvfl_loc).values[:,1:]
        # rmse,mase,mape=randomnn_err(edrvfl_pres,'edRVFLFSLR',rmse,mase,mape)
        
        # edrvfl_loc=pre_loc+'edRVFLBOAWVHT_multi_var_FS_DT_Enh'+station+year+'100'
        # edrvfl_pres = pd.read_csv(edrvfl_loc).values[:,1:]
        # rmse,mase,mape=randomnn_err(edrvfl_pres,'edRVFLFSDT',rmse,mase,mape)
        
#        edrvfl_loc=pre_loc+'edRVFLBOAWVHT_multi_var_FS_DT_all_poolhidden5'+station+year+'100'+'step'+str(step)
#        edrvfl_pres = pd.read_csv(edrvfl_loc).values[:,1:]
#        rmse,mase,mape=randomnn_err(edrvfl_pres,'edRVFLFSDTPool',rmse,mase,mape)
        #edRVFLBOALR_pool
        #edRVFLBOALR_pool_alllayer
        # edrvfl_loc=pre_loc+'edRVFLBOALR_pool_alllayer8'+station+year+'100'+'step'+str(step)
        # allp=pd.read_csv(edrvfl_loc).values[:,1:-30]
        # edrvfl_loc=pre_loc+'edRVFLBOALR_pool8'+station+year+'100'+'step'+str(step)
        # edrvfl_loc=pre_loc+'edRVFLBOAWVHT_multi_var'+station+year+'100'+'step'+str(step)
        edrvfl_loc=pre_loc+'gridedRVFLBOAWVHT_multi_var'+station+year+'100'+'step'+str(step)+'.csv'
        # DF=pd.read_csv(edrvfl_loc)
        # DF.to_csv(edrvfl_loc+'.csv',index=False)
        edrvfl_pres = pd.read_csv(edrvfl_loc).values[:,1:]
        rmse,mase,mape=randomnn_err(edrvfl_pres,'edRVFL',rmse,mase,mape)


 
        edrvfl_loc=pre_loc+'edRVFLFSLRPool5'+station+year+'100'+'step'+str(step)+'.csv'
        # DF=pd.read_csv(edrvfl_loc)
        # DF.to_csv(edrvfl_loc+'.csv',index=False)
        edrvfl_pres = pd.read_csv(edrvfl_loc).values[:,1:].T
        ps0=edrvfl_pres
        rmse,mase,mape=randomnn_err(edrvfl_pres,'edRVFLFSLRPool',rmse,mase,mape)
        
        
        
        #Hybrid_'+stats_m+'edRVFL'+str(Nls[0])+station+year+str(boat)+'step'+str(step)
        edrvfl_loc=pre_loc+'R_Hybrid_AddARIMAedRVFL5'+station+year+'100'+'step'+str(step)+'.csv'
        # DF=pd.read_csv(edrvfl_loc)
        # DF.to_csv(edrvfl_loc+'.csv',index=False)
        ps1 = pd.read_csv(edrvfl_loc).values[:,1:].T
  
        rmse,mase,mape=randomnn_err(ps1,'AARIMAedRVFL',rmse,mase,mape)
        
        edrvfl_loc=pre_loc+'R_Hybrid_MulARIMAedRVFL5'+station+year+'100'+'step'+str(step)+'.csv'
        # DF=pd.read_csv(edrvfl_loc)
        # DF.to_csv(edrvfl_loc+'.csv',index=False)

        ps2 = pd.read_csv(edrvfl_loc).values[:,1:].T      
        rmse,mase,mape=randomnn_err(ps2,'MARIMAedRVFL',rmse,mase,mape)
        
        # edrvfl_loc=pre_loc+'R_test_raw_Hybrid_AddETSedRVFL5'+station+year+'100'+'step'+str(step)
        # ps3 = pd.read_csv(edrvfl_loc).values[:,1:].T       
        # rmse,mase,mape=randomnn_err(ps3,'AETSedRVFL',rmse,mase,mape)
        
        # edrvfl_loc=pre_loc+'R_test_raw_Hybrid_mULETSedRVFL5'+station+year+'100'+'step'+str(step)
        # ps4 = pd.read_csv(edrvfl_loc).values[:,1:].T       
        # rmse,mase,mape=randomnn_err(ps4,'METSedRVFL',rmse,mase,mape)
        
        rmse,mase,mape=randomnn_err((ps1+ps2+ps0)/3,'Ensemble',rmse,mase,mape)
        # fig, ax = plt.subplots(figsize=(10,7))
       
        # ax.grid(color='white')        
        # ax.plot(ps.mean(axis=1),label='Forecasts',linestyle='dotted',marker="o",markersize=3.5,markevery=9)        
   
        
        # ax.plot(target,label='Raw data',linestyle='dotted',marker=">",markersize=3.5,markevery=5)
        
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.legend(framealpha=0,fontsize=fs)
        # ax.set_facecolor('lightgrey')
        # prediction['RVFL']=edrvfl_dtpres.mean(axis=1)
        # edrvfl_dt_rmse=np.zeros(edrvfl_dtpres.shape[1])
        # edrvfl_dt_mape=np.zeros(edrvfl_dtpres.shape[1])
        # edrvfl_dt_mase=np.zeros(edrvfl_dtpres.shape[1])
        # for k in range(edrvfl_dtpres.shape[1]):
        #     edrvfl_dt_rmse[k]=metric.RMSE(target, edrvfl_dtpres[:,k])
        #     edrvfl_dt_mape[k]=metric.MAPE(target, edrvfl_dtpres[:,k])
        #     edrvfl_dt_mase[k]=metric.MASE(target, edrvfl_dtpres[:,k],history)
        # rmse['RVFLBOA']=edrvfl_dt_rmse.mean()
        # mase['RVFLBOA']=edrvfl_dt_mase.mean()
        # mape['RVFLBOA']=edrvfl_dt_mape.mean()
        
        error['RMSE']=rmse
        error['MAPE']=mape
        error['MASE']=mase
        
       
        
        
        
        error_df=pd.DataFrame.from_dict(error,orient='index')
        # e_df=error_df.reindex(['RMSE','MAPE','MASE'])
        rmse_all.append(error_df.loc['RMSE'].values.reshape(1,-1))
        mape_all.append(error_df.loc['MAPE'].values.reshape(1,-1))
        mase_all.append(error_df.loc['MASE'].values.reshape(1,-1))
        # fig, ax = plt.subplots(4,1,figsize=(10,7))
        # fig, ax = plt.subplots(figsize=(10,7))
       
        # ax.grid(color='white')        
        # ax.plot(edrvfl_dtpres.mean(axis=1),label='Forecasts',linestyle='dotted',marker="o",markersize=3.5,markevery=9)        
   
        
        # ax.plot(target,label='Raw data',linestyle='dotted',marker=">",markersize=3.5,markevery=5)
        
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.legend(framealpha=0,fontsize=7)
        # ax.set_facecolor('lightgrey')
        # fig.savefig(pre_loc+'EWTedRVFLBOA50'+year+station+'step'+str(step)+'.eps', dpi=1000,format='eps')
        # axi+=1
        # print(e_df)
    # fig.show()
    # fig.savefig(pre_loc+'EWTedRVFLBOA50'+loc+'.eps', dpi=1000,format='eps')
rmse_all_np=np.concatenate(rmse_all,axis=0)
mase_all_np=np.concatenate(mase_all,axis=0)
mape_all_np=np.concatenate(mape_all,axis=0)
rmsealldf=pd.DataFrame(data=rmse_all_np,columns=error_df.columns)

masealldf=pd.DataFrame(data=mase_all_np,columns=error_df.columns)
mapealldf=pd.DataFrame(data=mape_all_np,columns=error_df.columns)
ranks=np.zeros(rmse_all_np.shape)
av=[]
for err in [rmse_all_np,mase_all_np,mape_all_np]:
    for i in range(err.shape[0]):
        #     print(err.values[:,1:][i,:])
        ranks[i,:]=rankdata(err[i,:])
    # af1=friedmanchisquare(err[:,0],err[:,1],err[:,2],err[:,3],err[:,4],err[:,5],
    #                   err[:,6],err[:,7],err[:,8],err[:,9],err[:,10],err[:,11],
    #                   err[:,12],err[:,13],err[:,14])
    af2=friedmanchisquare(*err)
    print(af2)
    avranks=np.mean(ranks,axis=0).reshape(1,-1)
    cd=compute_CD(avranks.ravel(),err.shape[0])
    av.append(avranks)
    # print(avranks)
avrank=np.concatenate(av,axis=0)
avrank_df=pd.DataFrame(data=avrank,columns=error_df.columns)

#cd 4.796
# rank(*err)
rmse_nemenyi_scores = generate_scores(sp.posthoc_wilcoxon, {}, rmsealldf.values, avrank_df.columns)
mase_nemenyi_scores = generate_scores(sp.posthoc_wilcoxon, {}, masealldf.values, avrank_df.columns)
mape_nemenyi_scores = generate_scores(sp.posthoc_wilcoxon, {}, mapealldf.values, avrank_df.columns)
# edrvfl_loc=pre_loc+'edRVFLBOAWVHT_multi_var_FS_DT_Enhall10'+station+year+'100'
# edrvfl_pres = pd.read_csv(edrvfl_loc).values[:,1:]
col=['Year', 'Station', 'ARIMA','ARFIMA','ETS','GSVR', 'MLPrelu', 'LSTM', 'RPSOELM', 'RF',
        'EELM', 'Nbeats','RVFL', 'edRVFL', 'edRVFLFSLRPool','MARIMAedRVFL','AARIMAedRVFL','Ensemble']
avrank_df.insert(0,'Error',['RMSE','MASE','MAPE'])
avrank_df.set_index('Error')
#
print(avrank_df[col[2:]])
# avrank_df[col[2:]].to_csv('Ocean energy/Errors/Rank_step'+str(step)+'.csv')
y=np.array([2019,2019,2019,2019,2018,2018,2018,2018,2017,2017,2017,2017])
s=np.array([46083,46080,46076,46001,46083,46080,46076,46001,46083,46080,46076,46001])
rmsealldf.insert(0,'Year',y)
rmsealldf.insert(1,'Station',s)
rmsealldf.insert(2,'Metric',['RMSE' for i in range(12)])
# rmsealldf['Station']=np.array([46083,46080,46076,46001,46083,46080,46076,46001,46083,46080,46076,46001])
# ['Year', 'Station', 'test3SVR', 'MLPrelu', 'LSTM', 'PSOELM', 'RF',
#        'Naive', 'EELM', 'RVFL', 'edRVFLFSLRPool', 'edRVFL']

# rmsealldf[col].to_csv('Ocean energy/Errors/RMSE_step'+str(step)+'.csv')
masealldf.insert(0,'Year',y)
masealldf.insert(1,'Station',s)
masealldf.insert(2,'Metric',['MASE' for i in range(12)])
masealldf['Year']=np.array([2019,2019,2019,2019,2018,2018,2018,2018,2017,2017,2017,2017])
masealldf['Station']=np.array([46083,46080,46076,46001,46083,46080,46076,46001,46083,46080,46076,46001])
# masealldf[col].to_csv('Ocean energy/Errors/MASE_step'+str(step)+'.csv')
mapealldf.insert(0,'Year',y)
mapealldf.insert(1,'Station',s)
mapealldf.insert(2,'Metric',['MAPE' for i in range(12)])
mapealldf['Year']=np.array([2019,2019,2019,2019,2018,2018,2018,2018,2017,2017,2017,2017])
mapealldf['Station']=np.array([46083,46080,46076,46001,46083,46080,46076,46001,46083,46080,46076,46001])
# mapealldf[col].to_csv('Ocean energy/Errors/MAPE_step'+str(step)+'.csv')
#for i in [rmsealldf,masealldf,mapealldf]:
#    i.insert(2,'Metric',[])
values=[]
col=['Year', 'Station','Metric', 'ARIMA','ARFIMA','ETS','GSVR', 'MLPrelu', 'LSTM', 'RPSOELM', 'RF',
        'EELM', 'Nbeats','RVFL', 'edRVFL', 'edRVFLFSLRPool','MARIMAedRVFL','AARIMAedRVFL','Ensemble']
for i in range(12):
    for e in [rmsealldf,masealldf,mapealldf]:
        values.append(e[col].values[i:i+1,:])
ev=np.concatenate(values, axis=0)

errdf=pd.DataFrame(data=ev,columns=col)
# errdf.to_csv('Ocean energy/Errors/Error_step'+str(step)+'.csv')
# mase_nemenyi_scores[col[3:]].to_csv('Ocean energy/Errors/Wilc_mase_step'+str(step)+'.csv')