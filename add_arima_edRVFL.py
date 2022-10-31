'''
apply feature selection to enhancement and direct link

USE RAW WVHT TO predict arima residual

additive arima edrvfl
same hyper multiple evaluations
----
'''
from sklearn.tree import DecisionTreeRegressor
from itertools import product
import matplotlib.pyplot as plt
# import ForecastLib
from sklearn import preprocessing
import numpy as np
import random
import pandas as pd
from hyperopt import fmin, tpe, hp
# import DeepRVFL_
from DeepRVFL_.DeepRVFL import DeepRVFL
# import DeepRVFL
from utils import MSE, config_MG, load_MG
import pickle
# from skfeature.function.similarity_based import reliefF,lap_score
# from tsmoothie.smoother import ExponentialSmoother
from sklearn.linear_model import LinearRegression
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
def remove_outlier(df,col,val):
    df[df[col]==val]
def feature_importance(x,y,method='DT'):
    scaler=preprocessing.MinMaxScaler()
    x=scaler.fit_transform(x)
    if method=='DT':
        model=DecisionTreeRegressor(criterion='mse')
        model.fit(x,y)
        importance=model.feature_importances_
    elif method=='LR':
        model=LinearRegression()
        # print('fs',x.shape,y.shape)
        model.fit(x,y.ravel())
        importance=abs(model.coef_)
    return importance
def feature_ranking(score):
    """
    Rank features in descending order according to reliefF score, the higher the reliefF score, the more important the
    feature is
    """
    idx = np.argsort(score, 0)
    return idx[::-1]
def format_data(dat,order,target):
    n_sample=dat.shape[0]-order-step+1
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i]  =target[i+order+step-1]
    return x.T,y.T
def select_indexes(data, indexes):

    # if len(data) == 1:
    return data[:,indexes]
    
    # return [data[i] for i in indexes]
def compute_error(actuals,predictions,history=None):
    actuals=actuals.ravel()
    predictions=predictions.ravel()
    
    metric=TsMetric()
    error={}
    error['RMSE']=metric.RMSE(actuals, predictions)
    # error['MAPE']=metric.MAPE(actuals,predictions)
    error['MAE']=metric.MAE(actuals,predictions)
    if history is not None:
        history=history.ravel()
        error['MASE']=metric.MASE(actuals,predictions,history)
        
    
    return error
def get_data(name):
    #file_name = 'C:\\Users\\lenovo\\Desktop\\FuzzyTimeSeries\\pyFTS-master\\pyFTS\\'+name+'.csv'
    file_name = name+'.csv'
    #D:\Multivarate paper program\monthly_data
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    return dat,dat.columns
class Struct(object): pass

# sistemare indici per IP in config_pianomidi, mettere da un'altra parte
# sistema selezione indici con transiente messi all'interno della rete

def config_load(iss,IP_indexes):

    configs = Struct()
    
    
    configs.iss = iss # set insput scale 0.1 for all recurrent layers
    
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0 # activate pre-train
    

#    configs.IPconf.Nepochs=10
    configs.enhConf = Struct()
    configs.enhConf.connectivity = 1 # connectivity of recurrent matrix
    
    configs.readout = Struct()
    configs.readout.trainMethod = 'Ridge' # train with singular value decomposition (more accurate)
    # configs.readout.regularizations = 10.0**np.array(range(-16,-1,1))
    
    return configs 
def dRVFL_predict(hyper,data,train_idx,test_idx,layer,s,last_states=None):
   
    np.random.seed(s)
    Nu=data.inputs.shape[0]

    Nh = hyper[0][0] 
    # print(hyper[0][3])
    # ratio=hyper[0][3]
    Nl = layer # 
    
    reg=[]

    iss=[]
    ratios=[]
    for h in hyper:
        reg.append( h[1])        
        iss.append(h[2])
        ratios.append(h[-1])
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs,act=act)
    train_targets = select_indexes(data.targets, train_idx)
    
    
    # feature_score=reliefF.reliefF(data.inputs.T[:len(train_idx),:],train_targets.ravel())          
    # ranks=feature_ranking(feature_score)
    # idx1=ranks[:int(ratio*Nh)]
    
    if Nl==1:
        
        states = deepRVFL.computeLayerState(0,data.inputs)
        new_states=states
    else:
        
        # print('max',max(importance),'sel',importance[idx[0]])
        states=deepRVFL.computeLayerState(Nl-1,data.inputs,last_states[:,:])
    # print(states.shape,data.inputs.shape)
    features=np.concatenate([states,data.inputs],axis=0)
    # features=states
    # print('train',states.shape,data.inputs.shape,features.shape,train_targets.shape)
    importance=feature_importance(features[:,:len(train_idx)].T, train_targets.T,method=FS_method)
    idx=feature_ranking(importance)[:int(ratios[-1]*Nh)]  
    # direct_idx=[]
    # for i in idx:
    #     if i>states.shape[0]-1:
    #         direct_idx.append(i-states.shape[0])
    # direct_idx=np.array(direct_idx)  
    # # print(idx,direct_idx.shape,data.inputs.shape,direct_idx)
    # if len(direct_idx)>0:
    #     new_input=data.inputs[direct_idx,:]
    #     data.inputs=new_input
    # else:
    #     new_input=data.inputs
    new_input=data.inputs
    hidden_idx=[]
   
    for i in idx:
        if i<states.shape[0]:
            hidden_idx.append(i)
    hidden_idx=np.array(hidden_idx)  
    # print(idx,direct_idx.shape,data.inputs.shape,direct_idx)
    if len(hidden_idx)>0:
        new_states=states[hidden_idx,:]
        states=new_states
    else:
        states=states[np.random.randint(states.shape[0], size=1),:] 
    
    features=features[idx,:]
    train_states = select_indexes(features, train_idx)#(Nh,n_sample)
    # print(train_states.shape,train_targets.shape)
    # importance=feature_importance(train_states.T, train_targets.T,method='DT')
    # idx=feature_ranking(importance)[:int(ratio*Nh)]
    test_states = select_indexes(features, test_idx)
    # feature_score=reliefF.reliefF(train_states.T,train_targets.ravel())          
    # ranks=feature_ranking(feature_score)
    # idx=ranks[:int(ratio*Nh)]
    deepRVFL.trainReadout(train_states[:,:], train_targets, reg[-1])
 
    test_outputs_norm = deepRVFL.computeOutput(test_states[:,:]).T

    return test_outputs_norm,states[:,:],new_input
def edRVFL_predict(hyper,data,train_idx,test_idx,s):
    #idxs:list(train_idx) 
#    Nrs,Nls,regs,transients,spectral_radiuss,leaky_rates,input_scale
    np.random.seed(s)
    Nu=data.inputs.shape[0]
    Nr = int(hyper[0][0]) # number of recurrent units
    # ratio=hyper[0][3]
    Nl = len(hyper) # number of recurrent layers
    reg=[]
   
    iss=[]
    for h in hyper:
        reg.append( h[1])
       
        iss.append(h[2])
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nr, Nl, configs,act=act)
    last_states=None
    outputs=np.zeros((len(test_idx),Nl))
    insample=np.zeros((len(train_idx),Nl))
    train_targets = select_indexes(data.targets, train_idx)
    num_feas=np.zeros((Nl,4))
    for l in range(Nl):
        if l==0:
            # feature_score=reliefF.reliefF(data.inputs.T[:len(train_idx),:],train_targets.ravel())          
            # ranks=feature_ranking(feature_score)
            # idx1=ranks[:int(ratio*Nr)]
            states = deepRVFL.computeLayerState(l,data.inputs,inistate=None)
            # states = deepRVFL.computeGlobalState(data.inputs)
        else:
            
            # print(l,ratio)
            # importance=feature_importance(last_states[:,:len(train_idx)].T, train_targets.T,method='DT')
           
            # idx=feature_ranking(importance)[:int(ratio*Nr)]
            states=deepRVFL.computeLayerState(l,data.inputs,last_states)
        ratio=hyper[l][-1]
        features=np.concatenate([states,data.inputs],axis=0)
        # features=states
        importance=feature_importance(features[:,:len(train_idx)].T, train_targets.T,method=FS_method)
        idx=feature_ranking(importance)[:int(ratio*Nr)]      
        features=features[idx,:]
        # direct_idx=[]
        # for i in idx:
        #     if i>states.shape[0]-1:
        #         direct_idx.append(i-states.shape[0])
        # direct_idx=np.array(direct_idx)  
        # # print(idx,direct_idx.shape,data.inputs.shape,direct_idx)
        # if len(direct_idx)>0:
        #     new_input=data.inputs[direct_idx,:]
        #     data.inputs=new_input
        # else:
        #     new_input=data.inputs
        new_input=data.inputs
        hidden_idx=[]
       
        for i in idx:
            if i<states.shape[0]:
                hidden_idx.append(i)
        hidden_idx=np.array(hidden_idx) 
        num_feas[l,0]=len(hidden_idx)
        num_feas[l,1]=len(idx)-len(hidden_idx)
        num_feas[l,2]=len(hidden_idx)/len(idx)
        num_feas[l,3]=1-len(hidden_idx)/len(idx)
        # print(idx,direct_idx.shape,data.inputs.shape,direct_idx)
        if len(hidden_idx)>0:
            new_states=states[hidden_idx,:]
            states=new_states
        else:
            states=states[np.random.randint(states.shape[0], size=1),:]  
        last_states=states
        train_states = select_indexes(features, train_idx)
        train_targets = select_indexes(data.targets, train_idx)
        # importance=feature_importance(train_states.T, train_targets.T,method='DT')
        # idx=feature_ranking(importance)[:int(ratio*Nr)]
        # feature_score=reliefF.reliefF(train_states.T,train_targets.ravel())          
        # ranks=feature_ranking(feature_score)
        # idx=ranks[:int(ratio*Nr)]
        # print(l,Nu,Nr)
        
        test_states = select_indexes(features, test_idx)
        # print(train_states.shape,Nr)
        deepRVFL.trainReadout(train_states, train_targets, reg[l])
        test_outputs_norm = deepRVFL.computeOutput(test_states).T
        outputs[:,l:l+1]=test_outputs_norm
        
        train_outputs_norm = deepRVFL.computeOutput(train_states).T
        insample[:,l:l+1]=train_outputs_norm
    # print(train_states.shape,train_targets.shape)
    # # plt.figure()
    # print(train_targets.shape,np.mean(insample,axis=1).reshape(-1,1).shape)
    # plt.plot(np.mean(insample,axis=1).ravel()[:500],label='insample norm')
    # plt.plot(train_targets[0,:500])
    # plt.legend()
    # plt.show()
    # name='Ocean energy/Results/R_Hybrid_Add'+stats_m+'NUMFEASedRVFLFSLR'+str(Nls[0])+station+year+'step'+str(step)+'s'+str(s)+'.pkl'
    # f=open(name,'wb')
    # pickle.dump(deepRVFL,f)
    
    # f=open('DRVFL.pkl','rb')
    # pickle.load(f)
    # f.close()
    return np.mean(outputs,axis=1).reshape(-1,1),np.mean(insample,axis=1).reshape(-1,1),num_feas#outputs.mean(axis=1).reshape(-1,1)
def RVFL_predict(hyper,data,train_idx,test_idx,s):
    #idxs:list(train_idx) 
#    Nrs,Nls,regs,transients,spectral_radiuss,leaky_rates,input_scale
    np.random.seed(s)
    Nu=1
    Nh = hyper[0] # number of recurrent units
    Nl = 1#hyper[1] # number of recurrent layers
    reg = hyper[1]
    iss=hyper[2]
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs)
    states = deepRVFL.computeState(data.inputs, deepRVFL.IPconf.DeepIP)  
#    print(transient,states[0].shape,states[0][0,:2],states[0][0,-2:])              
    train_states = select_indexes(states, train_idx)
    train_targets = select_indexes(data.targets, train_idx)
    test_states = select_indexes(states, test_idx)
    outputs=np.zeros((len(test_idx),Nl))
    for i in range(Nl):              
        deepRVFL.trainReadout(train_states[i*Nh:i*Nh+Nh,:], train_targets, reg)
        test_outputs_norm = deepRVFL.computeOutput(test_states[i*Nh:i*Nh+Nh,:]).T
        outputs[:,i:i+1]=test_outputs_norm
#    test_outputs=scaler.inverse_transform(test_outputs_norm)
#    actuals=data_[-len(test_idx):]
#    test_err=compute_error(actuals,test_outputs,None)
    return np.mean(outputs,axis=1).reshape(-1,1)#outputs.mean(axis=1).reshape(-1,1)
def cross_validation(hypers,data,raw_data,train_idx,val_idx,Nl,regs,input_scale,ratios=[1],scaler=None,s=0,boat=50):
    best_hypers=[]
    np.random.seed(s)
    layer_s=None
    for i in range(Nl):
        # print(i,layer_s)
        layer=i+1
        layer_h,layer_s=layer_cross_validation(hypers,data,raw_data,train_idx,val_idx,layer,
                           scaler=scaler,s=s,last_states=layer_s,best_hypers=best_hypers.copy(),boat=boat)
        # print(layer_h)
        Nhs=[layer_h[0]]
        # ratios=[layer_h[-1]]
#        print(transients,layer_h)
        if layer==1:
            hypers=list(product(Nhs,regs,input_scale,ratios))        
        best_hypers.append(layer_h)
        # print(best_hypers)
    return best_hypers
def layer_cross_validation(hypers,data,raw_data,train_idx,val_idx,layer,
                           scaler=None,s=0,last_states=None,best_hypers=None,boat=50):
    cvloss=[]
    np.random.seed(s)
    states=[]
    space={'layer':hp.choice('layer', [layer]),
           'data':hp.choice('data', [data]),
           'raw_data':hp.choice('raw_data', [raw_data]),
           'last_states':hp.choice('last_states', [last_states]),
           'scaler':hp.choice('scaler', [scaler]),
           's':hp.choice('s', [s]),
           'val_idx':hp.choice('val_idx', [val_idx]),
           'train_idx':hp.choice('train_idx', [train_idx]),
           'best_hypers':hp.choice('best_hypers', [best_hypers]),
            'input_scale':hp.uniform('input_scale', 0.9,1.0000001),
            'ratio':hp.uniform('ratio', 0.5, 1),
            'regs':hp.uniform('regs', 0, 1)}
    if layer==1:
        space['Nhs']=hp.randint('Nhs', 10, 200)
    else:
        best_hidden=[best_hypers[0][0]]
        # space['Nhs']=hp.randint('Nhs', 100, 200)
        # space['Nhs']=hp.randint('Nhs', best_hypers[0][0]-5, best_hypers[0][0]+5)
        space['Nhs']=hp.choice('Nhs', [best_hypers[0][0]])
    args=fmin(fn=layer_obj,
                space=space,
                max_evals=boat,
                rstate=np.random.default_rng(0),
                 # rstate=np.random.RandomState(0),
                verbose=False,
                algo=tpe.suggest)
    #fmin(f_lgbm, lgbm_param, algo=tpe.suggest, max_evals=MAX_EVAL, trials=trials, rstate=np.random.default_rng(SEED))
    if layer==1:
        best_hyper=[args['Nhs'],args['regs'],args['input_scale'],args['ratio']]
    else:
        # best_hyper=[args['Nhs'],args['regs'],args['input_scale'],args['ratio']]
        best_hyper=[best_hidden[0],args['regs'],args['input_scale'],args['ratio']]
    # print(layer,best_hyper)
    if layer>1:
#            print('a',layer,best_hypers)
            hyper_=best_hypers.copy()#
            hyper_.append(best_hyper)
#            print('aa',hyper_,best_hypers)
    else:
#            print(layer,best_hypers)
        hyper_=[best_hyper]
    _,best_state,_=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                         s,last_states=last_states)
   
    # best_state=states[cvloss.index(min(cvloss))]
    return best_hyper,best_state
def layer_obj(args):
    layer=args['layer']
    best_hypers=args['best_hypers']
    # print('layer',best_hypers)
    #Nhs,regs,input_scale,ratios
    hyper=[args['Nhs'],args['regs'],args['input_scale'],args['ratio']]
    data=args['data']
    train_idx,val_idx=args['train_idx'],args['val_idx']
    scaler=args['scaler']
    s=args['s']
    raw_data,last_states=args['raw_data'],args['last_states']
    if layer>1:
#            print('a',layer,best_hypers)
            # hyper_=best_hypers.copy()#
            # hyper_.append(hyper)
            hyper_=[i for i in best_hypers]#.append(hyper)
            hyper_.append(hyper)
            # hyper_=[best_hypers[0],hyper]
            # print(hyper_)
#            print('aa',hyper_,best_hypers)
    else:
        hyper_=[hyper]
    # print('bh',best_hypers)
    # print('layer',layer,hyper_)
    e=0
    for s in range(seeds):
        test_outputs_norm,_,new_input=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                         s,last_states=last_states)
        test_outputs=scaler.inverse_transform(test_outputs_norm)
        # test_outputs=scaler.inverse_transform(0.5*(test_outputs_norm+1))
        actuals=raw_data[-len(val_idx):]
        # print(actuals.shape,test_outputs.shape)
        test_err=compute_error(actuals,test_outputs,None)['RMSE']
        e+=test_err
    return e
def ed_cross_validation(hypers,data,raw_data,train_idx,val_idx,Nl,scaler=None,s=0):
    # if len(hypers)!=Nl:
    #     print('Error')
    #     return None
    # cvloss=[]
    # for i in range(1,Nl):
    #     edhyper=hypers[:i+1]
    #     test_outputs_norm=edRVFL_predict(edhyper,data,train_idx,val_idx,s)
        
    #     test_outputs=scaler.inverse_transform(test_outputs_norm)
    #     actuals=raw_data[-len(val_idx):]
    #     test_err=compute_error(actuals,test_outputs,None)
    #     cvloss.append(test_err['RMSE'])
#    print(cvloss,cvloss.index(min(cvloss)))
    return hypers#[:cvloss.index(min(cvloss))+2]
stats_m='ARIMA'
# step=2
seeds=5
FS_method='LR'
act='sigmoid'
for step in [1,2,4,24]:
#     resp=main()    
# def main():
    Nhs=np.arange(50,300,50)
    for nnn in range(5,6):
        Nls=[nnn]#np.arange(2,12,4)
        regs=[0]
        input_scale=[0.1]#,0.1,0.001]#[0.1,0.01,0.001]
        ratios=np.arange(0.05,1,0.05)
        deepRVFL_hypers=list(product(Nhs,regs,input_scale,ratios))
        order=24
       
        
        
        # pd_d[pd_d['PROFILEAREA']=='CITIPOWER']
        # data_=data_all[:,3:]
        boat=100
        
        # for name in ['27035_gdf']:
        for year in ['2017','2018','2019'][:]:
            for station in ['46083h','46080h','46076h','46001h'][:]:
                features=[ 'WDIR', 'WSPD', 'GST','APD','WVHT']
                #/Users/GaoRuobin/OneDrive - Nanyang Technological University
                data=pd.read_csv('Ocean energy/'+station+year+'.txt',delim_whitespace=True)
                data=data[features]
                # data.fillna(method='ffill')
                # var_name=data.columns
                data=data.where(data!='99.0',np.nan)
                data=data.where(data!='99.00',np.nan)
                data=data.fillna(method='ffill')
                while data.isnull().values.any():
                    data=data.fillna(method='ffill')
                # print(data.isnull().values.any())
                # data[data['WVHT']=='99.00']=np.median(data['WVHT'].values[1:].astype(np.float))
                # data[data['DPD']=='99.00']=np.median(data['DPD'].values[1:].astype(np.float))
                # data[data['GST']=='99.00']=np.median(data['GST'].values[1:].astype(np.float))
                # print(data.values[:])
                data_=data['WVHT'].values[1:].astype(float).reshape(-1,1)
                # data_[data_==99]=np.median(data_)
                
                # test_pres_ed=[]
                test_pres_ea=[]
                val_l,test_l=int(0.2*data_.shape[0]),int(0.2*data_.shape[0])
                ml_data=pd.DataFrame(data[features].values[1:,:].astype(float),columns=features)
                statspre=pd.read_csv('Ocean energy/Results/R_all_ARIMA'+station+year+str(step)+'.csv').values
                statstpre=pd.read_csv('Ocean energy/Results/R_test_ARIMA'+station+year+str(step)+'.csv').values
                
              
                statspre[-test_l:]=statstpre
                stats_len=statspre.shape[0]
                # print(data_.shape,statspre.shape)
                residuals=data_[-stats_len:,0]-statspre[-stats_len:,0]
                # ml_data['WVHT']=data_
                ml_data=pd.DataFrame(data[features].values[-stats_len:,:].astype(float),columns=features)
                ml_data['Res']=residuals
                ml_data=ml_data[['WDIR', 'WSPD', 'GST','APD','WVHT']]
                scaler=preprocessing.MinMaxScaler() 
                target_scaler=preprocessing.MinMaxScaler() 
                # target_scaler=preprocessing.StandardScaler()
                # for s in np.arange(seeds):
                target_scaler.fit(residuals[:-test_l-val_l].reshape(-1,1))
                # target_norm=2*target_scaler.transform(residuals.reshape(-1,1))-1
                target_norm=target_scaler.transform(residuals.reshape(-1,1))
                scaler.fit(ml_data.values[:-test_l-val_l])
                norm_data=scaler.transform(ml_data.values)
                # print(norm_data.shape)
                data=Struct()
                data.inputs,data.targets=format_data(norm_data,order,target_norm)
                train_l=data.inputs.shape[1]-val_l-test_l
                train_idx=range(train_l)
                val_idx=range(train_l,train_l+val_l)
                test_idx=range(train_l+val_l,data.inputs.shape[1])
                # np.random.seed(s)
                # train_idx=range(train_l+val_l)
                best_hypers=cross_validation(deepRVFL_hypers[:],data,
                                              # data_,
                                              data_[:-test_l],
                                                  train_idx,
                                                  val_idx,
                                                  Nls[0],regs,
                                                  input_scale,ratios=ratios,scaler=target_scaler,s=seeds,boat=boat)
                ed_best_hypers=ed_cross_validation(best_hypers,data,data_[:-test_l],
                                    train_idx,val_idx,Nls[0],scaler=scaler,s=seeds)
                print(step,year+station)
                #args['Nhs'],args['regs'],args['input_scale'],args['ratio']
                # print(ed_best_hypers)
                # print(np.array(ed_best_hypers))
                hyper_np=np.array(ed_best_hypers)#np.zeros((len(ed_best_hypers),len(ed_best_hypers[0])))
                hyper_pd=pd.DataFrame(hyper_np,columns=['Nh','reg','iss','ratio'])
                hyper_pd.to_csv('R_HyperAdd'+year+station+str(step)+stats_m+'edRVFL.csv')
                # hyper_pd=pd.read_csv('R_HyperAdd'+year+station+str(step)+stats_m+'edRVFL.csv')
                # hyper_np=hyper_pd[['Nh','reg','iss','ratio']].values
                # ed_best_hypers=[]
                # for i in range(5):
                #     ed_best_hypers.append(hyper_np[i,:])
                train_idx=range(train_l+val_l)
                scaler.fit(ml_data.values[:-test_l])
                norm_data=scaler.transform(ml_data.values)
                # target_scaler=preprocessing.MinMaxScaler() 
                target_scaler.fit(residuals[:-test_l].reshape(-1,1))
                # target_norm=2*target_scaler.transform(residuals.reshape(-1,1))-1
                target_norm=target_scaler.transform(residuals.reshape(-1,1))
                
                data.inputs,data.targets=format_data(norm_data,order,target_norm)
              
                NUMFS=[]
                for s in np.arange(seeds):
                    np.random.seed(s)
                    test_outputs_norm_mea,insample_norm,num_feas=edRVFL_predict(ed_best_hypers,data,train_idx,test_idx,s)
                    NUMFS.append(num_feas)
                    # plt.figure()
                    # plt.plot(target_norm[-test_l:],label='Res')
                    # plt.plot(test_outputs_norm_mea,label='P')
                    # plt.legend()
                    # plt.show()
                    
                    # test_outputs_ed=scaler.inverse_transform(test_outputs_norm_med)
                    # test_outputs_res=target_scaler.inverse_transform(0.5*(test_outputs_norm_mea+1))#+data_[-test_l-1:-1]
                    # print(test_outputs_res.shape)
                    test_outputs_res=target_scaler.inverse_transform(test_outputs_norm_mea)
                    test_outputs_ea=test_outputs_res[:,0]+statspre[-test_l:,0]
                    # test_outputs_ea=residuals[-test_l:]+statspre[-test_l:]
                    # print(test_outputs_ea.shape)
                    # test_pres_ed.append(test_outputs_ed)
                    test_pres_ea.append(test_outputs_ea.reshape(1,-1))
                    actuals=data_[-test_l:]
                    history=data_[:-test_l]
                    # plt.figure()
                    # plt.plot(actuals)
                    # plt.plot(test_outputs_ea)
                    # plt.show()
                    # plt.figure()
                    # plt.plot(residuals)
                    # plt.show()
                    # plt.figure()
                    # plt.plot(residuals[-test_l:],label='Res')
                    # plt.plot(test_outputs_res,label='P')
                    # plt.legend()
                    # plt.show()
                    naive_err=compute_error(actuals, data_[-test_l-step:-step],history)
                    
                    test_err=compute_error(actuals,test_outputs_ea,history)
                    print(naive_err)
                    print(test_err)
                    # print(len(ed_best_hypers))
                # test_p=np.concatenate(test_pres_ed,axis=1)
                # dfed=pd.DataFrame(test_p)
                # #D:\DeepRVFL-master\Results
                # dfed.to_csv('D:\\DeepRVFL-master\\Wind\\edRVFLmed'+loc+month_+'.csv')
                # plt.figure()
                # plt.plot(residuals[-test_l:],label='Res')
                # plt.plot(test_outputs_res,label='P')
                # plt.legend()
                # plt.show()
                NUMFS=np.concatenate(NUMFS,axis=0)
                df=pd.DataFrame(NUMFS)
                # df.to_csv('Ocean energy/Results/R_Hybrid_Add'+stats_m+'NUMFEASedRVFLFSLR'+str(Nls[0])+station+year+'step'+str(step)+'.csv')
                test_p=np.concatenate(test_pres_ea,axis=0)
                # print(test_p.shape)
                dfea=pd.DataFrame(test_p)
                hyper_pd=pd.read_csv('R_HyperAdd'+year+station+str(step)+stats_m+'edRVFL.csv')[['Nh','reg','iss','ratio']]
                idx=[]
                for i in range(5):
                    idx.append(NUMFS[i*5:(i+1)*5,1:2])
                idx=np.concatenate(idx,axis=1)
                values=np.concatenate((hyper_pd.values,idx),axis=1)
                columns=['Nh','reg','iss','ratio',0,1,2,3,4]
                p=pd.DataFrame(values,columns=columns)
                p.to_csv('R_HyperAdd'+year+station+str(step)+stats_m+'edRVFL.csv')    
                #D:\River flow forecasting\Results
                dfea.to_csv('New_run/R_Hybrid_Add'+stats_m+'edRVFL'+str(Nls[0])+station+year+str(boat)+'step'+str(step))

    # return    test_outputs_res
                
# if __name__ == "__main__":
#     stats_m='ARIMA'
#     # step=2
#     seeds=5
#     FS_method='LR'
#     act='sigmoid'
#     for step in [1,2,4,24]:
#         resp=main()
