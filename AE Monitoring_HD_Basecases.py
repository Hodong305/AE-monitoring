'''
(20. 09. 04)
    Balance the comparison basis. Increase training data of original data-based model (control group) to balance the size of the training data with augmented case (experimental group)
'''
'''  clear all variables '''
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# %clear
# import keras 
# import sklearn
# import tensorflow as tf  # Lab. PC ver
from tensorflow.keras.models import Sequential, Model # Lab. PC ver
from tensorflow.keras.layers import Input, Dense # Lab. PC ver
# from tensorflow.keras.models import load_model # Lab. PC ver
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt 
# from pandas import DataFrame
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras import metrics
import numpy as np
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

from sklearn.neighbors import KernelDensity # KDE를 이용한 (x, x_grid) fitting 정의(line 22)
from sklearn.model_selection import GridSearchCV  # (19.07.08 수정)
#from sklearn.grid_search import GridSearchCV   # (19.07.08 변경)
from tensorflow.keras import regularizers # (19.08.06 변경)
# from tensorflow.keras.layers import Activation # (19.08.07 변경)
from tensorflow.keras.callbacks import ModelCheckpoint # (19.08.07 변경)
import time
import pandas as pd
# (20.08.19 변경) VAE를 이용한 Edge_sampling_generation dataset 만을 이용한 Monitoring model 구축 및 성능 비교 평가
# (20. 09. 02) import .csv file(generated data from VAE) - directly import data without MATLAB processing(mio)
from tensorflow.keras.layers import LeakyReLU
import datetime
# from tensorflow.keras.optimizers import Adam
# import os
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras import optimizers

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

# kde function for what?!  - for Control Limit selection 
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):           # **kwargs : dict(key:value)형태의 인자 ex) {name]'hodong', age='29'}
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])                              # np.newaxis는 뭐지?!
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)
#%%
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # assign GPU index '1' 
# save_file_name = "fullDBVAE30dim_50var_Edgerev3_moreAug_Case9_20per_VBetarev3_3LayerAE_overfit_" # (20. 09. 25) setting the distinct file name in a one representative string variable
    # Case study 2. wrt. the type of Aug. data (Gen. 0(Normal) / 1(shrunk Normal) / 2(Edge1) / 3(Edge2) ) in the same AE structure (w/o BN)
# save_file_name = "fullDB50_[40_30dim_leakyReLU]_InfoVAE(New_78Epoch)_[28Faults]_44_40_34_30dimAE_AugAE_Case0(NoAug)_10kEpoch_basecase_" 
# save_file_name = "Server_TEST_BaseCase_ES(5e1-4_300)_"
save_file_name = "Server_BaseCase_NewTrials_shuffleTrain_15kEp_"
# Generate dummy data
import scipy.io as mio # matlab format file 다루기 위한 라이브러리
#data_all = mio.loadmat("D:\\Dropbox\\모니터링\\UGM\\UGMMonitoring_code\\data_abnormals.mat") # double backslashes in path directory
#data_all = mio.loadmat("D:\\Dropbox\\모니터링\\UGM\\UGMMonitoring_code\\data_abnormals_rev.mat")  # 41 vars.
data_all = mio.loadmat("/home/lhd305/download/TEP_50var_20faults(fault1000start).mat") # 50 vars.
data_all_F28 = mio.loadmat("/home/lhd305/download/TEP_50var_F21_28(fault1000start).mat") # 50 vars.
# data_all = mio.loadmat("D:/Dropbox/모니터링/Part1_Monitoring/212010 임시 백업/TEP_50var_20faults(fault1000start).mat") # Lab. PC ver
# data_all_F28 = mio.loadmat("D:/Dropbox/모니터링/Part1_Monitoring/212010 임시 백업/TEP_50var_F21_28(fault1000start).mat") # Lab. PC ver

'''
Normal data based case - used for (test data) scaling
'''
normal_data = mio.loadmat("/home/lhd305/download/TEP_50var_normal.mat") # 50 vars.
# norm_data = normal_data['data_normal'] 
norm_data_orig = normal_data['data_normal'] 
# norm_data_orig = norm_data_orig[0:5000,0:41] # use only 5000 data with augmentation adding (4000) Gen. data
# data_all = mio.loadmat("/home/lhd305/download/data_abnormals_rev.mat") # (20. 08. 22) 41 vars.

''' 
Generated data from VAE
(20. 09. 02) directly import .csv file (Generation data from VAE)
'''
import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_[40_30DimLatent](ES_test)_EdgeSampling_50MMDweight_20kSampling_Standardization_(72kTrDB)2101261433" # test (21.01.26) -  Standardization scaled InfoVAE - 50MMD/ES_test
# import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_[40_30DimLatent](80Epoch)_EdgeSampling_50MMDweight_20kSampling_Standardization_(72kTrDB)2101252129" # test (21.01.25) -  Standardization scaled InfoVAE - 50MMD/80Epoch
# import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_[40_30DimLatent](ES_rev)_EdgeSampling_1MMDweight_4CasesEdgeSample_20kSampling_Standardization_(144kTrDB)2101241717" # test (21.01.24) -  Standardization scaled InfoVAE - New trial
# import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_[40_30DimLatent](ES_rev)_EdgeSampling_1MMDweight_4CasesEdgeSample_20kSampling_Standardization_21012414" # test (21.01.24) - Standardization scaled InfoVAE - New trial
# import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_[40_30DimLatent](30Epoch)_EdgeSampling_5MMDweight_4CasesEdgeSample_20kSampling_Standardization_21012216" # test (21.01.22) - Standardization scaled InfoVAE
# import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_[40_30DimLatent](30Epoch)_EdgeSampling_5MMDweight_4CasesEdgeSample_20kSampling_21010716" # ~(21.01.22)
# import_generationdata_file_name = "Results_Gen_fullDB_vanillaVAE_50var_[40_30DimLatent]_EdgeSampling_5MMDweight_4CasesEdgeSample_20kSampling_21010619"
# import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_[30DimLatent]_EdgeSampling_2500MMDweight_4CasesEdgeSample_woBN_20kSampling_leakyReLUEncoder_20120523"
# import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_[40, 30DimLatent]_EdgeSampling_2500MMDweight_4CasesEdgeSample_woBN_20kSampling_20120523"
# import_generationdata_file_name =  "Results_Gen_fullDB_InfoVAE_50var_30DimLatent_EdgeSampling_2500MMDweight_3CasesEdgeSampleTest2_woBN_20112414"
# import_generationdata_file_name =  "Results_Gen_fullDB_InfoVAE_50var_[45DimLatent]_EdgeSampling_2500MMDweight_4CasesEdgeSample_woBN_20kSampling_20120212" # 45 feature dim. VAE
# import_generationdata_file_name =  "Results_Gen_fullDB_InfoVAE_50var_30DimLatent_EdgeSampling_2500MMDweight_3CasesEdgeSampleTest_StandardizationDB_20110812" # best case '_Case1(Gen1&2&3)_2000Epoch_EdgeNarrow_95ver2011212148' 
# import_generationdata_file_name =  "Results_Gen_fullDB_InfoVAE_50var_30DimLatent_EdgeSampling_2500MMDweight_3CasesEdgeSampleTest_20110317" 
# import_generationdata_file_name = "Results_Gen_fullDB_InfoVAE_50var_30DimLatent_EdgeSampling_500MMDweight_20102317" # (10.24) InfoVAE first test
# import_generationdata_file_name = "Results_Gen_smallDB_VAE_50var_30DimLatent_EdgeSampling_rev2_VBetarev3_overfit_20100619" # ppt result case Gen. DB
# import_generationdata_file_name = "Results_Gen_fullDB_VAE_50var_50DimLatent_EdgeSampling_41ver_VBetarev3_20101019" # test
# (20. 09. 08) - 27 feature dim.
# normal_data_gen = np.loadtxt("/home/lhd305/download/VAE/result/Gen_data/Results_Gen_(27dim)ES 20090816.csv", delimiter ="," ,dtype = np.float32)
# normal_data_gen = np.loadtxt("/home/lhd305/download/VAE/result/Gen_data/Results_Gen_smallTrainDB_Edge(1_2sig)_ 20091711.csv", delimiter ="," ,dtype = np.float32) # 41var important Result added in hwp record.
# normal_data_gen = np.loadtxt("/home/lhd305/download/VAE/result/Gen_data/Results_Gen_smallDB_VAE_50var_preservingDiminLatent_20092814.csv", delimiter ="," ,dtype = np.float32) # (20. 09. 28)
# normal_data_gen = np.loadtxt("/home/lhd305/download/VAE/result/Gen_data/Results_Gen_fullDB_VAE_50var_30DimLatent_20092815.csv", delimiter ="," ,dtype = np.float32) # (20. 09. 28)
# normal_data_gen = np.loadtxt("/home/lhd305/download/VAE/result/Gen_data/Results_Gen_fullDB_VAE_50var_30DimLatent_EdgeSampling_rev2_20092921.csv", delimiter ="," ,dtype = np.float32) # (20. 09. 28)
normal_data_gen = np.loadtxt("/home/lhd305/download/VAE/result/Gen_data/"+import_generationdata_file_name +".csv", delimiter ="," ,dtype = np.float32) # (20. 10. 05)


    # [ Normal dis. sampling; Edge1 ; Edge2 ] 10,000 samples each
# norm_data_gen = normal_data_gen[:10000,:]   # Normal dist. sampling
# norm_data_gen1 = normal_data_gen[10000:20000,:]   # normal distribution shrunken by half variance(std)
# norm_data_gen2 = normal_data_gen[20000:30000,:]   # Edge-1 sigma
# norm_data_gen3 = normal_data_gen[30000:40000,:]   # Edge-2 sigma
    
    # [ Normal dis. sampling; Edge1 ; Edge2 ] 20,000 samples each
norm_data_gen = normal_data_gen[:20000,:]   # Normal dist. sampling
norm_data_gen1 = normal_data_gen[20000:40000,:]   # normal distribution shrunken by half variance(std)
norm_data_gen2 = normal_data_gen[40000:60000,:]   # Edge-1 sigma
norm_data_gen3 = normal_data_gen[60000:80000,:]   # Edge-2 sigma
norm_data_gen4 = normal_data_gen[80000:100000,:]   # 
norm_data_gen5 = normal_data_gen[100000:120000,:]   #
norm_data_genE1 = normal_data_gen[120000:140000,:]   #
norm_data_gen6 = normal_data_gen[140000:160000,:]   #
norm_data_gen7 = normal_data_gen[160000:180000,:]   #
norm_data_gen8 = normal_data_gen[180000:200000,:]   #
norm_data_gen9 = normal_data_gen[200000:220000,:]   #
norm_data_genE2 = normal_data_gen[220000:240000,:]   #

# Z_dim_check = np.loadtxt("/home/lhd305/download/VAE/result/Gen_data/Results_Z_sampling_smallTrainDB_50vars_ 20092114.csv", delimiter ="," ,dtype = np.float32) # (20. 09. 21)

# 50var. version inputs
# (20. 08. 19 수정) 41var. 로 변경

# norm_data = norm_data_orig[:,0:41]  # (41 vars ver)
norm_data = norm_data_orig# (50 vars ver)
#norm_data = norm_data[:201,:]
ab1_data  = data_all['data_F1_trial1']
ab2_data  = data_all['data_F2_trial1']
ab3_data  = data_all['data_F3_trial1']
ab4_data  = data_all['data_F4_trial1']
ab5_data  = data_all['data_F5_trial1']
ab6_data  = data_all['data_F6_trial1']
ab7_data  = data_all['data_F7_trial1']
ab8_data  = data_all['data_F8_trial1']
ab9_data  = data_all['data_F9_trial1']
ab10_data  = data_all['data_F10_trial1']
ab11_data  = data_all['data_F11_trial1']
ab12_data  = data_all['data_F12_trial1']
ab13_data  = data_all['data_F13_trial1']
ab14_data  = data_all['data_F14_trial1']
ab15_data  = data_all['data_F15_trial1']
ab16_data  = data_all['data_F16_trial1']
ab17_data  = data_all['data_F17_trial1']
ab18_data  = data_all['data_F18_trial1']
ab19_data  = data_all['data_F19_trial1']
ab20_data  = data_all['data_F20_trial1']

ab21_data  = data_all_F28['data_F21']
ab22_data  = data_all_F28['data_F22']
ab23_data  = data_all_F28['data_F23']
ab24_data  = data_all_F28['data_F24']
ab25_data  = data_all_F28['data_F25']
ab26_data  = data_all_F28['data_F26']
ab27_data  = data_all_F28['data_F27']
ab28_data  = data_all_F28['data_F28']

ab_data = [ab1_data, ab2_data, ab3_data, ab4_data, ab5_data, ab6_data, ab7_data, ab8_data, ab9_data, ab10_data, ab11_data, ab12_data, ab13_data, ab14_data, ab15_data, ab16_data, ab17_data, ab18_data, ab19_data, ab20_data, norm_data]
ab_data_28 = [ab1_data, ab2_data, ab3_data, ab4_data, ab5_data, ab6_data, ab7_data, ab8_data, ab9_data, ab10_data, ab11_data, ab12_data, ab13_data, ab14_data, ab15_data, ab16_data, ab17_data, ab18_data, ab19_data, ab20_data, ab21_data, ab22_data, ab23_data, ab24_data, ab25_data, ab26_data, ab27_data, ab28_data, norm_data]
    # 41 var. case
# ab_data = [ab1_data[:,0:41], ab2_data[:,0:41], ab3_data[:,0:41], ab4_data[:,0:41], ab5_data[:,0:41], ab6_data[:,0:41], ab7_data[:,0:41], ab8_data[:,0:41], ab9_data[:,0:41], ab10_data[:,0:41], ab11_data[:,0:41], ab12_data[:,0:41], ab13_data[:,0:41], ab14_data[:,0:41], ab15_data[:,0:41], ab16_data[:,0:41], ab17_data[:,0:41], ab18_data[:,0:41], ab19_data[:,0:41], ab20_data[:,0:41], norm_data[:,0:41]]
# 41 var. version inputs
#norm_data = data_all['data_normal']
#ab1_data  = data_all['data_abnormal1']
#ab4_data  = data_all['data_abnormal4']
#ab9_data  = data_all['data_abnormal9']
#print(norm_data.shape) # (7201,41)
#print(ab1_data.shape) # (7201,41)
#%%
# this is the size of our encoded representations(the number of nodes in hidden layer)
input_dim=np.size(norm_data,1)
# input_dim=41    # (20. 08. 22) 41 vars
encoding_dim0 = 46
encoding_dim1 = 42  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats  ?!?!?!?!?!?!?
encoding_dim2 = 38
encoding_dim3 = 34
# encoding_dim4 = 30
# feature_dim = 30
feature_dim = 30
decoding_dim1 = 34
decoding_dim2 = 38
decoding_dim3 = 42
decoding_dim4 = 46

    # ref> https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers
    # kernel_regularizer : regul. on the weight values of the layers(prevent large weight values of the network)
    # activity_regularizer : regul. on the output of the layer (kind of activation f'n punishing unintentionally massive output)
autoencoder = Sequential()

    #Encoder Layers
autoencoder.add(Dense(encoding_dim0, input_shape=(input_dim,),kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1)))
# autoencoder.add(Dense(encoding_dim1, input_shape=(input_dim,),use_bias=(True), kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1))) # (21.01.11) use_bias : adverse effect
autoencoder.add(LeakyReLU(alpha=0.2 ))# (20. 09. 17)
# autoencoder.add(Dense(encoding_dim2,activation='relu',kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))# (20. 09. 10)
# autoencoder.add(Dense(encoding_dim2,kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))# (20. 09. 10)
autoencoder.add(Dense(encoding_dim1,kernel_initializer = 'he_normal'))# (20. 09. 10)
autoencoder.add(LeakyReLU(alpha=0.2 ))# (20. 09. 17)
autoencoder.add(Dense(encoding_dim2,kernel_initializer = 'he_normal'))# (21. 01. 17)
autoencoder.add(LeakyReLU(alpha=0.2 ))# (21. 01. 17)
autoencoder.add(Dense(encoding_dim3,kernel_initializer = 'he_normal'))# (21. 01. 27) - 11Layer AE test
autoencoder.add(LeakyReLU(alpha=0.2 ))# (21. 01. 27) - 11Layer AE test
# autoencoder.add(Dense(encoding_dim3,activation='relu',kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))# (20. 09. 10)
# autoencoder.add(Dense(encoding_dim4,activation='relu',kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))# (20. 09. 10)
#autoencoder.add(Activation('relu'))
# autoencoder.add(Dense(feature_dim, activity_regularizer = regularizers.l1_l2(l1=0,l2=0.01))) # (20. 09. 10)
# autoencoder.add(Dense(feature_dim,use_bias=(True), activation='tanh'))# (21. 01. 08)
# autoencoder.add(Dense(feature_dim,activation='tanh'))# (21. 01. 08)
# autoencoder.add(Dense(feature_dim, activity_regularizer = regularizers.l1_l2(l1=0,l2=0.01)  ))# (21. 01. 27)
autoencoder.add(Dense(feature_dim )) # (21. 01. 25) - Output (especially) should be able to cover the value range which covers more than '3' absolute unit.
                                                                    # (21. 01. 25) - feature output may be appropriate to use 'tanh' activation if it is beneficial.
# autoencoder.add(LeakyReLU(alpha=0.2 ))# (21. 01. 08)

    #Decoder Layers
autoencoder.add(Dense(decoding_dim1,kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))
# autoencoder.add(Dense(decoding_dim1,use_bias=(True) ,kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))
autoencoder.add(LeakyReLU(alpha=0.2 ))# (20. 09. 17)
# autoencoder.add(Dense(decoding_dim2, activation='relu',kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))# (20. 09. 10)
# autoencoder.add(Dense(decoding_dim2, kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))# (20. 09. 10)
autoencoder.add(Dense(decoding_dim2, kernel_initializer = 'he_normal' ))# (20. 09. 10)
autoencoder.add(LeakyReLU(alpha=0.2 ))# (20. 09. 17)
autoencoder.add(Dense(decoding_dim3, kernel_initializer = 'he_normal' ))# (21. 01. 17)
autoencoder.add(LeakyReLU(alpha=0.2 ))# (21. 01. 17)
autoencoder.add(Dense(decoding_dim4, kernel_initializer = 'he_normal' ))# (21. 01. 27) - 11Layer AE test
autoencoder.add(LeakyReLU(alpha=0.2 ))# (21. 01. 27) - 11Layer AE test
# autoencoder.add(Dense(decoding_dim3, activation='relu',kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))# (20. 09. 10)
# autoencoder.add(Dense(decoding_dim4, activation='relu',kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.1) ))# (20. 09. 10)

# autoencoder.add(Dense(input_dim, use_bias=(True) ,activation='tanh', activity_regularizer = regularizers.l1_l2(l1=0,l2=0.01) ))# (20. 09. 10)
# autoencoder.add(Dense(input_dim,activation='tanh', activity_regularizer = regularizers.l1_l2(l1=0,l2=0.01) ))# (20. 09. 10)
# autoencoder.add(Dense(input_dim , activity_regularizer = regularizers.l1_l2(l1=0,l2=0.01)))# (21. 01. 25) - Output (especially) should be able to cover the value range which covers more than '3' absolute unit.
autoencoder.add(Dense(input_dim )) # (21. 01. 08)
# autoencoder.add(LeakyReLU(alpha=0.2 ))# (21. 01. 08)

autoencoder.summary()  # prints a summary representation of my model

# this is our input placeholder
input_data = Input(shape=(input_dim,))

encoder_layer1=autoencoder.layers[0]
encoder_layer2=autoencoder.layers[1]
encoder_layer3=autoencoder.layers[2]
encoder_layer4=autoencoder.layers[3]
encoder_layer5=autoencoder.layers[4]
encoder_layer6=autoencoder.layers[5]
encoder_layer7=autoencoder.layers[6]
encoder_layer8=autoencoder.layers[7]
encoder_layer9=autoencoder.layers[8]
encoder=Model(input_data,encoder_layer9(encoder_layer8(encoder_layer7(encoder_layer6(encoder_layer5(encoder_layer4(encoder_layer3(encoder_layer2((encoder_layer1(input_data)))))))))))
# encoder=Model(input_data,encoder_layer7(encoder_layer6(encoder_layer5(encoder_layer4(encoder_layer3(encoder_layer2((encoder_layer1(input_data)))))))))
# encoder=Model(input_data,encoder_layer5(encoder_layer4(encoder_layer3(encoder_layer2((encoder_layer1(input_data)))))))
# encoder=Model(input_data,encoder_layer4(encoder_layer3(encoder_layer2((encoder_layer1(input_data))))))
# encoder=Model(input_data,encoder_layer3(encoder_layer2((encoder_layer1(input_data)))))
# encoder=Model(input_data,(encoder_layer2(encoder_layer1(input_data))))


encoder.summary()

encoded_input = Input(shape=(feature_dim,))

decoder_layer1=autoencoder.layers[9]
decoder_layer2=autoencoder.layers[10]
decoder_layer3=autoencoder.layers[11]
decoder_layer4=autoencoder.layers[12]
decoder_layer5=autoencoder.layers[13]
decoder_layer6=autoencoder.layers[14]
decoder_layer7=autoencoder.layers[15]
decoder_layer8=autoencoder.layers[16]
decoder_layer9=autoencoder.layers[17]
decoder=Model(encoded_input, decoder_layer9(decoder_layer8(decoder_layer7(decoder_layer6(decoder_layer5(decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))))))))
# decoder=Model(encoded_input, decoder_layer7(decoder_layer6(decoder_layer5(decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))))))
# decoder=Model(encoded_input, decoder_layer5(decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))))
# decoder=Model(encoded_input, decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input)))))
# decoder=Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
# decoder=Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))



decoder.summary()
    # AE에서 encoder부분만 따로 모델 생성
# this model maps an input to its encoded representation
    
#encoder = Model(input_data, feature)   # 41-30 의 FC model
    # AE에서 decoder 부분만 따로 모델 생성
# create a placeholder for an encoded (32-dimensional) input
#encoded_input = Input(shape=(feature_dim,))

# retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[4:6]
# create the decoder model
#decoder = Model(inputs=encoded_input, outputs=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
#decoder = Model(encoded_input,output_layer)  # 가능한가?!

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])  # Sequential 로 training tool 구성
# autoencoder.compile(optimizer=Adam(lr = 0.001/2, beta_1 = 0.9, beta_2=0.999), loss='mean_squared_error', metrics=['mse'])  # Test (difference by Adam optimizer setting ) - Result in similar performance excluding the training trajectory
# autoencoder.compile(optimizer=RMSprop(lr = 0.001, rho = 0.9, decay = 0 ), loss='mean_squared_error', metrics=['mse'])  # Test 'RMSprop'
# autoencoder.compile(optimizer=SGD(lr = 0.1, decay = 1e-6, momentum = 0.9), loss='mean_squared_error', metrics=['mse'])  # Test 'SGD'
# autoencoder.compile(optimizer=Adam(lr = 0.001), loss='mean_squared_error', metrics=['mse'])  # Test (difference by Adam optimizer setting ) - Result in similar performance excluding the training trajectory
#  binary-crossentropy는 supervised learning에서 주로 사용하는 loss척도 / regression : MSE
#  metrics : 다중 class 문제에서는 customized metrics 사용가능 / metrics=[metrics.mse] 가능
#  regression 문제에서는 'accuracy'로 충분?!
# 
#mean_data = np.mean(norm_data,axis=0)   # data standardization 위해서
#std_data  = np.std(norm_data,axis=0)
##for Small Training Data case
#%%
''' Preprocessing for training data preparation (only scaling norm_data_orig & merge it with norm_data_gen(already scaled output))'''
#x_train = norm_data_orig[:4000,:]  # 5000개로 training
#x_valid  = norm_data_orig[4000:501,:]  # 나머지로 normal 에 대한 testing
    # slicing - original data : 'norm_data_orig' : unscaled original data -> need to be scaled

    

# X_train_all = np.concatenate((x_train,x_valid),axis=0)

'''# Scaling - [-1, 1]'''
# x_train_min = np.min(norm_data[:6000,:], axis = 0 )
# x_train_max = np.max(norm_data[:6000,:],axis = 0 )
# scalefactor = x_train_max - x_train_min
# shifting_factor = x_train_max+x_train_min
# train_total_data = (2.* norm_data[:6000,:] - shifting_factor)/(scalefactor)
# valid_total_data = (2.* norm_data[6000:,:] - shifting_factor)/(scalefactor)

'''# Scaling - Standardization (standardized Gaussian)~ [-3sig, 3sig]'''
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

# norm_data_shuffle = shuffle_along_axis(norm_data, axis=1)
# test_shuffle = test[ : , np.random.permutation(test.shape[1])]
norm_data_shuffle = norm_data[np.random.permutation(norm_data.shape[0]), :]
# x_train_orig = norm_data[:6000,:]  # 6000
# x_valid_orig  = norm_data[6000:,:]  # 1201
x_train_orig = norm_data_shuffle[:6000,:]  # 6000
x_valid_orig  = norm_data_shuffle[6000:,:]  # 1201
mean_data = np.mean(norm_data_orig,axis=0)   # processing of the orig. data(in real unit) for data standardization 
std_data  = np.std(norm_data_orig,axis=0)

''' Original data case'''
x_train_orig_scaled = (x_train_orig-mean_data)/std_data   # data standardization
x_valid_orig_scaled = (x_valid_orig-mean_data)/std_data
    

''' Gen. data case'''
    #  scaling using Gen. data
# mean_data = np.mean(norm_data_gen,axis=0)   # processing of the orig. data(in real unit) for data standardization 
# std_data  = np.std(norm_data_gen,axis=0)

    # slicing - generation data
# x_train_gen0 = norm_data_gen[:18000,:]    # 4000 : 4.5 times of the original small Train DB
# x_valid_gen0 = norm_data_gen[18000:20000,:]   # 1000 : 4.5 times 120
# x_train_gen0 = norm_data_gen[:4000,:]    # 4000 : 4.5 times of the original small Train DB
# x_valid_gen0 = norm_data_gen[4000:5000,:]   # 1000 : 4.5 times 120
x_train_gen0 = norm_data_gen[:6000,:]    # 4000 : 4.5 times of the original small Train DB
x_valid_gen0 = norm_data_gen[6000:7000,:]   # 1000 : 4.5 times 120
    
# x_train_gen1 = norm_data_gen1[:18000,:]    # 4000 : 4.5 times of the original small Train DB
# x_valid_gen1 = norm_data_gen1[18000:20000,:]   # 1000 : 4.5 times 120
# x_train_gen1 = norm_data_gen1[:4000,:]    # 4000 : 4.5 times of the original small Train DB
# x_valid_gen1 = norm_data_gen1[4000:5000,:]   # 1000 : 4.5 times 120
x_train_gen1 = norm_data_gen1[:6000,:]    # 4000 : 4.5 times of the original small Train DB
x_valid_gen1 = norm_data_gen1[6000:7000,:]   # 1000 : 4.5 times 120

# x_train_gen2 = norm_data_gen2[:1000,:]    # 4000 : 4.5 times of the original small Train DB
# x_valid_gen2 = norm_data_gen2[1000:1200,:]   # 1000
# x_train_gen3 = norm_data_gen3[:1000,:]
# x_valid_gen3 = norm_data_gen3[1000:1200,:]
# x_train_gen4 = norm_data_gen4[:1000,:]
# x_valid_gen4 = norm_data_gen4[1000:1200,:]
# x_train_gen5 = norm_data_gen5[:1000,:]
# x_valid_gen5 = norm_data_gen5[1000:1200,:]
# x_train_genE1 = norm_data_genE1[:1000,:]
# x_valid_genE1 = norm_data_genE1[1000:1200,:]

    # 250 inverse Aug.
# x_train_gen2 = norm_data_gen2[:250,:]    
# x_valid_gen2 = norm_data_gen2[250:300,:]   
# x_train_gen3 = norm_data_gen3[:500,:]
# x_valid_gen3 = norm_data_gen3[500:600,:]
# x_train_gen4 = norm_data_gen4[:750,:]
# x_valid_gen4 = norm_data_gen4[750:900,:]
# x_train_gen5 = norm_data_gen5[:1000,:]
# x_valid_gen5 = norm_data_gen5[1000:1200,:]
# x_train_genE1 = norm_data_genE1[:1250,:]
# x_valid_genE1 = norm_data_genE1[1250:1500,:]
# x_train_gen5 = norm_data_gen5[:750,:]
# x_valid_gen5 = norm_data_gen5[750:900,:]
# x_train_genE1 = norm_data_genE1[:750,:]
# x_valid_genE1 = norm_data_genE1[750:900,:]

    # Exactly Quarter size Aug.
x_train_gen2 = norm_data_gen2[:200,:]    
x_valid_gen2 = norm_data_gen2[200:240,:]   
x_train_gen3 = norm_data_gen3[:400,:]
x_valid_gen3 = norm_data_gen3[400:480,:]
x_train_gen4 = norm_data_gen4[:600,:]
x_valid_gen4 = norm_data_gen4[600:720,:]
x_train_gen5 = norm_data_gen5[:800,:]
x_valid_gen5 = norm_data_gen5[800:960,:]
x_train_genE1 = norm_data_genE1[:1000,:]
x_valid_genE1 = norm_data_genE1[1000:1200,:]

    # Smaller than Quarter size Aug.
#x_train_gen2 = norm_data_gen2[:150,:]    # 4000 : 4.5 times of the original small Train DB
#x_valid_gen2 = norm_data_gen2[250:300,:]   # 1000
#x_train_gen3 = norm_data_gen3[:200,:]
#x_valid_gen3 = norm_data_gen3[500:600,:]
#x_train_gen4 = norm_data_gen4[:350,:]
#x_valid_gen4 = norm_data_gen4[750:900,:]
#x_train_gen5 = norm_data_gen5[:500,:]
#x_valid_gen5 = norm_data_gen5[1000:1200,:]
#x_train_genE1 = norm_data_genE1[:750,:]
#x_valid_genE1 = norm_data_genE1[1250:1500,:]

x_train_gen6 = norm_data_gen6[:1000,:]
x_valid_gen6 = norm_data_gen6[1000:1200,:]
x_train_gen7 = norm_data_gen7[:2000,:]
x_valid_gen7 = norm_data_gen7[2000:2400,:]
x_train_gen8 = norm_data_gen8[:3000,:]
x_valid_gen8 = norm_data_gen8[3000:3600,:]
x_train_gen9 = norm_data_gen9[:4000,:]
x_valid_gen9 = norm_data_gen9[4000:4800,:]
x_train_genE2 = norm_data_genE2[:4000,:]
x_valid_genE2 = norm_data_genE2[4000:4800,:]

    # Merge
x_train_merge = np.vstack([x_train_orig_scaled]) 
x_valid_merge = np.vstack([x_valid_orig_scaled])
    # Merge
# x_train_merge = np.vstack([x_train_orig_scaled,x_train_gen2, x_train_gen3, x_train_gen4, x_train_gen5, x_train_genE1]) # (4000+4000, 41)
# x_valid_merge = np.vstack([x_valid_orig_scaled,x_valid_gen2, x_valid_gen3, x_valid_gen4, x_valid_gen5, x_valid_genE1]) # (2201+1000, 41)

    # Input channel to unify the rest of the code
X_Train_Input = x_train_merge
X_Valid_Input = x_valid_merge
''' Training - AE model '''
# print(x_train.shape)
# print(x_valid.shape)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint # (21. 02. 24) To pinpoint the best model bsed on the val_loss
mc = ModelCheckpoint('/home/lhd305/download/AE/save_models/Best_Model_'+save_file_name+time.strftime('%y%m%d%H%M',time.localtime(time.time()))+'.h5', monitor='val_loss', mode='min', verbose = 0, save_best_only=(True))
# 'val_loss' 를 기준으로 'min'(감소) 가 '1e-3' 보다 '10' epoch 이상 만족되지 않을시 EarlyStop callback
#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=0, mode='min')
    # the smaller 'min_del' and the bigger 'patience' , the more trained the network is. 
early_stopping = EarlyStopping(monitor='val_loss', min_delta=7*1e-4, patience=200, verbose=1, mode='min')  # stopping 'callback fn' in the case of overfitting
# early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='min')  # stopping 'callback fn' in the case of overfitting
#mc = ModelCheckpoint('best_model_HD.h5', save_best_only=True)  # save best model after training
tic()
hist = autoencoder.fit(X_Train_Input, X_Train_Input,  # training data, target(label) data : AE 이기 때문에 y_data = x_data
                # epochs=2000,
                epochs=15000,
                batch_size=256,# modest Training dataset case
                # batch_size=50, # small Training dataset case
                # batch_size=175, # only 3000 Aug. Training dataset case - (4200 / 24 batches = 175)
                # shuffle=True, # uncertain 
                # validation_data=(x_valid, x_valid))
                validation_data=(X_Valid_Input, X_Valid_Input)
                # ,callbacks=[early_stopping,mc])
                   , )
                
print('Training time : ')
toc()

    # Plot the training process 
fig, loss_ax = plt.subplots()

#acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'][50:], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'][50:], 'r', label='val loss')

#acc_ax.plot(hist.history['acc'], 'b', label='train acc')
#acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
#acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
#acc_ax.legend(loc='lower left')

# plt.show()
# plt.close()
''' detail Train Loss trajectory '''
    # Plot the training process 
fig, loss_ax = plt.subplots()

#acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'][50:], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

#acc_ax.plot(hist.history['acc'], 'b', label='train acc')
#acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
#acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
#acc_ax.legend(loc='lower left')

# plt.show()
# plt.close()
''' detail Valid Loss trajectory '''
    # Plot the training process 
fig, loss_ax = plt.subplots()

#acc_ax = loss_ax.twinx()

# loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'][50:], 'r', label='val loss')

#acc_ax.plot(hist.history['acc'], 'b', label='train acc')
#acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
#acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
#acc_ax.legend(loc='lower left')

# plt.show()
# plt.close()
Train_loss = hist.history['loss']
Valid_loss = hist.history['val_loss']
Loss = [Train_loss, Valid_loss]

np.savetxt("/home/lhd305/download/AE/training_history/TrainHistory_"+save_file_name+time.strftime('%y%m%d%H%M',time.localtime(time.time())) + ".csv", Loss, fmt='%f', delimiter=',')
#%%
# calculating the h value and the e value
#h_val = encoder.predict(x_valid)  # encoder의 prediction value
#e_val = x_valid - decoder.predict(h_val)  # encoder의 prediction error value
#
#
## calculate statistics - 필요없음?!
#h_squares = np.dot(h_val,np.transpose(h_val))         # change it so it does transpose calculation / 30 by 30 array
#e_squares = np.dot(e_val,np.transpose(e_val))         # change it so it does transpose calculation / 41 by 41 array
#h_square = np.diagonal(h_squares)             # only the diagonal values are relevant
#e_square = np.diagonal(e_squares)             # 대각 성분만 추출해 낸 것
# print(h_square)
# print(e_square)
# print(h_square.shape)
# print(e_square.shape)

    #Preparation for KDE on total norm_data[:,:]
# norm_data_scaled = (X_train_all-mean_data)/std_data # 
#norm_data_scaled = (norm_data)/x_train_range
'''
Calculate the monitoring matric (H2, e2) to make Control limit in two different methods
1. KDE - backup(cross check) : original theretical basis but no module in Python
2. ecdf(mlxtend) - used in this code : alternative method instead of KDE
'''
    # (20. 09. 29) - TEST : use the original normal data when setting the control limit
X_CL_Input = X_Train_Input
# X_CL_Input = np.vstack([x_train_merge])
# X_CL_Input = np.vstack([x_train_orig_scaled, x_valid_orig_scaled])
# X_CL_Input = np.vstack([train_total_data, valid_total_data])
# X_CL_Input = np.vstack([x_train_merge, x_valid_merge])
# X_CL_Input = np.vstack([x_train_orig_scaled, x_valid_orig_scaled, x_train_gen1])

h_total_norm = encoder.predict(X_CL_Input)  # data for control limit decision using KDE
e_total_norm = X_CL_Input - decoder.predict(h_total_norm) 
h_squares_total = np.dot(h_total_norm,np.transpose(h_total_norm))         # change it so it does transpose calculation / 30 by 30 array
e_squares_total = np.dot(e_total_norm,np.transpose(e_total_norm))         # change it so it does transpose calculation / 41 by 41 array
h_square_total_Aug = np.diagonal(h_squares_total)             # only the diagonal values are relevant
e_square_total_Aug = np.diagonal(e_squares_total)             # 대각 성분만 추출해 낸 것

    # (20. 09. 29) - TEST : use the original normal data when setting the control limit
# X_CL_Input = X_Train_Input
X_CL_Input_orig = np.vstack([x_train_orig_scaled])

h_total_norm_orig = encoder.predict(X_CL_Input_orig)  # data for control limit decision using KDE
e_total_norm_orig = X_CL_Input_orig - decoder.predict(h_total_norm_orig) 
h_squares_total_orig = np.dot(h_total_norm_orig,np.transpose(h_total_norm_orig))         # change it so it does transpose calculation / 30 by 30 array
e_squares_total_orig = np.dot(e_total_norm_orig,np.transpose(e_total_norm_orig))         # change it so it does transpose calculation / 41 by 41 array
h_square_total_orig = np.diagonal(h_squares_total_orig)             # only the diagonal values are relevant
e_square_total_orig = np.diagonal(e_squares_total_orig)             # 대각 성분만 추출해 낸 것
# test = decoder.predict(h_total_norm) # for dimension check
'''
GridSearchCV는 처음 돌릴때만!, 오래걸리기때문에 best_param 결과 저장해두고 그것만 돌려서 쓰기!
'''    
    #Let's plot the result of 20-fold CV

tic()
grid_h = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.logspace(-1, 1.0, 20)},  # 'logspace' for broad search in initial phase (1st try to decide the range in linspace)
                    # {'bandwidth': np.linspace(0, 3.0, 20)},  # linspace(start, end, num) - precise search
                    cv=20) # 20-fold cross-validation
grid_h.fit(h_square_total_orig[:, None])
#h_best_param = {'bandwidth' : 155.1724} # 1/3 training data case
#h_best_param = {'bandwidth' : 77.5862} # 1/9 training data case
h_best_param = grid_h.best_params_ # 7-Layers_H2 : 113.7931
print('grid_h.best_params_ : ',grid_h.best_params_)
print('GridSearch time for H^2 : ')
toc()


tic()
grid_e = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.logspace(-1, 1, 20)}, # 'logspace' for broad search in initial phase (1st try to decide the range in linspace)
                    # {'bandwidth': np.linspace(1, 4, 20)}, # linspace(start, end, num) - precise search
                    cv=20) # 20-fold cross-validation
grid_e.fit(e_square_total_orig[:, None])
#e_best_param = {'bandwidth' : 2.80344} # 1/3 training data case
#e_best_param = {'bandwidth' : 4.66206} # 1/9 training data case
e_best_param = grid_e.best_params_ # 7-Layers_e2 : 3.81724
print('grid_e.best_params_ : ', grid_e.best_params_)
print('GridSearch time for e^2 : ')
toc()
''' Find the best parameter(bandwidth) for KDE '''
# x ~ h_squares (or e_sqaures), x_grid ~ range of h_squares (or e_squares)
#h_grid=np.linspace(min(h_square_total)-(max(h_square_total)-min(h_square_total))/2,max(h_square_total)+(max(h_square_total)-min(h_square_total))/2,100)
#e_grid=np.linspace(min(e_square_total)-(max(e_square_total)-min(e_square_total))/2,max(e_square_total)+(max(e_square_total)-min(e_square_total))/2,100)
h_padding=h_best_param['bandwidth'] # considering edge effect of (Gaussian)kernel
h_grid=np.linspace(min(h_square_total_orig)-h_padding,max(h_square_total_orig)+h_padding,100)
e_padding=e_best_param['bandwidth']
e_grid=np.linspace(min(e_square_total_orig)-e_padding,max(e_square_total_orig)+e_padding,100)

#(check point)final KDE for H^2 result using best 'grid_h.best_params_'(bandwidth)
kde_h=grid_h.best_estimator_
#Evaluate the density model on the data.(Compute the log-likelihood of each sample)
pdf_h=np.exp(kde_h.score_samples(h_grid[:,None])) # 
# threshold_h = np.percentile(pdf_h, 5) # 95 percentile value based on the KDE
fig, ax=plt.subplots()
plt.title('PDF of h^2')
ax.plot(h_grid,pdf_h)
ax.set_xlabel('H^2(h_grid)')
ax.set_ylabel('Probability')
plt.show()

''' find the 95 percentile based on the result of KDE for H2 '''
'''  '''
for k in range(0,100):
    if k ==0:
        xx = h_grid
        yy = pdf_h
    else:
        xx = h_grid[0:-k]
        yy = pdf_h[0:-k]
    v = np.trapz(yy,xx)
    print(f"Integral {k} from {xx[0]} to {xx[-1]} is equal to {v}")
    if v <= 0.95:
        break
h_threshold_Orig_95_KDE = xx[-1]

#(check point)final KDE for e^2 result using best 'grid_e.best_params_'(bandwidth)
kde_e=grid_e.best_estimator_
#Evaluate the density model on the data.(Compute the log-likelihood of each sample)
pdf_e=np.exp(kde_e.score_samples(e_grid[:,None]))
# threshold_e = np.percentile(pdf_e, 5) # 95 percentile value based on the KDE
fig, ax=plt.subplots()
plt.title('PDF of e^2')
ax.plot(e_grid,pdf_e)
ax.set_xlabel('e^2(e_grid)')
ax.set_ylabel('Probability')
plt.show()

''' find the 95 percentile based on the result of KDE for e2 '''
for k in range(0,100):
    if k ==0:
        xx = e_grid
        yy = pdf_e
    else:
        xx = e_grid[0:-k]
        yy = pdf_e[0:-k]
    v = np.trapz(yy,xx)
    print(f"Integral {k} from {xx[0]} to {xx[-1]} is equal to {v}")
    if v <= 0.95:
        break
e_threshold_Orig_95_KDE = xx[-1]
    
#%%
'''  After 'model_fit' and 'KDE' for detemining the control limit(bandwidth by Gridsearch) '  
    save models such as autoencoder, encoder and decoder respectively
'''

    # (20. 08. 22) Model save
    # Must change the name of the save
autoencoder.save('/home/lhd305/download/AE/save_models/Autoencoder_'+save_file_name+time.strftime('%y%m%d%H%M',time.localtime(time.time()))+'.h5') # (based on the original normal data)
encoder.save('/home/lhd305/download/AE/save_models/Encoder_'+save_file_name+time.strftime('%y%m%d%H%M',time.localtime(time.time()))+'.h5') # (based on the original normal data)
decoder.save('/home/lhd305/download/AE/save_models/Decoder_'+save_file_name+time.strftime('%y%m%d%H%M',time.localtime(time.time()))+'.h5') # (based on the original normal data)
# autoencoder.save('Model_gen_data_'+time.strftime('%y%m%d%H%M%S',time.localtime(time.time()))+'.h5') # (based on the generated normal data)

''' 
MATLAB을 통해 KDE-based Control Limit 계산 가능 (python에서는 방법을 못 찾음) 
대안으로 찾은 것이 아래의 ecdf(.) 인데, KDE를 이용한 방법인지는 알 수 없으나 결과는 비슷하게 나오는 것 확인
'''

#(Alternative 1 for Control Limit selection)Emperical Cumulative Density Function(ecdf)를 통해 Control Limit 구하기 (KDE 인지는 모르는데....ㅜㅜ)
from mlxtend.plotting import ecdf
'''
Find control limit based on the Augmented training data ( 'X_Train_Input')
'''
#plt.figure()
#ax_h_95,h_threshold_Aug_95,h_count_95=ecdf(x=h_square_total_Aug,x_label='H-square',percentile=0.95)
#ax_h_99,h_threshold_Aug_99,h_count_99=ecdf(x=h_square_total_Aug,x_label='H-square',percentile=0.99)
#
#plt.figure()
#ax_e_95,e_threshold_Aug_95,e_count_95=ecdf(x=e_square_total_Aug,x_label='e-square',percentile=0.95)
#ax_e_99,e_threshold_Aug_99,e_count_99=ecdf(x=e_square_total_Aug,x_label='e-square',percentile=0.99)

'''
Find control limit based on only 'x_train_orig_scaled' data - finally determined
'''
plt.figure()
ax_h_95,h_threshold_Orig_95,h_count_95=ecdf(x=h_square_total_orig,x_label='H-square',percentile=0.95)
ax_h_99,h_threshold_Orig_99,h_count_99=ecdf(x=h_square_total_orig,x_label='H-square',percentile=0.99)

plt.figure()
ax_e_95,e_threshold_Orig_95,e_count_95=ecdf(x=e_square_total_orig,x_label='e-square',percentile=0.95)
ax_e_99,e_threshold_Orig_99,e_count_99=ecdf(x=e_square_total_orig,x_label='e-square',percentile=0.99)
# h_95_limit = [h_threshold_99]*7201
# e_95_limit = [e_threshold_99]*7201

# simple 'percentile' control limit
# h_limit_95 = np.percentile(h_square_total,99,axis=0)  # h_square 의 95%에 
#h_limit_99 = np.percentile(h_square_total,99,axis=0) 
# e_limit = np.percentile(e_square_total,99,axis=0)
# h_limit = [h_limit_95]*7201
# e_limit = [e_limit]*7201

# test the abnormal data for process monitoring
# ab1_data, ab4_data, ab9_data
# first calculate the h value and e value of abnormal data
# df_normal = DataFrame(ab1_data)
# df_normal.to_csv('original_ab1data.csv')
           # 여기서부터 abnormal_1 data에 대한 monitoring
test_input_scaled = []           
h_sq_result=[]
e_sq_result=[]
for i in ab_data_28 :
    test_input = i              # 여기만 바꾸면 (50 vars. ver)
    # test_input = i[:,0:41]  # 여기만 바꾸면 (41 vars. ver)
    # test data의 mean, std가 아니라 train data의 mean, std로 normalization 해야하지 않나?!
    # scaling은 normal data에서 계산한 mean, std를 이용
    #mean_ab = np.mean(test_input,axis=0)  
    #std_ab  = np.std(test_input,axis=0)
    #ab_data_treated = (test_input-mean_ab)/std_ab 
     # normalized data 사용 안되고 있음
    test_input_normalized = (test_input-mean_data)/std_data
    test_input_scaled.append(test_input_normalized)
#    test_input_normalized = (test_input)/x_train_range

    # df_ab    = DataFrame(ab1_data)
    # df_ab.to_csv('normalized_ab1data.csv')
    
    h_val_ab = encoder.predict(test_input_normalized)   # input data를 normalized data로 입력하면?!
    e_val_ab = test_input_normalized - decoder.predict(h_val_ab)
    # calculate statistics
    h_squares_ab = np.dot(h_val_ab,np.transpose(h_val_ab))         # change it so it does transpose calculation / 30 by 30 array
    e_squares_ab = np.dot(e_val_ab,np.transpose(e_val_ab))         # change it so it does transpose calculation / 41 by 41 array
    h_square_ab = np.diagonal(h_squares_ab)             # only the diagonal values are relevant
    e_square_ab = np.diagonal(e_squares_ab)
    h_sq_result.append(h_square_ab)
    e_sq_result.append(e_square_ab)
#%%
# plot the monitoring results                         # 여기부터 아래는 한꺼번에 run 시켜야 plot이 된다. 
''' Aug DB case'''
# CTR_Limit_H_95 = h_threshold_Aug_95
# CTR_Limit_H_99 = h_threshold_Aug_99
# CTR_Limit_e_95 = e_threshold_Aug_95
# CTR_Limit_e_99 = e_threshold_Aug_99
'''Control Limit determined using "ecdf"  - Orig DB case'''
CTR_Limit_H_95 = h_threshold_Orig_95
CTR_Limit_H_99 = h_threshold_Orig_99
CTR_Limit_e_95 = e_threshold_Orig_95
CTR_Limit_e_99 = e_threshold_Orig_99

'''Control Limit determined using "KDE"  - Orig DB case'''
CTR_Limit_H_95 = h_threshold_Orig_95_KDE
# CTR_Limit_H_99 = h_threshold_Orig_99
CTR_Limit_e_95 = e_threshold_Orig_95_KDE
# CTR_Limit_e_99 = e_threshold_Orig_99

print( 'CTR Limit using ecdf : ',  h_threshold_Orig_95, ' / ' ,  e_threshold_Orig_95 ) 
print( 'CTR Limit using KDE : ',  h_threshold_Orig_95_KDE, ' / ' ,  e_threshold_Orig_95_KDE ) 
print('Error percentage of CTR Limit for H2 based on the ecdf compared to KDE : ',  np.abs(h_threshold_Orig_95_KDE - h_threshold_Orig_95)/  h_threshold_Orig_95_KDE * 100)
print('Error percentage of CTR Limit for e2 based on the ecdf compared to KDE : ',  np.abs(e_threshold_Orig_95_KDE - e_threshold_Orig_95)/  e_threshold_Orig_95_KDE * 100)

plot_case=29

plot_h = h_sq_result[plot_case-1] 
plot_e = e_sq_result[plot_case-1] 

x_grid = np.linspace(0, len(plot_h), num=len(plot_h))
fig, ax = plt.subplots(2,1,sharex=True)
h_95_limit = [CTR_Limit_H_95]*len(plot_h)
h_99_limit = [CTR_Limit_H_99]*len(plot_h)
e_95_limit = [CTR_Limit_e_95]*len(plot_h)
e_99_limit = [CTR_Limit_e_99]*len(plot_h)

# ax.set_xlim(0, 7201)
# ax.legend(loc='upper left')
#plt.subplot(2,1,1)
ax[0].plot(x_grid, plot_h,label='H^2 value', linewidth=0.5, color = 'blue')
ax[0].plot(x_grid, h_99_limit,linewidth = 1,color = 'green')
ax[0].plot(x_grid, h_95_limit,linewidth = 1,color = 'indigo')
ax[0].set_ylabel('H^2')
fig.suptitle('Fault case num_'+str(plot_case))
#plt.subplot(2,1,2)
ax[1].plot(x_grid, plot_e,label='E^2 value', linewidth=0.5, color = 'red')
ax[1].plot(x_grid, e_99_limit,linewidth = 1,color = 'green')
ax[1].plot(x_grid, e_95_limit,linewidth = 1,color = 'indigo')
ax[1].set_ylabel('e^2')
ax[1].set_xlabel('sample point')

plt.show()

''' Save the monitoring chart results to export it to plot in MATLAB'''
ControlLimits = [h_95_limit, e_95_limit]

DF_H2 = pd.DataFrame(h_sq_result)
DF_e2 = pd.DataFrame(e_sq_result)
CtrLimits = pd.DataFrame(ControlLimits)

DF_H2.to_csv("/home/lhd305/download/AE/save_monitoring_results/Monitoring_Chart_"+save_file_name+time.strftime('%y%m%d%H%M',time.localtime(time.time())) + "H2.csv")
DF_e2.to_csv("/home/lhd305/download/AE/save_monitoring_results/Monitoring_Chart_"+save_file_name+time.strftime('%y%m%d%H%M',time.localtime(time.time())) + "e2.csv")
CtrLimits.to_csv("/home/lhd305/download/AE/save_monitoring_results/Monitoring_Chart_"+save_file_name+time.strftime('%y%m%d%H%M',time.localtime(time.time())) + "CTRLimits.csv")

#%%
'''  95% Control Limit ver. '''
#calculate the performance metric of fault monitoring (FDR, FAR)
#FDR = # of detected sample / # of all sample point , FAR = # of falsely detected sample / # of all sample point
#FDR for H^2
fdr_H = []
far_H = []
num=0
for i in h_sq_result:
    #fdr_H calc.
    num=num+1
    count=0
    for j in range(999, len(i)-1): # fault start at t=1000
        if i[j] >= CTR_Limit_H_95:
            count=count+1
#        detect_pos[i].append(0)
#        else:
#            temp.append(1)
#        detect_pos[i].append(1)
    temp = round(count/(len(range(999, len(i)))-1),4)*100
    fdr_H.append(temp)
    # False Alarm Rate(far_H)
    count1=0
    count2=0
    for j in range(0, 1000-1): # false alarm rate is calculated in total monitoring time 
        if i[j] >= CTR_Limit_H_95:
            count1= count1+1
#    for j in range(999,len(i)-1): # miss detection rate (fault detection rate의 complement)
#        if i[j] < h_threshold_95:
#            count2=count2+1
    temp1 = round((count1+count2)/((len(range(0,999)))-1),4)*100
    far_H.append(temp1)
    if num==29: # Normal operation case (#29) 
        count1=0
        for j in range(0, len(i)-1): # false alarm rate is calculated in total monitoring time 
            if i[j] >= CTR_Limit_H_95:
                count1= count1+1
        temp1 = round((count1)/((len(i))-1),4)*100
        far_H[num-1]=temp1

fdr_e = []
far_e = []
num=0
for i in e_sq_result:
    #fdr_H calc.
    num=num+1
    count=0
    for j in range(999, len(i)-1): # fault start at t=1000
        if i[j] >= CTR_Limit_e_95:
            count=count+1
#        detect_pos[i].append(0)
#        else:
#            temp.append(1)
#        detect_pos[i].append(1)
    temp = round(count/(len(range(999, len(i)))-1),4)*100
    fdr_e.append(temp)
    # False Alarm Rate(far_H)
    count1=0
    count2=0
    for j in range(0, 1000-1): # false alarm rate is calculated in total monitoring time 
        if i[j] >= CTR_Limit_e_95:
            count1= count1+1
#    for j in range(999,len(i)-1): 
#        if i[j] < e_threshold_95:
#            count2=count2+1
    temp1 = round((count1+count2)/((len(range(0,999)))-1),4)*100
    far_e.append(temp1)
    if num==29:
        count1=0
        for j in range(0,len(i)-1): # false alarm rate is calculated in total monitoring time 
            if i[j] >= CTR_Limit_e_95:
                count1= count1+1
        temp1 = round((count1)/((len(i))-1),4)*100
        far_e[num-1]=temp1

''' Calculate the composite FDR of H2 & e2'''
fdr_composite = []

num=0 # Fault cases (1~28, 29)
for i in h_sq_result:
    num = num+1 # Fault number
    count = 0 # to count the number of sample points that cross the control limit(thus classified as a fault case)
    for j in range(999, len(i)-1):
        if i[j] >= CTR_Limit_H_95 or e_sq_result[num-1][j] >= CTR_Limit_e_95:
            count = count + 1
    temp = round(count/(len(range(999,len(i)))-1), 4)*100 # percentage of fault detection (rate)
    fdr_composite.append(temp)

''' Check the detection delay in H2 & e2 - How fast it detects the fault since it occurs'''
fdd_H2 = []
fdd_e2 = []
for i in h_sq_result:
    # detection delay in H2
    num=num+1
    breaker = False
    for j in range(1000, len(i)-1): # fault start at t=1000
        if i[j] > CTR_Limit_H_95:
            fdd_H2.append(j)
            break
    # if breaker ==True:
    #     break

for i in e_sq_result:
    # detection delay in e2
    num=num+1
    breaker = False
    for j in range(999, len(i)-1): # fault start at t=1000
        if i[j] > CTR_Limit_e_95:
            fdd_e2.append(j)
            break

''' Save the final result of process monitoring via the trained AE models'''
FDR_FAR_Augment = [fdr_H, far_H, fdr_e, far_e, fdr_composite, fdd_H2, fdd_e2]
np.savetxt("/home/lhd305/download/AE/save_monitoring_results/Monitoring_result_[28F]_"+save_file_name+"95ver"+time.strftime('%y%m%d%H%M',time.localtime(time.time())) + ".csv", FDR_FAR_Augment, delimiter=',')

#%%
'''  99% Control Limit ver. '''
#calculate the performance metric of fault monitoring (FDR, FAR)
#FDR = # of detected sample / # of all sample point , FAR = # of falsely detected sample / # of all sample point
#FDR for H^2
fdr_H_99 = []
far_H_99 = []
num=0
for i in h_sq_result:
    #fdr_H calc.
    num=num+1
    count=0
    for j in range(999, len(i)-1): # fault start at t=1000
        if i[j] >= CTR_Limit_H_99:
            count=count+1
#        detect_pos[i].append(0)
#        else:
#            temp.append(1)
#        detect_pos[i].append(1)
    temp = round(count/(len(range(999, len(i)))-1),4)*100
    fdr_H_99.append(temp)
    # False Alarm Rate(far_H)
    count1=0
    count2=0
    for j in range(0, 1000-1): # false alarm rate is calculated in total monitoring time 
        if i[j] >= CTR_Limit_H_99:
            count1= count1+1
#    for j in range(999,len(i)-1): # miss detection rate (fault detection rate의 complement)
#        if i[j] < h_threshold_99:
#            count2=count2+1
    temp1 = round((count1+count2)/((len(range(0,999)))-1),4)*100
    far_H_99.append(temp1)
    if num==29:
        count1=0
        for j in range(0, len(i)-1): # false alarm rate is calculated in total monitoring time 
            if i[j] >= CTR_Limit_H_99:
                count1= count1+1
        temp1 = round((count1)/((len(i))-1),4)*100
        far_H_99[num-1]=temp1

fdr_e_99 = []
far_e_99 = []
num=0
for i in e_sq_result:
    #fdr_H calc.
    num=num+1
    count=0
    for j in range(999, len(i)-1): # fault start at t=1000
        if i[j] >= CTR_Limit_e_99:
            count=count+1
#        detect_pos[i].append(0)
#        else:
#            temp.append(1)
#        detect_pos[i].append(1)
    temp = round(count/(len(range(999, len(i)))-1),4)*100
    fdr_e_99.append(temp)
    # False Alarm Rate(far_H)
    count1=0
    count2=0
    for j in range(0, 1000-1): # false alarm rate is calculated in total monitoring time 
        if i[j] >= CTR_Limit_e_99:
            count1= count1+1
#    for j in range(999,len(i)-1): 
#        if i[j] < e_threshold_99:
#            count2=count2+1
    temp1 = round((count1+count2)/((len(range(0,999)))-1),4)*100
    far_e_99.append(temp1)
    if num==29:
        count1=0
        for j in range(0,len(i)-1): # false alarm rate is calculated in total monitoring time 
            if i[j] >= CTR_Limit_e_99:
                count1= count1+1
        temp1 = round((count1)/((len(i))-1),4)*100
        far_e_99[num-1]=temp1


''' Save the final result of process monitoring via the trained AE models'''
FDR_FAR_Augment_99 = [fdr_H_99, far_H_99, fdr_e_99, far_e_99]
#np.savetxt("/home/lhd305/download/AE/save_monitoring_results/Monitoring_result_[28F]_"+save_file_name+"99ver"+time.strftime('%y%m%d%H%M',time.localtime(time.time())) + ".csv", FDR_FAR_Augment_99, fmt='%f', delimiter=',')

print( 'Simulation was finished at ',datetime.datetime.now())