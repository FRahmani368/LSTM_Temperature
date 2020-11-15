import sys
import os
import numpy as np
import pandas as pd
import pyarrow
import importlib
sys.path.append('G:\Farshid\CONUS_Temp\Example3')  ## HydroDL package
import pyarrow
import matplotlib.pyplot as plt
import torch
import random
import import_forcing

# random seed
randomseed = 0    # or None
if randomseed is not None:
    print('Hi')
    random.seed(randomseed)
    torch.manual_seed(randomseed)
    np.random.seed(randomseed)



### creating ML model

import torch
from torch.autograd import Variable
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x).clamp(min=0)
        return out

### create data
####### Address of forcing pandas dataframe file ################

forcing_path = os.path.join('G:\Farshid\CONUS_Temp\Example3\scratch\\SNTemp\Forcing\Forcing_new\\no_dam_forcing_60%_days118sites.feather')
forcingT = import_forcing.import_forcing(forcing_path)
#pd.read_feather(forcing_path)
site_no = forcingT['site_no'].unique()

forcingT.loc[(forcingT['00060_Mean'] != forcingT['00060_Mean']) & \
            (forcingT['datetime']>='2010-10-01') &  \
            (forcingT['datetime']<'2016-10-01'), '00060_Mean' ] = forcingT.loc[(forcingT['00060_Mean'] != forcingT['00060_Mean']) & \
            (forcingT['datetime']>='2010-10-01') &\
            (forcingT['datetime']<'2016-10-01'), 'combine_discharge' ]

forcing = forcingT.loc[(forcingT['datetime']>='2010-10-01') &
            (forcingT['datetime']<'2016-10-01')]


col1 = [ 'prcp(mm/day)', 'tmax(C)',
       'tmin(C)', '00060_Mean', '00010_Mean', 'datetime', 'site_no']
col = ['dayl(s)','srad(W/m2)', 'swe(mm)', 'vp(Pa)', 'site_no', '00010_Maximum',
       '00010_Mean', '00010_Minimum', 'site_no_from_TS',
       'pred_discharge', 'combine_discharge', 'datetime']

f_s_0 = forcingT[col1]

f_s_1 = forcingT.shift(periods=1)
f_s_1 = f_s_1.drop(col, axis=1)

f_s_2 = forcingT.shift(periods=2)
f_s_2 = f_s_2.drop(col, axis=1)

f_s_3 = forcingT.shift(periods=3)
f_s_3 = f_s_3.drop(col, axis=1)

f_s_4 = forcingT.shift(periods=4)
f_s_4 = f_s_4.drop(col, axis=1)

f_s_5 = forcingT.shift(periods=5)
f_s_5 = f_s_5.drop(col, axis=1)



ft = pd.concat([f_s_0, f_s_1, f_s_2, f_s_3, f_s_4, f_s_5], axis = 1)



ft.columns = ['prcp(mm/day)', 'tmax(C)', 'tmin(C)', '00060_Mean', '00010_Mean', 'datetime','site_no',
              'prcp(mm/day)_1', 'tmax(C)_1', 'tmin(C)_1', '00060_Mean_1',
              'prcp(mm/day)_2', 'tmax(C)_2', 'tmin(C)_2', '00060_Mean_2',
              'prcp(mm/day)_3', 'tmax(C)_3', 'tmin(C)_3', '00060_Mean_3',
              'prcp(mm/day)_4', 'tmax(C)_4', 'tmin(C)_4', '00060_Mean_4',
              'prcp(mm/day)_5', 'tmax(C)_5', 'tmin(C)_5', '00060_Mean_5']




forcing = ft.loc[(ft['datetime']>='2010-10-01') &
            (ft['datetime']<'2016-10-01')]


col = ['prcp(mm/day)', 'tmax(C)', 'tmin(C)', '00060_Mean']
mean_f = np.mean(forcing.loc[:,col])
std_f = np.std(forcing.loc[:, col])
mean_Tsw = np.nanmean(forcing.loc[forcing['00010_Mean']>(-10),'00010_Mean'])
std_Tsw = np.nanstd(forcing.loc[forcing['00010_Mean']>(-10),'00010_Mean'])

sites = forcing['site_no'].unique()

print(mean_f)

# inputs_col = ['prcp(mm/day)', 'tmax(C)', 'tmin(C)', '00060_Mean',
#             'prcp(mm/day)_1', 'tmax(C)_1', 'tmin(C)_1', '00060_Mean_1',
#            'prcp(mm/day)_2', 'tmax(C)_2', 'tmin(C)_2', '00060_Mean_2',
#           'prcp(mm/day)_3', 'tmax(C)_3', 'tmin(C)_3', '00060_Mean_3',
#          'prcp(mm/day)_4', 'tmax(C)_4', 'tmin(C)_4', '00060_Mean_4',
#         'prcp(mm/day)_5', 'tmax(C)_5', 'tmin(C)_5', '00060_Mean_5']
inputs_col = ['prcp(mm/day)', 'tmax(C)', 'tmin(C)', '00060_Mean',
              'prcp(mm/day)', 'tmax(C)_1', 'tmin(C)_1', '00060_Mean_1',
              'prcp(mm/day)', 'tmax(C)_2', 'tmin(C)_2', '00060_Mean_2',
              #  'tmax(C)_3', 'tmin(C)_3', '00060_Mean_3',
              # 'tmax(C)_4', 'tmin(C)_4', '00060_Mean_4',
              #  'tmax(C)_5', 'tmin(C)_5', '00060_Mean_5'
              ]

ab = np.zeros(
    (len(site_no), len(inputs_col) + 5))  # saves wieghts, bias, loss intraining, and loss in testing for each basin
pred_test = np.zeros((len(site_no), 731))
pred_train = np.zeros((len(site_no), 1461))
obs_train = np.zeros((len(site_no), 1461))
airT = np.zeros((len(site_no), 1461))
obs_test = np.zeros((len(site_no), 731))

for j, jj in enumerate(sites):

    t_start_train = '2010-10-01'
    t_end_train = '2014-10-01'
    t_start_test = '2014-10-01'
    t_end_test = '2016-10-01'
    inputs_train_T = forcing.loc[(forcing['datetime'] >= t_start_train) &
                                 (forcing['datetime'] < t_end_train) &
                                 (forcing['site_no'] == jj), inputs_col].to_numpy(dtype=np.float32)
    # tmax_train_T = forcing.loc[(forcing['datetime']>=t_start_train) &
    #                        (forcing['datetime']<t_end_train) &
    #                       (forcing['site_no']==jj), inputs_col].to_numpy(dtype=np.float32)
    # tmin_train_T = forcing.loc[(forcing['datetime']>=t_start_train) &
    #                       (forcing['datetime']<t_end_train) &
    #                      (forcing['site_no']==jj), 'tmin(C)'].to_numpy(dtype=np.float32)
    Tsw_train_T = forcing.loc[(forcing['datetime'] >= t_start_train) &
                              (forcing['datetime'] < t_end_train) &
                              (forcing['site_no'] == jj), '00010_Mean'].to_numpy(dtype=np.float32)

    # tmean_train_T = (tmax_train_T+tmin_train_T)/2
    nan_array_train = np.isnan(Tsw_train_T)
    not_nan_array_train = ~ nan_array_train
    Tsw_train = Tsw_train_T[not_nan_array_train]
    inputs_train = inputs_train_T[not_nan_array_train]
    # mask = (inputs>=0)   # eliminating elements with air temperature less than zero
    # Tsw_train = Tsw_train[mask]   # eliminating elements with air temperature less than zero
    # tmean_train = tmean_train[mask]   # eliminating elements with air temperature less than zero

    # normalizing

    # print(Tsw_train[1])
    # print(Tsw_train_T[194])
    # print(tmax_train_T[194])
    # print(tmin_train_T[194])
    # print(tmean_train[1])
    inputs_test_T = forcing.loc[(forcing['datetime'] >= t_start_test) &
                                (forcing['datetime'] < t_end_test) &
                                (forcing['site_no'] == jj), inputs_col].to_numpy(dtype=np.float32)
    Tsw_test_T = forcing.loc[(forcing['datetime'] >= t_start_test) &
                             (forcing['datetime'] < t_end_test) &
                             (forcing['site_no'] == jj), '00010_Mean'].to_numpy(dtype=np.float32)

    # tmax_test_T = forcing.loc[(forcing['datetime']>=t_start_test) &
    #                       (forcing['datetime']<t_end_test) &
    #                      (forcing['site_no']==jj), 'tmax(C)'].to_numpy(dtype=np.float32)
    # tmin_test_T = forcing.loc[(forcing['datetime']>=t_start_test) &
    #                       (forcing['datetime']<t_end_test) &
    #                      (forcing['site_no']==jj), 'tmin(C)'].to_numpy(dtype=np.float32)
    Tsw_test_T = forcing.loc[(forcing['datetime'] >= t_start_test) &
                             (forcing['datetime'] < t_end_test) &
                             (forcing['site_no'] == jj), '00010_Mean'].to_numpy(dtype=np.float32)
    # tmean_test_T = (tmax_test_T+tmin_test_T)/2
    nan_array_test = np.isnan(Tsw_test_T)
    not_nan_array_test = ~ nan_array_test
    Tsw_test = Tsw_test_T[not_nan_array_test]
    inputs_test = inputs_test_T[not_nan_array_test]
    # mask_test = (tmean_test>=0)   # eliminating elements with air temperature less than zero
    # Tsw_test = Tsw_test[mask_test]   # eliminating elements with air temperature less than zero
    # tmean_test = tmean_test[mask_test]   # eliminating elements with air temperature less than zero

    # normalizing
    mean_values_train_T = np.tile(np.tile(mean_f.to_numpy(), int(len(inputs_col) / 4)), (len(inputs_train_T), 1))
    std_values_train_T = np.tile(np.tile(std_f.to_numpy(), int(len(inputs_col) / 4)), (len(inputs_train_T), 1))
    mean_values_train = np.tile(np.tile(mean_f.to_numpy(), int(len(inputs_col) / 4)), (len(inputs_train), 1))
    std_values_train = np.tile(np.tile(std_f.to_numpy(), int(len(inputs_col) / 4)), (len(inputs_train), 1))
    mean_values_test_T = np.tile(np.tile(mean_f.to_numpy(), int(len(inputs_col) / 4)), (len(inputs_test_T), 1))
    std_values_test_T = np.tile(np.tile(std_f.to_numpy(), int(len(inputs_col) / 4)), (len(inputs_test_T), 1))
    mean_values_test = np.tile(np.tile(mean_f.to_numpy(), int(len(inputs_col) / 4)), (len(inputs_test), 1))
    std_values_test = np.tile(np.tile(std_f.to_numpy(), int(len(inputs_col) / 4)), (len(inputs_test), 1))

    inputs_train = (inputs_train - mean_values_train) / std_values_train
    Tsw_train = (Tsw_train - mean_Tsw) / std_Tsw
    #Tsw_train_T = (Tsw_train_T - mean_Tsw) / std_Tsw
    inputs_test = (inputs_test - mean_values_test) / std_values_test
    Tsw_test = (Tsw_test - mean_Tsw) / std_Tsw
    inputs_test_T1 = (inputs_test_T - mean_values_test_T) / std_values_test_T
    inputs_train_T = (inputs_train_T - mean_values_train_T) / std_values_train_T

    x_train = inputs_train.reshape(Tsw_train.shape[0], len(inputs_col))
    x_train_T = inputs_train_T.reshape(Tsw_train_T.shape[0], len(inputs_col))
    y_train = Tsw_train.reshape(Tsw_train.shape[0], 1)
    y_train_T = Tsw_train_T.reshape(Tsw_train_T.shape[0], 1)
    x_test_T = inputs_test_T1.reshape(inputs_test_T1.shape[0], len(inputs_col))
    x_test = inputs_test.reshape(inputs_test.shape[0], len(inputs_col))
    y_test = Tsw_test.reshape(Tsw_test.shape[0], 1)
    y_test_T = Tsw_test_T.reshape(Tsw_test_T.shape[0], 1)
    inputs_train_T = inputs_train_T.reshape(inputs_train_T.shape[0], len(inputs_col))

    # now we initiate model

    inputDim = len(inputs_col) + 2 # takes variable 'x' + simulated value as an input
    outputDim = 1  # takes variable 'y'
    learningRate = 0.001
    epochs = 20

    model = linearRegression(inputDim, outputDim)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    # initiating loss and optimization function

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())  # , lr=learningRate)

    ## training model
    for jj in range(2):
        y_init1 = torch.tensor([[10]], requires_grad=False, dtype=torch.float).cuda()  # torch.randn((x_train_T[k,:].shape[0], 1)).cuda()
        y_init2 = torch.tensor([[11]], requires_grad=False, dtype=torch.float).cuda()
        for k in range(x_train_T.shape[0]):
            # Converting inputs and labels to Variable
            #loss = torch.zeros((1), dtype=torch.float).cuda()



            if torch.cuda.is_available():
                # inputs = Variable(torch.from_numpy(x_train).cuda())
                # labels = Variable(torch.from_numpy(y_train).cuda())

                inputs0 = torch.tensor(x_train_T[0:k+1, :], requires_grad=False, dtype=torch.float).cuda()
                #y_init1 = torch.tensor(20, requires_grad=False, dtype=torch.float).cuda()  #torch.randn((x_train_T[k,:].shape[0], 1)).cuda()
                #y_init2 = torch.tensor(20, requires_grad=False, dtype=torch.float).cuda()  #torch.randn((x_train_T[k,:].shape[0], 1)).cuda()

                #inputs = torch.cat((inputs0, A, B), 1)
                inputs = torch.cat((inputs0, y_init2, y_init1), 1)
                labels = torch.tensor(y_train_T[0:k+1,:], requires_grad=False).cuda()
                # y_sim_past = model(inputs.float()).cuda()
                # y_sim_past2 = B
               # else:
               #      inputs = torch.tensor(x_train_T[k,:], requires_grad=True, dtype=torch.float).cuda()
               #      #inputs = torch.cat((inputs, y_sim_past2, y_sim_past), 0)
               #      inputs = torch.cat((inputs, y_sim_past2.view(1), y_sim_past.view(1)), 0)
               #      labels = torch.tensor(y_train_T[k,:], requires_grad=False).cuda()
               #      A = model(inputs.float()).cuda()
               #      y_sim_past2 = y_sim_past
               #      y_sim_past = A
                    # inputs = torch.tensor(x_train, requires_grad=True, dtype=torch.float).cuda()
                    # A = model(inputs.float()).cuda()
                    # A1 = A.data.cpu().numpy()
                    # #AAA = np.where(y_train_T == y_train_T, y_train_T, A1)
                    # #AAA = np.where(y_train_T[k,:] == y_train_T[k,:], y_train_T[k,:], A1)
                    # #A = torch.tensor(AAA, requires_grad=False, dtype=torch.float).cuda()
                    # AA = torch.roll(A, shifts=1)
                    # B = AA.detach().clone()   #torch.copy(torch.roll(A, shifts=1), requires_grad=False).cuda()
                    #inputs[:,-1] = B.squeeze()
                    # inputs[-1] = B.squeeze()
                    # AA = torch.roll(A, shifts=2)
                    # B = AA.detach().clone()  # torch.copy(torch.roll(A, shifts=1), requires_grad=False).cuda()
                    # inputs[-2] = B.squeeze()
                    #inputs[:, -2] = B.squeeze()
                   # labels = torch.tensor(y_train, requires_grad=False).cuda()

            else:
                # inputs = Variable(torch.from_numpy(x_train))
                # labels = Variable(torch.from_numpy(y_train))
                inputs = torch.tensor(x_train_T, requires_grad=False)
                inputs = torch.cat((inputs.float(), model(inputs.float())), 1)
                labels = torch.tensor(y_train_T, requires_grad=False)

            if (labels[-1, :] != labels[-1, :]):
                e = 1
            else:
                e = epochs
            for epoch in range(e):
                # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
                optimizer.zero_grad()

            # get output from the model, given the inputs
                outputs = model(inputs.float())

            # get loss for the predicted output
                ysim = outputs[not_nan_array_train[0:k + 1]]
                yobs = labels[not_nan_array_train[0:k + 1]]
                if (ysim.nelement==0):
                    loss = torch.tensor([0.1], requires_grad=True, dtype=torch.float).cuda()
                else:

            #     loss = loss + criterion(outputs, labels)
                    loss = criterion(ysim, yobs)
                #optimizer.zero_grad()
            ##  print(loss)
            # get gradients w.r.t to parameters

                loss.backward()
            #for param in model.parameters():
             #   print(param)
            # update parameters
                optimizer.step()
            y_init2 = torch.cat((y_init2, torch.tensor([[y_init1[-1,:]]]).cuda()), 0)

            y_init1 = torch.cat((y_init1, torch.tensor([[outputs[-1,:]]]).cuda()), 0)
       # print('epoch {}, loss {}'.format(epoch, loss.item()))
    plt.scatter(outputs[not_nan_array_train].detach().cpu().numpy(), labels[not_nan_array_train].detach().cpu().numpy())
    plt.show()
    ### testing model
    predicted = np.zeros((x_test_T.shape[0], 1))
    with torch.no_grad():  # we don't need gradients in the testing phase
        # if y_train_T[-1,0] == y_train_T[-1,0]:
        #     y_sim_past = y_train_T[-1,0]
        # else:
        #     y_sim_past = outputs[-1, 0].item()  # for the first day of testing period, last day of prediction in training set is used
        #
        # if y_train_T[-2,0] == y_train_T[-2,0]:
        #     y_sim_past2 = y_train_T[-2,0]
        # else:
        #     y_sim_past2 = outputs[-2, 0].item()

        y_sim_past2 = y_init2[-1,:].cpu()
        y_sim_past = y_init1[-1,:].cpu()
        for i in range(x_test_T.shape[0]):
            A = x_test_T[i,:]
            A = np.append(A, y_sim_past2)
            A = np.append(A, y_sim_past)
            if torch.cuda.is_available():
                in_test = torch.tensor(A, dtype=torch.float).cuda()
                predicted[i, 0] = model(Variable(in_test).float()).cpu().data.numpy()
                y_sim_past2 = y_sim_past
                # if y_test_T[i, 0] == y_test_T[i, 0]:
                #     y_sim_past =y_test_T[i, 0]
                # else:
                #     y_sim_past = predicted[i, 0].item()  # today's prediction used for tomorrow's prediction
                y_sim_past = predicted[i, 0].item()
                #predicted_T = model(Variable(torch.from_numpy(x_test_T).cuda()).float()).cpu().data.numpy()
               # p_train = model(Variable(torch.from_numpy(inputs_train_T).cuda()).float()).cpu().data.numpy()
            else:
                predicted = model(Variable(torch.from_numpy(x_test)).float()).data.numpy()
                predicted_T = model(Variable(torch.from_numpy(x_test_T)).float()).data.numpy()
                p_train = model(Variable(torch.from_numpy(inputs_train_T)).float()).data.numpy()
        # print(predicted)

    # denormalizeing
    x_test = (x_test * std_values_test) + mean_values_test
    x_test_T = (x_test_T * std_values_test_T) + mean_values_test_T
    y_test = (y_test * std_Tsw) + mean_Tsw
    #predicted = (predicted * std_Tsw) + mean_Tsw
    #predicted_T = (predicted_T * std_Tsw) + mean_Tsw
    #p_train = (p_train * std_Tsw) + mean_Tsw
    #x_train = (x_train * std_values_train) + mean_values_train
   # inputs_train = (inputs_train * std_values_train) + mean_values_train
   # Tsw_train = (Tsw_train * std_Tsw) + mean_Tsw
   # inputs_train_T = (inputs_train_T * mean_values_train_T) + mean_values_train_T

    predicted[predicted < 0] = 0
    plt.clf()
    # plt.plot(x_test, y_test, 'go', label='True data', alpha=0.25)
    plt.scatter(y_test_T.squeeze(), predicted)  # , '--', label='Predictions', alpha=0.25)
    # plt.legend(loc='best')
    # plt.xlim(0,34)
    plt.show()

    # plt.clf()
    # plt.plot(t_train, Tsw_train, 'go', label='True data', alpha=0.25)
    # plt.plot(tmean_train_T, p_train, '--', label='Predictions', alpha=0.25)
    # plt.legend(loc='best')
    # plt.xlim(0,34)
    #plt.show()

    ##  saving a & b in y=ax+b

    for param, i in zip(model.parameters(), range(2)):
        if i == 0:
            ab[j, 0:len(inputs_col)+1] = param.data[0][0:len(inputs_col)+1].detach().cpu()
        else:  # factor
            ab[j, len(inputs_col)+1] = param.data[0]  # factor

    predicted = torch.tensor(predicted, requires_grad=False, dtype=torch.float)
    labels_test = torch.tensor(y_test_T, requires_grad=False)
    ysim_test = predicted[not_nan_array_test]
    yobs_test = labels_test[not_nan_array_test]
    loss_test = criterion(ysim_test, yobs_test)
    A = predicted.reshape(731)
    #B = p_train.reshape(1461)
    # C = inputs_train_T.reshape(1461)
    pred_test[j, :] = A
    ab[j, len(inputs_col) + 2] = loss_test
    ab[j, len(inputs_col) + 3] = loss
    # airT[j,:] = C
    obs_train[j, :] = Tsw_train_T
    #pred_train[j, :] = B
    obs_test[j, :] = Tsw_test_T
    print(j)




ab1 = np.expand_dims(ab, axis=2)
#pred_train1 = np.expand_dims(pred_train, axis=2)
pred_test1 = np.expand_dims(pred_test, axis=2)
#obs_train1 = np.expand_dims(obs_train, axis=2)
obs_test1 = np.expand_dims(obs_test, axis=2)


### saving the results. Please change the paths
path_ab = os.path.join('G:\\Farshid\\CONUS_Temp\\Example3\\TempDemo\\FirstRun\\After_eliminating_SWE\\ERL_paper\\AutoReg\\ab.npy')
np.save(path_ab, ab1)

#path_pred_train = os.path.join('G:\\Farshid\\CONUS_Temp\\Example3\\TempDemo\\FirstRun\\After_eliminating_SWE\\ERL_paper\\AutoReg\\pred_train.npy')
#np.save(path_pred_train,pred_train1)
path_pred_test = os.path.join('G:\\Farshid\\CONUS_Temp\\Example3\\TempDemo\\FirstRun\\After_eliminating_SWE\\ERL_paper\\AutoReg\\pred_test.npy')
np.save(path_pred_test, pred_test1)

#path_obs_train = os.path.join('G:\\Farshid\\CONUS_Temp\\Example3\\TempDemo\\FirstRun\\After_eliminating_SWE\\ERL_paper\\AutoReg\\obs_train.npy')
#np.save(path_obs_train,obs_train1)
path_obs_test = os.path.join('G:\\Farshid\\CONUS_Temp\\Example3\\TempDemo\\FirstRun\\After_eliminating_SWE\\ERL_paper\\AutoReg\\obs_test.npy')
np.save(path_obs_test,obs_test1)

#path_airT = os.path.join('G:\\Farshid\\CONUS_Temp\\Example3\\TempDemo\\FirstRun\\After_eliminating_SWE\\ERL_paper\\AutoReg\\airT_train.npy')
#np.save(path_airT,airT)

print('END')

