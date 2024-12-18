import numpy as np
from numpy.lib.type_check import real
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('/media/shawey/SSD8T/UNC_FC_Traverse/ROIs360/ROI360_FC_IDAgeGroup_AP')
import math
import os
import pandas as pd
import copy

indexs = [1] #9,8,6,4,3,1
#MLPredictResults/LinearSVRWithNystroemApproximateSVR
#MLPredictResults/BayesianRidge

MAE_re = []
Corr_re = []

for index in indexs:
  FC_recon_path = '/media/shawey/SSD8T/UNC_FC_Traverse/ROIs360Code/MultiGPUs_ICM_AgeMerging_FCDis_30Days_AP_addML/ExpResults/Exp' + str(index) +'_512_4_1'
  files = os.listdir(FC_recon_path)
  files_co = copy.deepcopy(files)
  temp = []

  ID_record = []
  for filename in files:
    if(filename.split('.')[0] != 'pred_ori_corr' and filename.split('.')[-1] == 'txt'):

      realID_recon = filename.split('.')[0].split('_')[0]
      #day_recon = filename.split('.')[0].split('_')[1]
      #print('ID_recon is {0}, day_recon is {1}'.format(realID_recon,day_recon))
      #FC_recon = np.loadtxt(FC_recon_path + '/' + filename)
      
      if (realID_recon in ID_record):
        continue
      ID_record.append(realID_recon)
      print('realID_recon ',realID_recon)
      for filename_co in files_co:
          realID_recon_co = filename_co.split('.')[0].split('_')[0]
          if (realID_recon_co == realID_recon):
            print(filename_co)
            day_recon_co = filename_co.split('.')[0].split('_')[1]
            print('ID_recon is {0}, day_recon is {1}'.format(realID_recon_co,day_recon_co))
            FC_recon = np.loadtxt(FC_recon_path + '/' + filename_co)

            #find the original file
            file_path = '/media/shawey/SSD8T/UNC_FC_Traverse/ROIs360/ROI360_FC_IDAgeGroup_AP'
            for root, dirs, files in os.walk(file_path):
                for file_name in files:
                    #print(file_name)
                    if(file_name.split('.')[1] == 'txt'):
                      realID = file_name.split('.')[0].split('_')[0]
                      days = file_name.split('.')[0].split('_')[1]
                      if(realID == realID_recon and int(day_recon_co) == int( np.floor( int(days)/30)) ):          
                        print('real ID is {0}, days is {1}'.format(realID, days))
                        FC = np.loadtxt(file_path+'/'+file_name)
                        break

            if (temp):
              print('FC_recon prev and FC_recon dis is', np.mean(abs(temp[-1]-FC_recon)))
              print('FC_recon prev and FC_recon corr is', np.corrcoef(temp[-1].flatten(), FC_recon.flatten())[0,1])
            temp.append(FC_recon)
              
            FC_recon_fla = FC_recon.flatten()
            FC_fla = FC.flatten()
            print('FC_recon min and max is {0} {1}, FC {2} {3}'.format(min(FC_recon_fla), max(FC_recon_fla), min(FC_fla), max(FC_fla)))
              
            FC_FC_recon_dis = abs(FC-FC_recon).flatten()
            print('FC and FC_recon dis is', np.mean(abs(FC-FC_recon)))
            print('FC and FC_recon corr is', np.corrcoef(FC.flatten(), FC_recon.flatten())[0,1])
            if(np.corrcoef(FC.flatten(), FC_recon.flatten())[0,1] > 0.5):
              MAE_re.append(np.mean(abs(FC-FC_recon)))
              Corr_re.append(np.corrcoef(FC.flatten(), FC_recon.flatten())[0,1])         
   

print('Group MAE mean and var is {0}, {1}'.format( np.mean(np.array(MAE_re)), np.var(np.array(MAE_re)) ))
print('Group Corr mean and var is {0}, {1}'.format( np.mean(np.array(Corr_re)), np.var(np.array(Corr_re)) ))

