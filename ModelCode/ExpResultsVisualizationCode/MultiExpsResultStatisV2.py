#ADD geo distance
#ADD RELATIVE ERROR
#show FC_recon variation 
import numpy as np
from numpy.lib.type_check import real
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('/media/shawey/SSD8T/UNC_FC_Traverse/ROI70/ROI70_FC_AP')
import math
import os
import pandas as pd
import copy
from distance_FC import *
from math import log
from sklearn.cluster import KMeans

def NVI_NMI(X, Y):  #Normalized variation of information (VIn) and mutual information (MIn)
    #X or Y contains the all elements, because X and Y are the partition of the whole set
    n = float(sum([len(x) for x in X]))
    sigma = 0.0
    for x in X:
        p = len(x) / n
        for y in Y:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (log(r / p, 2) + log(r / q, 2))
    NVI = abs(sigma/ log(n,2))
    part2 = 0.0
    part1 = 0.0
    for x in X:
        p = len(x) / n
        part1 += p * log(p, 2)
    for y in Y:
        q = len(y) / n
        part2 +=  q * log(q, 2)
    NMI = 1 + NVI * log(n,2) / (part1 + part2)
    return NVI, NMI

GAP = 100
indexs = [1,2,3,4] #
#20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,18
#MLPredictResults/LinearSVRWithNystroemApproximateSVR
#MLPredictResults/BayesianRidge

MAE_re = []
Corr_re = []
geo_dist = []

intra_FC_variation = []
inter_FC_variation = []
NMI = []
NVI = []
for index in indexs:
  FC_recon_path = '/media/shawey/SSD8T/UNC_FC_Traverse/ROI70/MultiGPUs_ICM_AgeMerging_100Days_AP_MAELoss_NoLatentFeatureAdd2DecoderReLU2/ExpResults/Exp' + str(index) +'_unit_count_1500ReLU2_pccLoss'#+'_unit_count_256ReLU'

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

      intra_flag = 0
      for filename_co in files_co:
          realID_recon_co = filename_co.split('.')[0].split('_')[0]
          if (realID_recon_co == realID_recon):
            intra_flag += 1
            print(filename_co)
            day_recon_co = filename_co.split('.')[0].split('_')[1]
            print('ID_recon is {0}, day_recon is {1}'.format(realID_recon_co,day_recon_co))
            FC_recon = np.loadtxt(FC_recon_path + '/' + filename_co)

            #find the original file
            file_path = '/media/shawey/SSD8T/UNC_FC_Traverse/ROI70/ROI70_FC_AP'
            for root, dirs, files in os.walk(file_path):
                for file_name in files:
                    #print(file_name)
                    if(file_name.split('.')[1] == 'txt'):
                      realID = file_name.split('.')[0].split('_')[0]
                      days = file_name.split('.')[0].split('_')[1]
                      if(realID == realID_recon and int(day_recon_co) == int( np.floor( int(days)/GAP)) ):          
                        print('real ID is {0}, days is {1}'.format(realID, days))
                        FC = np.loadtxt(file_path+'/'+file_name)
                        break

            if (temp):
              print('FC_recon prev and FC_recon dis is', np.mean(abs(temp[-1]-FC_recon)))
              print('FC_recon prev and FC_recon corr is', np.corrcoef(temp[-1].flatten(), FC_recon.flatten())[0,1])            
  
            if (temp and intra_flag > 1 ):              
              intra_FC_variation.append(np.corrcoef(temp[-1].flatten(), FC_recon.flatten())[0,1])
            if (temp and intra_flag == 1 ):              
              inter_FC_variation.append(np.corrcoef(temp[-1].flatten(), FC_recon.flatten())[0,1])
            
            temp.append(FC_recon)
              
            FC_recon_fla = FC_recon.flatten()
            FC = abs(FC)
            FC_fla = FC.flatten()
            print('FC_recon min and max is {0} {1}, FC {2} {3}'.format(min(FC_recon_fla), max(FC_recon_fla), min(FC_fla), max(FC_fla)))
              
            FC_FC_recon_dis = abs(FC-FC_recon).flatten()
            print('FC and FC_recon dis is', np.mean(abs(FC-FC_recon)))
            print('FC and FC_recon corr is', np.corrcoef(FC.flatten(), FC_recon.flatten())[0,1])
            if(np.corrcoef(FC.flatten(), FC_recon.flatten())[0,1] > 0.0):
              MAE_re.append(np.mean(abs(FC-FC_recon)))
              Corr_re.append(np.corrcoef(FC.flatten(), FC_recon.flatten())[0,1])   
              dist = distance_FC(FC_recon, FC)      
              geo_dist.append( dist.geodesic() )
              print('FC and FC_recon geo_dist is', dist.geodesic())
 

              num_cluster = 5
              km = KMeans(n_clusters = num_cluster)
              km.fit(FC_fla.reshape(-1,1))
#print(km.cluster_centers_)
              x_labels = km.labels_
              x_clusters = []
              for i_index in range(num_cluster):
                x_clusters.append( [i for i in range(len(x_labels)) if x_labels[i] == i_index] )
              km.fit(FC_recon_fla.reshape(-1,1))
#print(km.cluster_centers_)
              y_labels = km.labels_
              y_clusters = []
              for i_index in range(num_cluster):
                y_clusters.append( [i for i in range(len(y_labels)) if y_labels[i] == i_index] )

              NVI_temp, NMI_temp = NVI_NMI(x_clusters, y_clusters)
              NVI.append(NVI_temp)
              NMI.append(NMI_temp)
              print('len(NVI) {0}, NVI {1}, NMI {2}'.format(len(NVI), NVI_temp, NMI_temp) )


print('Group MAE mean and var is {0}, {1}'.format( np.mean(np.array(MAE_re)), np.std(np.array(MAE_re)) ))
print('Group Corr mean and var is {0}, {1}'.format( np.mean(np.array(Corr_re)), np.std(np.array(Corr_re)) ))
print('Group geo_dist mean and var is {0}, {1}'.format( np.mean(np.array(geo_dist)), np.std(np.array(geo_dist)) ))


inter_FC_variation = 1 - abs(np.array(inter_FC_variation))
intra_FC_variation = 1 - abs(np.array(intra_FC_variation))
print(np.mean(inter_FC_variation), np.std(inter_FC_variation))
print(np.mean(intra_FC_variation), np.std(intra_FC_variation))

print('Group NVI mean and std {0}, {1}'.format(np.mean(NVI), np.std(NVI)))
print('Group NMI mean and std {0}, {1}'.format(np.mean(NMI), np.std(NMI)))

'''
ax_x = []
for i in range(len(inter_FC_variation)):
   ax_x.append(i)
plt.subplot(121)
plt.scatter(ax_x, inter_FC_variation, c = 'b', marker = 's')
plt.title("inter_FC Variation")
plt.ylabel('Variation(Percentage)')
plt.xlabel('Data Samples')
ax_x = []
for i in range(len(intra_FC_variation)):
   ax_x.append(i)
plt.subplot(122)
plt.scatter(ax_x, intra_FC_variation, c = 'r', marker = 's')
plt.title("intra_FC Variation")
plt.ylabel('Variation(Percentage)')
plt.xlabel('Data Samples')

plt.show()
'''

