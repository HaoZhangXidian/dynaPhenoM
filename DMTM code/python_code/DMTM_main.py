"""
===========================================
DMTM
===========================================

"""

# Author: Hao
# License: Apache License Version 2.0


import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
from pydpm.model import PGDS_multi_sample


data = sio.loadmat('../data/diagnosis.mat')
train_data_diagnosis = coo_matrix((np.squeeze(data['val']), (np.squeeze(data['row']), np.squeeze(data['col']))), shape=(data['num_row'][0,0], data['num_col'][0,0])).toarray()
train_data_diagnosis[train_data_diagnosis>1] = 1
train_data_diagnosis = np.array(train_data_diagnosis, order='C')
Patient_label = data['Patient_label']

data = sio.loadmat('../data/drug.mat')
train_data_drug = coo_matrix((np.squeeze(data['val']), (np.squeeze(data['row']), np.squeeze(data['col']))), shape=(data['num_row'][0,0], data['num_col'][0,0])).toarray()
train_data_drug[train_data_drug>1] = 1
train_data_drug = np.array(train_data_drug, order='C')

data = sio.loadmat('../data/procedure.mat')
train_data_procedure = coo_matrix((np.squeeze(data['val']), (np.squeeze(data['row']), np.squeeze(data['col']))), shape=(data['num_row'][0,0], data['num_col'][0,0])).toarray()
train_data_procedure[train_data_procedure>1] = 1
train_data_procedure = np.array(train_data_procedure, order='C')

ii_diag, jj_diag = np.nonzero(train_data_diagnosis)
iijj_diag = np.nonzero(train_data_diagnosis.flatten())[0]

ii_drug, jj_drug = np.nonzero(train_data_drug)
iijj_drug = np.nonzero(train_data_drug.flatten())[0]

ii_procedure, jj_procedure = np.nonzero(train_data_procedure)
iijj_procedure = np.nonzero(train_data_procedure.flatten())[0]

K=30
Phi_diag = np.random.rand(train_data_diagnosis.shape[0], K)
Phi_diag = Phi_diag / np.sum(Phi_diag, axis=0)

Phi_procedure = np.random.rand(train_data_procedure.shape[0], K)
Phi_procedure = Phi_procedure / np.sum(Phi_procedure, axis=0)

Phi_drug = np.random.rand(train_data_drug.shape[0], K)
Phi_drug = Phi_drug / np.sum(Phi_drug, axis=0)

model=PGDS_multi_sample(K, 'cpu')
model.initial(train_data_diagnosis, train_data_drug, train_data_procedure,
              Patient_label, Phi_diag, Phi_drug, Phi_procedure,
              ii_diag, jj_diag, iijj_diag,
              ii_drug, jj_drug, iijj_drug,
              ii_procedure, jj_procedure, iijj_procedure)
model.train(iter_all=500)


