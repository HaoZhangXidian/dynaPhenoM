"""
===========================================
DMTM
===========================================

"""

eps = 2.2204e-16
import scipy.io as sio
import time
import numpy as np
from pydpm.utils.Metric import *
from pydpm.utils import Model_Sampler_CPU
import scipy
from tqdm import tqdm
from scipy.sparse import coo_matrix

class PGDS_multi_sample(object):

    def __init__(self, K, device='cpu'):

        self.K = K
        self.L = 1
        if device == 'cpu':
            self.device = 'cpu'
            self.Multrnd_Matrix = Model_Sampler_CPU.Multrnd_Matrix
            self.Multrnd_Matrix_CPU = Model_Sampler_CPU.Multrnd_Matrix
            self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix

        elif device == 'gpu':
            self.device = 'gpu'
            from pydpm.utils import Model_Sampler_GPU
            self.Multrnd_Matrix = Model_Sampler_GPU.Multrnd_Matrix_GPU
            self.Multrnd_Matrix_CPU = Model_Sampler_CPU.Multrnd_Matrix
            self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix

    def initial(self, X_train_diag, X_train_drug, X_train_procedure,
                Patient_label, Phi_diag, Phi_drug, Phi_procedure,
                ii_diag, jj_diag, iijj_diag,
                ii_drug, jj_drug, iijj_drug,
                ii_procedure, jj_procedure, iijj_procedure,
                burn_in = 200, step=5):

        self.data_diag = X_train_diag
        self.data_drug = X_train_drug
        self.data_procedure = X_train_procedure
        self.Patient_label = Patient_label

        K = self.K
        self.V_diag, self.N = self.data_diag.shape
        self.V_drug, _ = self.data_drug.shape
        self.V_procedure, _ = self.data_procedure.shape

        L = self.L
        self.num_patient = int(Patient_label.max())+1

        self.ii_diag = ii_diag
        self.jj_diag = jj_diag
        self.iijj_diag = iijj_diag

        self.ii_drug = ii_drug
        self.jj_drug = jj_drug
        self.iijj_drug = iijj_drug

        self.ii_procedure = ii_procedure
        self.jj_procedure = jj_procedure
        self.iijj_procedure = iijj_procedure


        #setting
        self.Setting={}
        self.Setting['Stationary'] = 0
        self.Setting['Burn_in'] = burn_in
        self.Setting['Step'] = step

        ## self.Supara
        self.Supara = {}
        self.Supara['tao0'] = 1
        self.Supara['gamma0'] = 100
        self.Supara['eta0'] = 0.01
        self.Supara['epilson0'] = 0.1

        ## self.Para
        self.Para = {}

        self.Para['Phi_diag'] = Phi_diag
        self.Para['Phi_drug'] = Phi_drug
        self.Para['Phi_procedure'] = Phi_procedure
        self.Para['Pi'] = np.eye(K)

        self.Para['Xi'] = 1
        self.Para['V'] = np.ones((K, 1))
        self.Para['h'] = np.zeros((K, K))
        self.Para['beta'] = 1
        self.Para['q'] = np.ones((K, 1))
        self.Para['n'] = np.ones((K, 1))
        self.Para['rou'] = np.zeros((K, 1))

        self.Para['Theta'] = np.ones((K, self.N)) / K
        self.Para['delta_diag'] = np.ones((self.N, 1))
        self.Para['delta_drug'] = np.ones((self.N, 1))
        self.Para['delta_procedure'] = np.ones((self.N, 1))
        self.Para['L_dotkt_all'] = []
        #############################################################################

        self.collection = {}
        self.collection['delta_diag_sum'] = 0
        self.collection['delta_drug_sum'] = 0
        self.collection['delta_procedure_sum'] = 0
        self.collection['Phi_diag_sum'] = 0
        self.collection['Phi_drug_sum'] = 0
        self.collection['Phi_procedure_sum'] = 0
        self.collection['Theta_sum'] = 0
        self.collection['Pi_sum'] = 0
        self.collection['flag'] = 0



    def train(self, iter_all=200, train_flag=True):

        self.Likelihood = []
        for i in range(iter_all):
            begin = time.time()

            Rate = np.dot(self.Para['Phi_diag'], self.Para['delta_diag'].transpose()*self.Para['Theta'])
            M = truncated_Poisson_sample(Rate.flatten()[self.iijj_diag])
            data = coo_matrix((np.squeeze(M), (np.squeeze(self.ii_diag), np.squeeze(self.jj_diag))), shape=(self.V_diag, self.N)).toarray()
            X_train_diag = np.array(data, dtype=np.double, order='C')

            Rate = np.dot(self.Para['Phi_drug'], self.Para['delta_drug'].transpose() * self.Para['Theta'])
            M = truncated_Poisson_sample(Rate.flatten()[self.iijj_drug])
            data = coo_matrix((np.squeeze(M), (np.squeeze(self.ii_drug), np.squeeze(self.jj_drug))), shape=(self.V_drug, self.N)).toarray()
            X_train_drug = np.array(data, dtype=np.double, order='C')

            Rate = np.dot(self.Para['Phi_procedure'], self.Para['delta_procedure'].transpose() * self.Para['Theta'])
            M = truncated_Poisson_sample(Rate.flatten()[self.iijj_procedure])
            data = coo_matrix((np.squeeze(M), (np.squeeze(self.ii_procedure), np.squeeze(self.jj_procedure))), shape=(self.V_procedure, self.N)).toarray()
            X_train_procedure = np.array(data, dtype=np.double, order='C')

            [A_KN_diag, A_VK_diag] = self.Multrnd_Matrix(X_train_diag, self.Para['Phi_diag'], self.Para['delta_diag'].transpose()*self.Para['Theta'])
            [A_KN_drug, A_VK_drug] = self.Multrnd_Matrix(X_train_drug, self.Para['Phi_drug'], self.Para['delta_drug'].transpose() * self.Para['Theta'])
            [A_KN_procedure, A_VK_procedure] = self.Multrnd_Matrix(X_train_procedure, self.Para['Phi_procedure'], self.Para['delta_procedure'].transpose() * self.Para['Theta'])

            L_KK = np.zeros((self.K, self.K))
            L_kdott_for_V = np.zeros((self.K, self.num_patient))

            for n in range(self.num_patient):
                index  = np.squeeze(self.Patient_label==n)
                x_patient_diag = X_train_diag[:, index]
                x_patient_drug = X_train_drug[:, index]
                x_patient_procedure = X_train_procedure[:, index]

                patient_T = x_patient_diag.shape[1]
                L_kdott = np.zeros((self.K, patient_T))

                Theta_patient = self.Para['Theta'][:, index]
                A_KN_patient = A_KN_diag[:, index] + A_KN_drug[:, index] + A_KN_procedure[:, index]

                if i==0:
                    self.Para['L_dotkt_all'].append(np.zeros((self.K, patient_T+1)))

                for t in range(patient_T-1, 0, -1):
                    tmp1 = A_KN_patient[:, t] + self.Para['L_dotkt_all'][n][:, t+1]
                    tmp1 = tmp1[:,np.newaxis]
                    tmp2 = self.Supara['tao0'] * np.dot(self.Para['Pi'], Theta_patient[:,t-1])
                    tmp2 = tmp2[:,np.newaxis]
                    L_kdott[:, t:t+1] = self.Crt_Matrix(tmp1.astype('double'), tmp2)

                    [self.Para['L_dotkt_all'][n][:, t:t+1], tmp] = self.Multrnd_Matrix_CPU(
                        np.array(L_kdott[:, t], dtype=np.double, order='C')[:, np.newaxis], self.Para['Pi'],
                        np.array(Theta_patient[:,t-1], dtype=np.double, order='C')[:, np.newaxis])

                    L_KK += tmp

            # Sample Phi
            if train_flag:
                self.Para['Phi_diag'] = Model_Sampler_CPU.Sample_Pi(A_VK_diag, self.Supara['eta0'])
                self.Para['Phi_drug'] = Model_Sampler_CPU.Sample_Pi(A_VK_drug, self.Supara['eta0'])
                self.Para['Phi_procedure'] = Model_Sampler_CPU.Sample_Pi(A_VK_procedure, self.Supara['eta0'])

            # Sample Pi
            Piprior = np.dot(self.Para['V'], np.transpose(self.Para['V']))
            Piprior[np.arange(Piprior.shape[0]), np.arange(Piprior.shape[1])] = 0

            Piprior = Piprior + np.diag(np.reshape(self.Para['Xi'] * self.Para['V'], [self.Para['V'].shape[0]]))
            self.Para['Pi'] = Model_Sampler_CPU.Sample_Pi(L_KK, Piprior)

            ###################################################### Calculate for local
            L_dotkt_all_sum = 0
            Zeta_all_for_v = 0

            # Sample Theta
            for n in range(self.num_patient):
                index = np.squeeze(self.Patient_label == n)
                x_patient_diag = X_train_diag[:, index]
                patient_T = x_patient_diag.shape[1]
                Theta_patient = self.Para['Theta'][:, index]

                A_KN_patient = A_KN_diag[:, index] + A_KN_drug[:, index] + A_KN_procedure[:, index]
                delta_patient = self.Para['delta_diag'][index] + self.Para['delta_drug'][index] + self.Para['delta_procedure'][index]
                Zeta_patient = np.zeros((patient_T + 1, 1))

                if self.Setting['Stationary'] == 0:
                    for t in range(patient_T, 0, -1):
                        Zeta_patient[t-1] = np.real(np.log(1 + delta_patient[t-1] / self.Supara['tao0'] + Zeta_patient[t]))

                Zeta_all_for_v = Zeta_all_for_v + Zeta_patient[0]
                L_dotkt_all_sum = L_dotkt_all_sum + np.sum(self.Para['L_dotkt_all'][n], 1)

                for t in range(patient_T):
                    if t == 0:
                        shape = A_KN_patient[:, t]+ self.Para['L_dotkt_all'][n][:, t + 1]+ self.Supara['tao0'] * np.squeeze(self.Para['V'])
                    else:
                        shape = A_KN_patient[:, t]+ self.Para['L_dotkt_all'][n][:, t + 1]+ self.Supara['tao0'] * np.dot(self.Para['Pi'], Theta_patient[:, t-1])
                    scale = delta_patient[t] + self.Supara['tao0'] + self.Supara['tao0'] * Zeta_patient[t + 1]
                    Theta_patient[:, t] = np.random.gamma(shape) / scale

                self.Para['Theta'][:, index] = Theta_patient

            # Sample Beta
            shape = self.Supara['epilson0'] + self.Supara['gamma0']
            scale = self.Supara['epilson0'] + np.sum(self.Para['V'])
            self.Para['beta'] = np.random.gamma(shape) / scale

            # Sample q checked
            a = L_dotkt_all_sum
            #a[a == 0] = 1e-10
            b = self.Para['V'] * (self.Para['Xi'] + np.repeat(np.sum(self.Para['V']), self.K, axis=0).reshape([self.K, 1]) - self.Para['V'])
            #b[b == 0] = 1e-10  #
            self.Para['q'] = np.maximum(np.random.beta(b.squeeze(), a), 2.2251e-308)
            # Sample h checked
            for k1 in range(self.K):
                for k2 in range(self.K):
                    self.Para['h'][k1:k1 + 1, k2:k2 + 1] = Model_Sampler_CPU.Crt_Matrix(L_KK[k1:k1 + 1, k2:k2 + 1], Piprior[k1:k1 + 1, k2:k2 + 1])
            # Sample Xi checked
            shape = self.Supara['gamma0'] / self.K + np.trace(self.Para['h'])
            scale = self.Para['beta'] - np.dot(np.transpose(self.Para['V']), np.log(self.Para['q']))
            self.Para['Xi'] = np.transpose(np.random.gamma(shape) / scale)

            # Sample V # check
            for n in range(self.num_patient):
                index = np.squeeze(self.Patient_label == n)
                A_KN_patient = A_KN_diag[:, index] + A_KN_drug[:, index] + A_KN_procedure[:, index]
                for k in range(self.K):
                    L_kdott_for_V[k,n] = Model_Sampler_CPU.Crt_Matrix(np.reshape(A_KN_patient[k,0] +  self.Para['L_dotkt_all'][n][k,1], (1,1)),
                                                                      np.reshape(self.Supara['tao0'] * self.Para['V'][k], (1, 1)))
            for k in range(self.K):
                self.Para['n'][k] = np.sum(self.Para['h'][k, :] + np.transpose(self.Para['h'][:, k])) - self.Para['h'][k, k] + L_kdott_for_V[k,:].sum()
                self.Para['rou'][k] = - np.log(self.Para['q'][k]) * (self.Para['Xi'] + np.sum(self.Para['V']) - self.Para['V'][k]) \
                                      - np.dot(np.transpose(np.log(self.Para['q'])), self.Para['V']) \
                                      + np.log(self.Para['q'][k]) * self.Para['V'][k] + Zeta_all_for_v

            shape_top = self.Supara['gamma0'] / self.K + self.Para['n']
            scale_top = self.Para['beta'] + self.Para['rou']
            self.Para['V'] = np.random.gamma(shape_top) / scale_top

            if self.Setting['Stationary'] == 0:
                for n in range(self.num_patient):
                    index = np.squeeze(self.Patient_label == n)
                    x_patient_diag = X_train_diag[:, index]
                    x_patient_drug = X_train_drug[:, index]
                    x_patient_procedure = X_train_procedure[:, index]
                    Theta_patient = self.Para['Theta'][:, index]

                    shape = self.Supara['epilson0'] + x_patient_diag.sum(0)
                    scale = self.Supara['epilson0'] + Theta_patient.sum(0)
                    delta_patient = np.random.gamma(shape) / scale
                    self.Para['delta_drug'][index, 0] = delta_patient

                    shape = self.Supara['epilson0'] + x_patient_drug.sum(0)
                    scale = self.Supara['epilson0'] + Theta_patient.sum(0)
                    delta_patient = np.random.gamma(shape) / scale
                    self.Para['delta_drug'][index, 0] = delta_patient

                    shape = self.Supara['epilson0'] + x_patient_procedure.sum(0)
                    scale = self.Supara['epilson0'] + Theta_patient.sum(0)
                    delta_patient = np.random.gamma(shape) / scale
                    self.Para['delta_procedure'][index, 0] = delta_patient

            # Likelihood

            Lambda = np.dot(self.Para['Phi_diag'], self.Para['delta_diag'].transpose()*self.Para['Theta'])
            P = 1-np.exp(-Lambda)
            P[P==0] = eps
            P[P==1] = 1-eps
            like_diag = np.sum(np.sum(self.data_diag*np.log(P) + (1-self.data_diag)*np.log(1-P))) / self.V_diag

            Lambda = np.dot(self.Para['Phi_drug'], self.Para['delta_drug'].transpose() * self.Para['Theta'])
            P = 1 - np.exp(-Lambda)
            P[P == 0] = eps
            P[P == 1] = 1 - eps
            like_drug = np.sum(np.sum(self.data_drug * np.log(P) + (1 - self.data_drug) * np.log(1 - P))) / self.V_drug

            Lambda = np.dot(self.Para['Phi_procedure'], self.Para['delta_procedure'].transpose() * self.Para['Theta'])
            P = 1 - np.exp(-Lambda)
            P[P == 0] = eps
            P[P == 1] = 1 - eps
            like_procedure = np.sum(np.sum(self.data_procedure * np.log(P) + (1 - self.data_procedure) * np.log(1 - P))) / self.V_procedure

            end = time.time()
            Time = end-begin

            print('Iteration {}/{}, Diagnosis Likelihood {}, Drug Likelihood {}, Procedure Llikelihood {}, Time in seconds {}'.format(i, iter_all, like_diag, like_drug, like_procedure, int(Time)))

            if i > self.Setting['Burn_in']:
                if np.mod(i, self.Setting['Step']) == 0:
                    self.collection['delta_diag_sum'] += self.Para['delta_diag']
                    self.collection['delta_drug_sum'] += self.Para['delta_drug']
                    self.collection['delta_procedure_sum'] += self.Para['delta_procedure']

                    self.collection['Phi_diag_sum'] += self.Para['Phi_diag']
                    self.collection['Phi_drug_sum'] += self.Para['Phi_drug']
                    self.collection['Phi_procedure_sum'] += self.Para['Phi_procedure']

                    self.collection['Theta_sum'] += self.Para['Theta']
                    self.collection['Pi_sum'] += self.Para['Pi']
                    self.collection['flag'] += 1

                    Phi_diag_mean = self.collection['Phi_diag_sum'] / self.collection['flag']
                    Phi_drug_mean = self.collection['Phi_drug_sum'] / self.collection['flag']
                    Phi_procedure_mean = self.collection['Phi_procedure_sum'] / self.collection['flag']

                    delta_diag_mean = self.collection['delta_diag_sum'] / self.collection['flag']
                    delta_drug_mean = self.collection['delta_drug_sum'] / self.collection['flag']
                    delta_procedure_mean = self.collection['delta_procedure_sum'] / self.collection['flag']

                    Theta_mean = self.collection['Theta_sum'] / self.collection['flag']
                    Pi_mean = self.collection['Pi_sum'] / self.collection['flag']

                    Lambda = np.dot(Phi_diag_mean, delta_diag_mean.transpose() * Theta_mean)
                    P = 1 - np.exp(-Lambda)
                    P[P == 0] = eps
                    P[P == 1] = 1 - eps
                    like_diag = np.sum(np.sum(self.data_diag * np.log(P) + (1 - self.data_diag) * np.log(1 - P))) / self.V_diag

                    Lambda = np.dot(Phi_drug_mean, delta_drug_mean.transpose() * Theta_mean)
                    P = 1 - np.exp(-Lambda)
                    P[P == 0] = eps
                    P[P == 1] = 1 - eps
                    like_drug = np.sum(np.sum(self.data_drug * np.log(P) + (1 - self.data_drug) * np.log(1 - P))) / self.V_drug

                    Lambda = np.dot(Phi_procedure_mean, delta_procedure_mean.transpose() * Theta_mean)
                    P = 1 - np.exp(-Lambda)
                    P[P == 0] = eps
                    P[P == 1] = 1 - eps
                    like_procedure = np.sum(np.sum(self.data_procedure * np.log(P) + (1 - self.data_procedure) * np.log(1 - P))) / self.V_procedure

                    print('Iteration {}/{}, collected diagnosis Likelihood {}, collected drug Likelihood {}, collected procedure Likelihood {}'.format(i, iter_all, like_diag, like_drug, like_procedure))

        Theta_mean = self.collection['Theta_sum'] / self.collection['flag']
        Phi_diag_mean = self.collection['Phi_diag_sum'] / self.collection['flag']
        Phi_drug_mean = self.collection['Phi_drug_sum'] / self.collection['flag']
        Phi_procedure_mean = self.collection['Phi_procedure_sum'] / self.collection['flag']
        Pi_mean = self.collection['Pi_sum'] / self.collection['flag']
        if train_flag:
            np.save('./trained_model/Theta_mean.npy', Theta_mean)
            np.save('./trained_model/Phi_diag_mean.npy', Phi_diag_mean)
            np.save('./trained_model/Phi_drug_mean.npy', Phi_drug_mean)
            np.save('./trained_model/Phi_procedure_mean.npy', Phi_procedure_mean)
            np.save('./trained_model/Pi_mean.npy', Pi_mean)

def truncated_Poisson_sample(poisson_rate):

    poisson_rate_1 = poisson_rate[np.where(poisson_rate>1)]
    poisson_rate_2 = poisson_rate[np.where(poisson_rate<=1)]

    x = np.zeros_like(poisson_rate)
    x_1 = np.zeros([poisson_rate_1.size])
    x_2 = np.zeros([poisson_rate_2.size])

    while True:
        sample_index = np.where(x_1 == 0)
        if sample_index[0].size == 0:
            break
        else:
            rate_1_remain = poisson_rate_1[sample_index]
            temp = np.random.poisson(rate_1_remain)
            index = temp > 0
            x_1[sample_index[0][index]] = temp[index]
    x[np.where(poisson_rate>1)] = x_1

    while True:
        sample_index = np.where(x_2 == 0)
        if sample_index[0].size == 0:
            break
        else:
            rate_2_remain = poisson_rate_2[sample_index]
            temp = 1 + np.random.poisson(rate_2_remain)
            index = np.random.rand(temp.size) < (1./temp)
            x_2[sample_index[0][index]] = temp[index]
    x[np.where(poisson_rate<=1)] = x_2

    return x