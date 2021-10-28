"""
===========================================
Poisson Gamma Dynamical Systems
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0


import time
import numpy as np
from pydpm.utils.Metric import *
from pydpm.utils import Model_Sampler_CPU
import scipy

class mPGDS_multi_sample(object):

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

    def initial(self, X_diag, X_drug, X_proc, Patient_label, burn_in = 200, step=5):
        self.X_diag = X_diag
        self.X_drug = X_drug
        self.X_proc = X_proc

        self.Patient_label = Patient_label
        K = self.K
        self.V_diag, self.N = self.X_diag.shape
        self.V_drug = self.X_drug.shape[0]
        self.V_proc = self.X_proc.shape[0]
        L = self.L
        self.num_patient = int(Patient_label.max())


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

        self.Para['Phi_diag'] = np.random.rand(self.V_diag, K)
        self.Para['Phi_diag'] = self.Para['Phi_diag'] / np.sum(self.Para['Phi_diag'], axis=0)
        self.Para['Phi_drug'] = np.random.rand(self.V_drug, K)
        self.Para['Phi_drug'] = self.Para['Phi_drug'] / np.sum(self.Para['Phi_drug'], axis=0)
        self.Para['Phi_proc'] = np.random.rand(self.V_proc, K)
        self.Para['Phi_proc'] = self.Para['Phi_proc'] / np.sum(self.Para['Phi_proc'], axis=0)
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
        self.Para['delta_proc'] = np.ones((self.N, 1))
        self.Para['L_dotkt_all'] = []
        #############################################################################

        self.collection = {}
        self.collection['delta_sum'] = 0
        self.collection['Phi_sum'] = 0
        self.collection['Theta_sum'] = 0
        self.collection['Pi_sum'] = 0
        self.collection['flag'] = 0



    def train(self, iter_all=200):

        self.Likelihood = []
        for i in range(iter_all):

            X_diag = np.array(self.X_diag, dtype=np.double, order='C')
            [A_KN_diag, A_VK_diag] = self.Multrnd_Matrix(X_diag, self.Para['Phi_diag'], self.Para['delta_diag'].transpose()*self.Para['Theta'])

            X_drug = np.array(self.X_drug, dtype=np.double, order='C')
            [A_KN_drug, A_VK_drug] = self.Multrnd_Matrix(X_drug, self.Para['Phi_drug'], self.Para['delta_drug'].transpose() * self.Para['Theta'])

            X_proc = np.array(self.X_drug, dtype=np.double, order='C')
            [A_KN_proc, A_VK_proc] = self.Multrnd_Matrix(X_proc, self.Para['Phi_proc'], self.Para['delta_proc'].transpose() * self.Para['Theta'])

            L_KK = np.zeros((self.K, self.K))
            L_kdott_for_V = np.zeros((self.K, self.num_patient))

            for n in range(self.num_patient):
                index  = np.squeeze(self.Patient_label==n)
                x_patient_diag = X_diag[:, index]
                x_patient_drug = X_drug[:, index]
                x_patient_proc = X_proc[:, index]

                patient_T = x_patient.shape[1]
                L_kdott = np.zeros((self.K, patient_T))

                Theta_patient = self.Para['Theta'][:, index]
                A_KN_patient = A_KN[:, index]

                if i==0:
                    self.Para['L_dotkt_all'].append(np.zeros((self.K, patient_T+1)))

                for t in range(patient_T-1, 1, -1):
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
            self.Para['Phi'] = Model_Sampler_CPU.Sample_Pi(A_VK, self.Supara['eta0'])

            # Sample Pi
            Piprior = np.dot(self.Para['V'], np.transpose(self.Para['V']))
            Piprior[np.arange(Piprior.shape[0]), np.arange(Piprior.shape[1])] = 0

            Piprior = Piprior + np.diag(np.reshape(self.Para['Xi'] * self.Para['V'], [self.Para['V'].shape[0], 1]))
            self.Para['Pi'] = Model_Sampler_CPU.Sample_Pi(L_KK, Piprior)

            ###################################################### Calculate for local
            L_dotkt_all_sum = 0
            Zeta_all_for_v = 0

            # Sample Theta
            for n in range(self.num_patient):
                index = np.squeeze(self.Patient_label == n)
                x_patient = X_train[:, index]
                patient_T = x_patient.shape[1]
                Theta_patient = self.Para['Theta'][:, index]

                A_KN_patient = A_KN[:, index]
                delta_patient = self.Para['delta'][index]
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
            self.Para['q'] = np.maximum(np.random.beta(b.squeeze(), a), 1e-10)
            # Sample h checked
            for k1 in range(self.K):
                for k2 in range(self.K):
                    self.Para['h'][k1:k1 + 1, k2:k2 + 1] = Model_Sampler_CPU.Crt_Matrix(L_KK[k1:k1 + 1, k2:k2 + 1], Piprior[k1:k1 + 1, k2:k2 + 1])
            # Sample Xi checked
            shape = self.Supara['gamma0'] / self.K + np.trace(self.Para['h'])
            scale = self.Para['beta'] - np.dot(np.transpose(self.Para['V']), log_max(self.Para['q']))
            self.Para['Xi'] = np.transpose(np.random.gamma(shape) / scale)

            # Sample V # check
            for n in range(self.num_patient):
                index = np.squeeze(self.Patient_label == n)
                A_KN_patient = A_KN[:, index]
                for k in range(self.K):
                    L_kdott_for_V[k,n] = Model_Sampler_CPU.Crt_Matrix(np.reshape(A_KN_patient[k,0] +  self.Para['L_dotkt_all'][n][k,1], (1,1)),
                                                                      np.reshape(self.Supara['tao0'] * self.Para['V'][k], (1, 1)))
            for k in range(self.K):
                self.Para['n'][k] = np.sum(self.Para['h'][k, :] + np.transpose(self.Para['h'][:, k])) - self.Para['h'][k, k] + L_kdott_for_V[k,:].sum()
                self.Para['rou'][k] = - log_max(self.Para['q'][k]) * (self.Para['Xi'] + np.sum(self.Para['V']) - self.Para['V'][k]) \
                                      - np.dot(np.transpose(log_max(self.Para['q'])), self.Para['V']) \
                                      + log_max(self.Para['q'][k]) * self.Para['V'][k] + Zeta_all_for_v

            shape_top = self.Supara['gamma0'] / self.K + self.Para['n']
            scale_top = self.Para['beta'] + self.Para['rou']
            self.Para['V'] = np.random.gamma(shape_top) / scale_top

            if self.Setting['Stationary'] == 0:
                for n in range(self.num_patient):
                    index = np.squeeze(self.Patient_label == n)
                    x_patient = X_train[:, index]
                    Theta_patient = self.Para['Theta'][:, index]

                    shape = self.Supara['epilson0'] + x_patient.sum(0)
                    scale = self.Supara['epilson0'] + Theta_patient.sum(0)
                    delta_patient = np.random.gamma(shape) / scale
                    self.Para['delta'][index, 0] = delta_patient

            # Likelihood

            if self.Setting['Stationary'] == 0:
                Lambda = np.dot(self.Para['Phi'], self.Para['delta'].transpose()*self.Para['Theta'])
                like = np.sum(self.data * np.log(Lambda) - Lambda) / self.V
            print('Iteration {}/{}, Likelihood {}'.format(i, iter_all, like))

            if i > self.Setting['Burn_in']:
                if i / self.Setting['Step'] == 0:
                    self.collection['delta_sum'] += self.Para['delta']
                    self.collection['Phi_sum'] += self.Para['Phi']
                    self.collection['Theta_sum'] += self.Para['Theta']
                    self.collection['Pi_sum'] += self.Para['Pi']
                    self.collection['flag'] += 1

                    Phi_mean = self.collection['Phi_sum'] / self.collection['flag']
                    delta_mean = self.collection['delta_sum'] / self.collection['flag']
                    Theta_mean = self.collection['Theta_sum'] / self.collection['flag']
                    Pi_mean = self.collection['Pi_sum'] / self.collection['flag']

                    Lambda = np.dot(Phi_mean, delta_mean.transpose() * Theta_mean)
                    like = np.sum(self.data * np.log(Lambda) - Lambda) / self.V
                    print('Iteration {}/{}, collected Likelihood {}'.format(i, iter_all, like))