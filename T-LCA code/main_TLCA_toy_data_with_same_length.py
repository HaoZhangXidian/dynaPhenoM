import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t = np.arange(0,14,1)
mu1 = 1+0.01*t+0.02*(t**2)+0.03*(t**3)
X1_1 = mu1[np.newaxis,:] + 2*np.random.randn(300,14)
mu2 = 0.8-0.01*t-0.02*(t**2)-0.03*(t**3)
X1_2 = mu2[np.newaxis,:] + 1.5*np.random.randn(300,14)
X1 = np.concatenate([X1_1[:,:,np.newaxis], X1_2[:,:,np.newaxis]], axis=2)

mu1 = 1+0.005*t+0.01*(t**2)+0.015*(t**3)
X2_1 = mu1[np.newaxis,:] + 0.5*np.random.randn(300,14)
mu2 = 0.6+0.02*t+0.01*(t**2)+0.01*(t**3)
X2_2 = mu2[np.newaxis,:] + 3*np.random.randn(300,14)
X2 = np.concatenate([X2_1[:,:,np.newaxis], X2_2[:,:,np.newaxis]], axis=2)

X = np.concatenate([X1,X2], axis=0)

num_patient = X.shape[0]
num_feature = X.shape[2]
length = X.shape[1]

# Initialize parameters
num_cluster = 2
Beta = 0.01*np.random.randn(num_feature, 4, num_cluster)
Alpha = 1/num_cluster * np.ones(num_cluster)
sigma = np.ones([num_feature])

Tau = np.zeros([length, 4])
for t in range(length):
    Tau[t, :] = np.array([1, t/length, (t/length)**2, (t/length)**3])

num_iteration = 5000

Omega = np.zeros([num_patient, num_cluster])
c = 2*np.pi

Matrix_tmp = np.zeros([num_patient, num_cluster])

for i in range(num_iteration):
    mu = np.matmul(np.repeat(Tau[np.newaxis,:,:], num_feature, axis=0), Beta).transpose(1,2,0)
    ## E step
    for k in range(num_cluster):
        tmp = ((X - mu[:, k, :]) / sigma) ** 2

        Matrix_tmp[:, k] = (-0.5*tmp).sum(1).sum(1) + np.log(Alpha[k])

    Matrix_tmp = Matrix_tmp - Matrix_tmp.max(1, keepdims=True)

    Probability = np.exp(Matrix_tmp) / np.exp(Matrix_tmp).sum(1, keepdims=True)

    ## M step
    # Update alpha
    Alpha = Probability.sum(0)/num_patient

    # Update beta
    for k in range(num_cluster):
        for j in range(4):
            for d in range(num_feature):
                mu = np.matmul(np.repeat(Tau[np.newaxis,:,:], num_feature, axis=0), Beta).transpose(1,2,0)
                tmp = Tau[:, j]*Beta[d, j, k]
                A = X[:,:,d] - mu[:,k,d]+tmp
                Beta[d,j,k] = ((Probability[:, k][:, np.newaxis]*A*Tau[:, j]).sum()) / ((Probability[:, k][:, np.newaxis]*((Tau[:, j][np.newaxis,:])**2)).sum())

    # Update sigma
    fenzi = 0
    fenmu = 0
    mu = np.matmul(np.repeat(Tau[np.newaxis, :, :], num_feature, axis=0), Beta).transpose(1, 2, 0)
    for d in range(num_feature):
        for k in range(num_cluster):
            fenzi += (Probability[:, k][:, np.newaxis] * ((X[:,:,d] - mu[:, k, d]) ** 2)).sum()
            fenmu += (Probability[:, k] * length).sum()

        sigma[d] = np.sqrt(fenzi/fenmu)

    ## Calculate loss function
    mu = np.matmul(np.repeat(Tau[np.newaxis, :, :], num_feature, axis=0), Beta).transpose(1, 2, 0)
    LL = 0
    for k in range(num_cluster):
        tmp = ((X - mu[:, k, :]) / sigma) ** 2
        LL = LL + Alpha[k] * np.exp((-0.5 * tmp).sum(1).sum(1)) * 1 / ( np.prod(sigma**(length)) * ((np.sqrt(c)) **(length*num_feature)) )

    likelihood = np.sum(np.log(LL))


    print('Iteration: %d Likelihood: %.4f' % (i, likelihood))

# plot the results

Labels = Probability.argmax(1)
mu = np.matmul(np.repeat(Tau[np.newaxis, :, :], num_feature, axis=0), Beta).transpose(1, 2, 0)

Centers_mean = np.zeros([num_cluster, length, num_feature])
Centers_std = np.zeros([num_cluster, length, num_feature])
for i in range(num_cluster):
    for d in range(num_feature):
        #Centers_mean[i, :, d] = X[Labels==i, :, d].mean(0)
        Centers_mean[i, :, d] = mu[:,i,d]
        Centers_std[i, :, d] = X[Labels==i, :, d].std(0)

T = np.arange(0,14,1)
plt.figure()
for i in range(num_cluster):
    plt.plot(T, Centers_mean[i, :, 0])
plt.title('Feature 1')
plt.savefig('./results/feature_1.png')

plt.figure()
for i in range(num_cluster):
    plt.plot(T, Centers_mean[i, :, 1])
plt.title('Feature 2')
plt.savefig('./results/feature_2.png')