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


############################################################################
X_new = np.zeros(X.shape)
length_new = np.zeros([num_patient])
Index_record = np.zeros([num_patient, length])
for i in range(num_patient):
    a = int(np.random.choice([6,9,12], 1))
    b = np.sort(np.random.choice(np.arange(1,13), a, replace=False))
    Index_record[i, 0:a+2] = np.concatenate([np.array([0]), b, np.array([13])])
    X_new[i, 0, :] = X[i, 0, :]
    X_new[i, 1:a + 1, :] = X[i, b, :]
    X_new[i, a + 1, :] = X[i, -1, :]
    length_new[i] = a+2

X = X_new
############################################################################
length_unqiue = np.unique(length_new)
DD = []
for i in length_unqiue:
    DD.append(X[length_new==i])

plt.figure()
for i in range(len(DD)):
    T = np.arange(0,length_unqiue[i])
    dd = DD[i].mean(0)
    plt.plot(T, dd[0:int(length_unqiue[i]), 0])
plt.title('data_mean_Feature 1')
plt.savefig('./results/data_mean_Feature 1.png')

plt.figure()
for i in range(len(DD)):
    T = np.arange(0,length_unqiue[i])
    dd = DD[i].mean(0)
    plt.plot(T, dd[0:int(length_unqiue[i]), 1])
plt.title('data_mean_Feature 2')
plt.savefig('./results/data_mean_Feature 2.png')

##############################################################################

# Initialize parameters
num_cluster = 10
Beta = 0.0001*np.random.randn(num_feature, 4, num_cluster)
Alpha = 1/num_cluster * np.ones(num_cluster)
sigma = np.ones([num_feature])

Tau = np.zeros([num_patient, length, 4])
for i in range(num_patient):
    for t in range(int(length_new[i])):
        #Tau[i, t, :] = np.array([1, t/(length_new[i]-1), (t/(length_new[i]-1))**2, (t/(length_new[i]-1))**3])
        #Tau[i, t, :] = np.array([1, Index_record[i,t], Index_record[i,t] ** 2, Index_record[i,t] ** 3])
        Tau[i, t, :] = np.array([1, t, t ** 2, t ** 3])

num_iteration = 1000

Omega = np.zeros([num_patient, num_cluster])
c = 2*np.pi

Matrix_tmp = np.zeros([num_patient, num_cluster])

for i in range(num_iteration):

    Tau_tmp = np.repeat(Tau[:, np.newaxis, :, :], num_feature, axis=1)
    Beta_tmp = np.repeat(Beta[np.newaxis, :, :], num_patient, axis=0)

    mu = np.matmul(Tau_tmp, Beta_tmp).transpose(0,2,3,1) # N L K D
    ## E step
    for k in range(num_cluster):
        tmp = ((X - mu[:, :, k, :]) / sigma) ** 2

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
                Tau_tmp = np.repeat(Tau[:, np.newaxis, :, :], num_feature, axis=1)
                Beta_tmp = np.repeat(Beta[np.newaxis, :, :], num_patient, axis=0)
                mu = np.matmul(Tau_tmp, Beta_tmp).transpose(0,2,3,1)
                tmp = Tau[:, :, j]*Beta[d, j, k]
                A = X[:,:,d] - mu[:,:,k,d]+tmp
                Beta[d,j,k] = ((Probability[:, k][:, np.newaxis]*A*Tau[:, :, j]).sum()) / ((Probability[:, k][:, np.newaxis]*((Tau[:, :, j])**2)).sum())

    # Update sigma
    fenzi = 0
    fenmu = 0

    Tau_tmp = np.repeat(Tau[:, np.newaxis, :, :], num_feature, axis=1)
    Beta_tmp = np.repeat(Beta[np.newaxis, :, :], num_patient, axis=0)
    mu = np.matmul(Tau_tmp, Beta_tmp).transpose(0, 2, 3, 1)  # N L K D

    for d in range(num_feature):
        for k in range(num_cluster):
            fenzi += (Probability[:, k][:, np.newaxis] * ((X[:,:,d] - mu[:, :, k, d]) ** 2)).sum()
            fenmu += (Probability[:, k] * length_new).sum()

        sigma[d] = np.sqrt(fenzi/fenmu)

    ## Calculate loss function
    Tau_tmp = np.repeat(Tau[:, np.newaxis, :, :], num_feature, axis=1)
    Beta_tmp = np.repeat(Beta[np.newaxis, :, :], num_patient, axis=0)
    mu = np.matmul(Tau_tmp, Beta_tmp).transpose(0, 2, 3, 1)  # N L K D
    LL = 0
    for k in range(num_cluster):
        tmp = ((X - mu[:, :, k, :]) / sigma) ** 2
        LL = LL + Alpha[k] * np.exp((-0.5 * tmp).sum(1).sum(1)) * 1 / ( (np.prod(sigma)**(length_new)) * ( (np.sqrt(c)) **(length_new*num_feature) ) )


    likelihood = np.sum(np.log(LL))


    print('Iteration: %d Likelihood: %.4f' % (i, likelihood))

# plot the results

Labels = Probability.argmax(1)
length_unqiue = np.unique(length_new)
Centers_mean = {}

for ll in length_unqiue:
    Centers_mean[str(int(ll))] = np.zeros([num_cluster, int(ll), num_feature])
    DD1 = X[length_new==ll]
    labels1 = Labels[length_new==ll]
    T = np.arange(0, int(ll), 1)

    for d in range(num_feature):
        plt.figure()

        for i in range(num_cluster):
            if (labels1==i).sum() == 0:
                Centers_mean[str(int(ll))][i, :, d] = np.zeros([int(ll)])
            else:
                Centers_mean[str(int(ll))][i, :, d] = DD1[labels1==i, 0:int(ll), d].mean(0)

            plt.plot(T, Centers_mean[str(int(ll))][i, :, d])
        plt.savefig('./results/l'+str(int(ll))+'_d'+str(d)+'.png')

