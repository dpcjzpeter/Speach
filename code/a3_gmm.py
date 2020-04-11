import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    M, d = myTheta.mu.shape

    if not preComputedForM:
        return -0.5 * np.sum((x - myTheta.mu[m])**2 / myTheta.Sigma[m]) \
               - d/2 * np.log(2*np.pi) \
               - 0.5*np.log(np.prod(myTheta.Sigma[m]))
    else:
        return -np.sum((0.5*(x**2) - myTheta.mu[m]*x) / myTheta.Sigma[m]) -preComputedForM[m]


def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M = myTheta.omega.shape[0]

    omega_log_b_m_xs = [myTheta.omega[i].dot(log_b_m_x(i, x, myTheta, preComputedForM=[])) for i in range(M)]

    return myTheta.omega[m].dot(log_b_m_x(m, x, myTheta, preComputedForM=[])) / np.sum(omega_log_b_m_xs)


def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    return np.sum(logsumexp(log_Bs, axis=0, b=myTheta.omega))


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta(speaker, M, X.shape[1])

    T, _ = X.shape

    prev_log_lik, improvement = -float('inf'), float('inf')

    for i in range(maxIter):
        log_Bs = [log_p_m_x(j, X, myTheta) for j in range(M)]
        log_Bs = np.array(log_Bs)
        log_lik = logLik(log_Bs, myTheta)

        sum_log_Bs = np.sum(log_Bs, axis=1).reshape((M, 1))
        myTheta.omega = sum_log_Bs / float(T)
        myTheta.mu = np.sum(log_Bs.dot(X), axis=1) / sum_log_Bs
        myTheta.Sigma = (np.sum(log_Bs.dot(X ** 2), axis=1) / sum_log_Bs) - (myTheta.mu ** 2)

        imporovement = log_lik - prev_log_lik
        if imporovement >= epsilon:
            break

        prev_log_lik = log_lik

    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    #bestModel = -1

    lst_log_Bs = [np.array([log_p_m_x(i, mfcc, model) for i in range(models[0].omega.shape[0])])
                  for model in models]

    predictions = [(i, model, logLik(lst_log_Bs[i], model)) for i, model in enumerate(models)]

    predictions = sorted(predictions, key=lambda x: x[2])

    bestModel = predictions[-1][0]

    print(models[correctID].name)
    for i in range(min(k, len(models))):
        print('{} {}'.format(predictions[i][1].name, predictions[i][2]))

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
