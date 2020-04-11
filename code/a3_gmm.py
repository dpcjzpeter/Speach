import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = '/u/cs401/A3/data/'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m: int):
        """
        Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        sigma = self.Sigma[m]
        return (
                np.sum((self.mu[m] ** 2) / (2 * sigma))
                + (self._d / 2) * np.log(2 * np.pi)
                + 0.5 * np.log(np.prod(sigma)))


def log_b_m_x(m: int, x: np.ndarray, my_theta: theta) -> float:
    """ Returns the log probability of d-dimensional vector x using only
        component m of model my_theta (See equation 1 of the handout)
    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `my_theta.preComputedForM(m)` for this.
    Return shape:
        (single row) if x.shape == [d], then return value is float
            (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]
    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    sigma = np.reciprocal(my_theta.Sigma[m],
                          where=my_theta.Sigma[m] != 0)
    if len(x.shape) == 1:
        return -np.sum(
            0.5 * (x ** 2) * sigma
            - (my_theta.mu[m] * (x.T)) * sigma) \
            - my_theta.precomputedForM(m)
    else:
        return -np.sum(
            0.5 * (x ** 2) * sigma
            - my_theta.mu[m] * (x) * sigma, axis=1) \
            - my_theta.precomputedForM(m)


def log_p_m_x(log_Bs, myTheta):
    """ The newer version!!

    Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    return log_Bs * myTheta.omega / np.sum(log_Bs * myTheta.omega, axis=0)


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
        log_Bs = np.array([log_b_m_x(j, X, myTheta) for j in range(M)])

        log_lik = logLik(log_Bs, myTheta)

        log_pmx = log_p_m_x(log_Bs, myTheta)
        sum_log_pmx = np.sum(log_pmx, axis=1).reshape((M, 1))

        myTheta.omega = sum_log_pmx / float(T)
        myTheta.mu = np.sum(log_pmx.dot(X), axis=1) / sum_log_pmx
        myTheta.Sigma = (np.sum(log_pmx.dot(X ** 2), axis=1) / sum_log_pmx) - (myTheta.mu ** 2)

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
