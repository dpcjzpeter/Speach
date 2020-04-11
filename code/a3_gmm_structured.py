import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        return np.sum((self.mu[m] ** 2) / (2 * self.Sigma[m])) \
               + (self._d / 2) * np.log(2 * np.pi) \
               + 0.5 * np.log(np.prod(self.Sigma[m]))

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    sigma = np.reciprocal(myTheta.Sigma[m],
                          where=myTheta.Sigma[m] != 0)
    if len(x.shape) == 1:
        return -np.sum(0.5 * (x ** 2) * sigma - myTheta.mu[m] * x.T * sigma) - myTheta.precomputedForM(m)
    else:
        return -np.sum(0.5 * (x ** 2) * sigma - myTheta.mu[m] * x * sigma, axis=1) - myTheta.precomputedForM(m)


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    return log_Bs * myTheta.omega / np.sum(log_Bs * myTheta.omega, axis=0)


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    return np.sum(logsumexp(log_Bs, axis=0, b=myTheta.omega))


def train(speaker, X: np.ndarray,
          M: int = 8, epsilon: float = 0.0,
          max_iter: int = 20) -> theta:
    """ Train a model for the given speaker. Returns the theta
        (omega, mu, sigma)
    """

    T, d = X.shape
    my_theta = theta(speaker, M, d)
    # perform initialization (Slide 32)
    indices = random.sample(range(T), M)
    my_theta.reset_mu(X[indices])
    my_theta.reset_omega(np.ones((M, 1)) / M)
    my_theta.reset_Sigma(np.ones((M, d)))

    i = 0
    prev_L = float('-inf')
    improvement = float('inf')
    while i <= max_iter and improvement >= epsilon:
        log_Bs = np.array([log_b_m_x(m=i, x=X, myTheta=my_theta)
                           for i in range(M)])
        assert log_Bs.shape == (M, T), \
            f"log_Bs is of shape {log_Bs.shape} and should be ({M}, {T})"

        L = logLik(log_Bs, my_theta)
        # p_xt = np.exp(np.log(my_theta.omega) + log_Bs -
        #               logsumexp(log_Bs, b=my_theta.omega, axis=0))
        p_xt = log_p_m_x(log_Bs, my_theta)
        p_xt_sum = np.sum(p_xt, axis=1).reshape((M, 1))
        my_theta.omega = p_xt_sum / float(T)
        my_theta.mu = np.divide(
            p_xt.dot(X),
            p_xt_sum,
            out=np.zeros((M, d)), where=p_xt_sum != 0)
        my_theta.Sigma = np.divide(
            p_xt.dot(X ** 2),
            p_xt_sum,
            out=np.zeros((M, d)), where=p_xt_sum != 0) - my_theta.mu ** 2

        assert my_theta.omega.size == my_theta._M, \
            "`omega` must contain M elements"
        assert my_theta.mu.shape == (
            my_theta._M, my_theta._d), "`mu` must be of size (M,d)"
        assert my_theta.Sigma.shape == (
            my_theta._M, my_theta._d), "`Sigma` must be of size (M,d)"
        improvement = L - prev_L
        prev_L = L
        i += 1
    return my_theta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    lst_log_Bs = [np.array([log_b_m_x(i, mfcc, model) for i in range(models[0].omega.shape[0])])
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
    print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
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
