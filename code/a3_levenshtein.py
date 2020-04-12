import os
import numpy as np
import re
import string
from scipy import stats

dataDir = '/u/cs401/A3/data/'


UP_LEFT = {'up': 1, 'left': 2}


def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    N, M = len(r), len(h)

    if N == 0 and M == 0:
        return float(0), 0, 0, 0
    if N == 0 and M != 0:
        return float('inf'), 0, M, 0
    if N != 0 and M == 0:
        return 1., 0, 0, N

    R, B = np.zeros((N + 1, M + 1)), np.zeros((N + 1, M + 1))

    R[0, :] = np.arange(M + 1)
    R[:, 0] = np.arange(N + 1)

    for i in range(1, N + 1):
        for j in range(1, M + 1):

            if r[i - 1] != h[j - 1]:
                R[i - 1, j - 1] += 1

            R[i, j] = min([R[i - 1, j] + 1, R[i - 1, j - 1], R[i, j - 1] + 1])

            if R[i, j] == R[i - 1, j] + 1:
                B[i, j] = UP_LEFT['up']
            elif R[i, j] == R[i, j - 1] + 1:
                B[i, j] = UP_LEFT['left']
            else:
                B[i, j] = UP_LEFT['up'] + UP_LEFT['left']

    r = N
    c = M
    count = [0, 0]
    while True:
        if r <= 0 and c <= 0:
            break
        if B[r, c] == UP_LEFT['up']:
            count[1] += 1
            r -= 1
        if B[r, c] == UP_LEFT['left']:
            count[0] += 1
            c -= 1
        if B[r, c] == UP_LEFT['up'] + UP_LEFT['left']:
            r -= 1
            c -= 1

    return R[N, M] / float(N), int(R[N, M] - sum(count)), count[0], count[1]


# helper to preprocess each line
def preprocess(line):
    line = line.lower()
    line = line.replace('\n', ' ').replace('\r', ' ')
    line = re.sub(r"\s+", " ", line)
    line = line.strip()

    # punctuations without '[' and ']'
    stripped_punctuation = string.punctuation
    stripped_punctuation = stripped_punctuation.replace('[', '').replace(']', '')
    pattern = re.compile(r'[{}]'.format(stripped_punctuation))

    line = re.sub(pattern, ' ', line)
    line = re.sub(' +', ' ', line)
    line = line.strip()

    return line


def helper(lines):
    if lines[-1] == '':
        return lines[:-1]
    return lines


if __name__ == "__main__":
    kaldi_err, google_err = [], []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            trans_path = os.path.join(dataDir, speaker, 'transcripts.txt')

            trans_google_path = os.path.join(dataDir, speaker, 'transcripts.Google.txt')
            trans_kaldi_path = os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')

            trans_lines = open(trans_path, 'r').read().split('\n')
            old_lines = helper(trans_lines)
            trans_lines = []
            for i, line in enumerate(old_lines):
                trans_lines.append(preprocess(line))

            google_lines = open(trans_google_path, 'r').read().split('\n')
            old_lines = helper(google_lines)
            google_lines = []
            for i, line in enumerate(old_lines):
                google_lines.append(preprocess(line))

            kaldi_lines = open(trans_kaldi_path, 'r').read().split('\n')
            old_lines = helper(kaldi_lines)
            kaldi_lines = []
            for i, line in enumerate(old_lines):
                kaldi_lines.append(preprocess(line))

            for i in range(min(len(trans_lines), len(google_lines), len(kaldi_lines))):
                google_result = Levenshtein(trans_lines, google_lines)
                kaldi_result = Levenshtein(trans_lines, kaldi_lines)

                google_err.append(google_result[0])
                kaldi_err.append(kaldi_result[0])

                # [SPEAKER] [SYSTEM] [i]  [WER] S:[numSubstitutions], I:[numInsertions], D:[numDeletions]
                print('{0} {1} {2}  {3:1.4f} S:{4}, I:{5}, D:{6}'.format(speaker, 'Google', i,
                                                                        google_result[0], google_result[1],
                                                                        google_result[2], google_result[3]))

                print('{0} {1} {2}  {3:1.4f} S:{4}, I:{5}, D:{6}'.format(speaker, 'Kaldi', i,
                                                                        kaldi_result[0], kaldi_result[1],
                                                                        kaldi_result[2], kaldi_result[3]))
            print('\n')

    google_err = np.array(google_err)
    kaldi_err = np.array(kaldi_err)

    print('Google WER Average: {0:1.4f}, Google WER Standard Deviation: {1:1.4f}\n'
          'Kaldi WER Average: {2:1.4f}, Kaldi Standard Deviation: {3:1.4f}'.format(np.mean(google_err),
                                                                                   np.std(google_err),
                                                                                   np.mean(kaldi_err),
                                                                                   np.std(kaldi_err)))

