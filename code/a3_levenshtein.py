import os
import numpy as np
import re
import string
from scipy import stats

from pathlib import Path

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


def preprocess_lines(lines: List[str]) -> str:
    processed_lines = list()
    for line in lines:
        line = line.lower()\
                   .replace("\n", " ")\
                   .replace("\r", " ")
        line = re.sub(r"\s+", " ", line)
        line = line.strip()

        remove = string.punctuation
        remove = remove.replace("[", "")
        remove = remove.replace("]", "")
        pattern = r"[{}]".format(remove)

        line = re.sub(pattern, " ", line)
        line = re.sub(' +', ' ', line)
        line = line.strip()
        processed_lines.append(line)
    return processed_lines


if __name__ == "__main__":
    out_file = open("asrDiscussion.txt", "w")
    kaldi_err = list()
    google_err = list()
    print("Sanity check")
    print(f'Got: {Levenshtein("who is there".split(), "is there".split())}')
    print("Expected: (0.333, 0, 0, 1)")
    print(f'Got: {Levenshtein("who is there".split(), "".split())}')
    print("Expected: (1.0, 0, 0, 3)")
    print(f'Got: {Levenshtein("".split(), "who is there".split())}')
    print("Expected: (Inf, 0, 3, 0)")
    print("\n")

    for speaker in Path(dataDir).iterdir():
        print(f"speaker: {speaker}")
        ref_lines = (speaker / 'transcripts.txt').open().read().split('\n')
        if ref_lines[-1] == '':
            ref_lines = ref_lines[:-1]
        ref_lines = preprocess_lines(ref_lines)

        kaldi_lines = (
            speaker / 'transcripts.Kaldi.txt').open().read().split('\n')
        if kaldi_lines[-1] == '':
            kaldi_lines = kaldi_lines[:-1]
        kaldi_lines = preprocess_lines(kaldi_lines)

        google_lines = (
            speaker / 'transcripts.Google.txt').open().read().split('\n')
        if google_lines[-1] == '':
            google_lines = google_lines[:-1]
        google_lines = preprocess_lines(google_lines)

        num_lines = min(len(ref_lines), len(kaldi_lines), len(google_lines))

        speaker_name = str(speaker).split('/')[-1]
        for i in range(num_lines):
            print(f"line: {i} / {num_lines}")
            ref_list = ref_lines[i].split()
            kaldi_list = kaldi_lines[i].split()
            google_list = google_lines[i].split()

            kaldi_err_temp = Levenshtein(ref_list, kaldi_list)
            google_err_temp = Levenshtein(ref_list, google_list)

            out_file.write(
                '{0} {1} {2} {3: 1.4f} S:{4}, I:{5}, D:{6}\n'.format(
                    speaker_name, 'Kaldi', i,
                    kaldi_err_temp[0], kaldi_err_temp[1], kaldi_err_temp[2],
                    kaldi_err_temp[3]))
            out_file.write(
                '{0} {1} {2} {3: 1.4f} S:{4}, I:{5}, D:{6}\n'.format(
                    speaker_name, 'Google', i,
                    google_err_temp[0], google_err_temp[1], google_err_temp[2],
                    google_err_temp[3]))
            kaldi_err.append(kaldi_err_temp)
            google_err.append(google_err_temp)

        out_file.write('\n')

    kaldi_err = np.array(kaldi_err)
    google_err = np.array(google_err)
    t_value, p_value = stats.ttest_ind(
        google_err, kaldi_err, equal_var=False)
    if not isinstance(t_value, float):
        t_value = t_value[0]
    if not isinstance(p_value, float):
        p_value = p_value[0]
    output =\
        'Google WER Average: {0: .4f}, Google WER Standard Deviation: ' +\
        '{1: 1.4f}\nKaldi WER Average: {2: .4f}, Kaldi WER Standard ' +\
        'Deviation: {3: .4f}\n'
    out_file.write(output.format(
        np.mean(google_err[:, 0]), np.std(
            google_err[:, 0]), np.mean(kaldi_err[:, 0]),
        np.std(kaldi_err[:, 0])))
    output =\
        'Google S Average: {0: .4f}, Google S Standard Deviation: ' +\
        '{1: 1.4f}\nKaldi S Average: {2: .4f}, Kaldi S Standard ' +\
        'Deviation: {3: .4f}\n'
    out_file.write(output.format(
        np.mean(google_err[:, 1]), np.std(
            google_err[:, 1]), np.mean(kaldi_err[:, 1]),
        np.std(kaldi_err[:, 1])))
    output =\
        'Google I Average: {0: .4f}, Google I Standard Deviation: ' +\
        '{1: 1.4f}\nKaldi I Average: {2: .4f}, Kaldi I Standard ' +\
        'Deviation: {3: .4f}\n'
    out_file.write(output.format(
        np.mean(google_err[:, 2]), np.std(
            google_err[:, 2]), np.mean(kaldi_err[:, 2]),
        np.std(kaldi_err[:, 2])))
    output =\
        'Google D Average: {0: .4f}, Google D Standard Deviation: ' +\
        '{1: 1.4f}\nKaldi D Average: {2: .4f}, Kaldi D Standard ' +\
        'Deviation: {3: .4f}\n'
    out_file.write(output.format(
        np.mean(google_err[:, 3]), np.std(
            google_err[:, 3]), np.mean(kaldi_err[:, 3]),
        np.std(kaldi_err[:, 3])))
    out_file.close()