from typing import List, Tuple
from pathlib import Path
from scipy import stats
from enum import Enum

import numpy as np
import operator
import string
import re

dataDir = '/u/cs401/A3/data/'
# dataDir = '../data'


class BackTrackElement(Enum):
    # corresponds to indexes
    UP = 0
    LEFT = 2
    UP_LEFT = 1


def Levenshtein(r: List[str], h: List[str]) -> Tuple[float, int, int, int]:
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
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions,
    insertions, and deletions respectively
    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    N = len(r)  # num reference words
    M = len(h)  # num hypothesis words
    if N == 0 and M == 0:
        return (float(0), 0, 0, 0)
    if N == 0 and M != 0:
        return (float('inf'), 0, M, 0)
    if N != 0 and M == 0:
        return (1., 0, 0, N)
    # TODO +2?
    B = np.zeros((N+1, M+1))  # backtracking matrix
    R = np.zeros((N+1, M+1))  # matrix of distances
    R[0, :] = np.arange(M+1)
    R[:, 0] = np.arange(N+1)
    for i in range(1, N+1):
        for j in range(1, M+1):
            val = 1 if r[i-1] != h[j-1] else 0
            B[i, j], R[i, j] = min(enumerate(
                [R[i-1, j] + 1,  # deletion error (UP)
                 R[i-1, j-1] + val,  # UP_LEFT
                 R[i, j-1] + 1]),  # insertion error # LEFT
                key=operator.itemgetter(1))
    row = N
    col = M
    counts = [0, 0]
    while True:
        if row <= 0 and col <= 0:
            break
        if B[row, col] == BackTrackElement.UP.value:
            counts[1] += 1
            row -= 1
        if B[row, col] == BackTrackElement.LEFT.value:
            counts[0] += 1
            col -= 1
        if B[row, col] == BackTrackElement.UP_LEFT.value:
            row -= 1
            col -= 1

    return (R[N, M] / float(N),
            int(R[N, M] - counts[0] - counts[1]),
            counts[0], counts[1])


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
                '{0} {1} {2} WER:{3: 1.4f} S:{4}, I:{5}, D:{6}\n'.format(
                    speaker_name, 'Kaldi', i,
                    kaldi_err_temp[0], kaldi_err_temp[1], kaldi_err_temp[2],
                    kaldi_err_temp[3]))
            out_file.write(
                '{0} {1} {2} WER:{3: 1.4f} S:{4}, I:{5}, D:{6}\n'.format(
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