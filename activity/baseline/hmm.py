import csv
import sys
import math
import time
import numpy as np
from hmmlearn import hmm

'''
This script used hidden markov models to detect activity of unknown trajectories
using by comparing to other labeled examples of each trajectory type.
it is necessary to first do feature extraction by running the script: './FeatureExtraction.py'
before running this script. This may take a while to run, ~3.6 hours for 500 samples
'''

models = []


def get_mean_and_cov(subtrajectory):
    num_of_vectors = len(subtrajectory)
    num_of_features = len(subtrajectory[0])
    new_state = np.array(subtrajectory)
    mean = []
    covar = []
    if num_of_vectors == 1:
        mean = new_state[0]
        covar = np.array([1] * num_of_features)
    else:
        mean = np.mean(new_state, axis=0)
        covar = np.var(new_state, axis=0)
        i = 0
        for val in covar:
            if val == 0:
                covar[i] = 1
            i = i + 1
        mean[1] = mean[1] * num_of_vectors
    return mean, covar


def read_coordinate(file_name):
    with open(file_name, "r") as file:
        next(file)
        reader = csv.reader(file)
        states = []
        means = []
        covars = []
        subtrajectory = []
        last_quarter = -1
        for row in reader:
            the_row = []
            for cell in row:
                the_row.append(float(cell))
            subtrajectory.append(the_row)
            mean, covar = get_mean_and_cov(subtrajectory)
            states.append(mean)
            means.append(mean)
            covars.append(covar)
            subtrajectory = []

        return np.array(states), np.array(means), np.array(covars)


def get_probabilities(num_of_states):
    trans_matrix = np.zeros((num_of_states, num_of_states))
    for i in range(num_of_states - 1):
        num_of_following = (num_of_states - i - 1)
        denominator = num_of_following * (num_of_following + 1) / 2
        for j in range(i + 1, num_of_states):
            trans_matrix[i][j] = (num_of_following - (j - i - 1)) / denominator
    trans_matrix[-1] = [1 / num_of_states] * num_of_states
    start_prob = [1] + [0] * (num_of_states - 1)
    return start_prob, trans_matrix


# def compare(samples_states, samples_means, samples_covars, test_states, index):
#     scores = []
#     for i in range(len(samples)):
#         if i == index or i == index + 1:
#             scores.append(float('-inf'))
#             continue
#         states = samples_states[i]
#         means = samples_means[i]
#         covars = samples_covars[i]
#
#         num_of_states, number_of_features = states.shape
#         print(f'states shape = {states.shape}')
#         print(f'n states = {num_of_states}')
#         model = hmm.GaussianHMM(n_components=num_of_states, covariance_type="diag")
#         model.startprob_, model.transmat_ = get_probabilities(num_of_states)
#         model.means_ = np.array(means)
#         model.covars_ = np.array(covars)
#         scores.append(model.score(test_states))
#     return scores


def compare(test_states, index):
    return [models[i].score(test_states) if (i != index and 1 != index + 1) else float('-inf') for i in
            range(len(samples))]


def train_models(samples_states, samples_means, samples_covars):
    for i in range(len(samples_states)):
        states = samples_states[i]
        means = samples_means[i]
        covars = samples_covars[i]

        num_of_states, number_of_features = states.shape
        model = hmm.GaussianHMM(n_components=num_of_states, covariance_type="diag")
        model.startprob_, model.transmat_ = get_probabilities(num_of_states)
        model.means_ = np.array(means)
        model.covars_ = np.array(covars)
        models.append(model)


if __name__ == "__main__":
    n_samples = 100
    n_train_samples = 50
    types = ['circling', 'approach', 'random', 'chase']
    samples = []
    for t in types:
        for x in range(n_samples):
            samples.append(f'../datafiles/feature/{t}_{x}_feature.csv')

    samples_states = []
    samples_means = []
    samples_covars = []
    for file in samples:
        means = []
        covars = []
        states, means, covars = read_coordinate(file)
        samples_states.append(states)
        samples_means.append(means)
        samples_covars.append(covars)

    all = [[t] * n_samples for t in types]
    correct = 0
    incorrect = 0
    classification = [[0] * len(types) for x in types]

    a = time.time()
    print('training samples')
    train_models(samples_states, samples_means, samples_covars)
    b = time.time()
    print(f'execution time: {b - a}')
    a = time.time()
    print(len(models))

    for index, test_states in enumerate(samples_states):
        if index % 10 == 0:
            print(f'{index}/{len(samples_states)}')

        scores = compare(test_states, index)
        max_index = scores.index(max(scores))

        this_type = ((samples[index].split('/'))[-1].split('_'))[0]
        predicted = ((samples[max_index].split('/'))[-1].split('_'))[0]

        classification[types.index(this_type)][types.index(predicted)] += 1
        if predicted == this_type:
            correct = correct + 1
        else:
            incorrect = incorrect + 1

        print(f'predicted = {predicted}, actual = {this_type}, correct/incorrect = {correct}/{incorrect}')

    print(f"correct, incorrect: {correct} {incorrect}")
    print(f'accuracy: {correct / (correct + incorrect)}')
    print(f'types: {types}')
    for i, c in enumerate(classification):
        print(f"class {types[i]} predictions: {c}")

    b = time.time()
    print(f'execution time: {b - a}')


#     n_points = 50
#     point_size = 45
# correct, incorrect: 840 160
# accuracy: 0.84
# types: ['circling', 'approach', 'random', 'chase']
# class circling predictions: [237, 1, 12, 0]
# class approach predictions: [16, 182, 49, 3]
# class random predictions: [15, 28, 184, 23]
# class chase predictions: [0, 3, 10, 237]
# execution time: 4004.0299258232117

#     n_points = 50
#     point_size = 25
# correct, incorrect: 678 322
# accuracy: 0.678
# types: ['circling', 'approach', 'random', 'chase']
# class circling predictions: [184, 16, 32, 18]
# class approach predictions: [22, 167, 49, 12]
# class random predictions: [33, 68, 114, 35]
# class chase predictions: [1, 25, 11, 213]
# execution time: 3499.4781999588013




# noisy data

# predicted = circling, actual = chase, correct/incorrect = 1213/786
# predicted = chase, actual = chase, correct/incorrect = 1214/786
# correct, incorrect: 1214 786
# accuracy: 0.607
# types: ['circling', 'approach', 'random', 'chase']
# class circling predictions: [336, 47, 59, 58]
# class approach predictions: [66, 279, 103, 52]
# class random predictions: [94, 140, 191, 75]
# class chase predictions: [15, 40, 37, 408]
# execution time: 11530.099460363388
