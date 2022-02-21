import time
from sklearn import svm
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
types = ['circling', 'approach', 'random', 'chase']

'''
This script contains code to create and test the Bayesian models for activity recognition of trajectories saved 
in '../datafiles/csv_files/'.

the most import components are as follows and each has its own descriptor explaining its function:
ActivityClassifier, train_single_model(), results()
'''


class ActivityClassifier:
    """
    class that wraps up the model in a way that can be saved and loaded for model persistence
    https://scikit-learn.org/stable/modules/model_persistence.html
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.classifier = naive_bayes.GaussianNB()
        # self.classifier = svm.SVC(kernel='poly', cache_size=7000, gamma='scale', degree=4, probability=True)
        self.classes = []

    def fit(self, X_, y_, classes):
        print(X_.shape)
        self.classifier = self.classifier.fit(X_, y_)
        self.classes = classes
        dump(self, self.file_path)

    def predict_sample(self, x):
        # returns the predicted result and its probability
        result = self.classes[self.classifier.predict([x])[0]]
        prob = max(self.classifier.predict_proba([x])[0])
        return [result, prob]



def to_polar(point, center=(512, 512)):
    """
    not true polar coordinates because angle is abs angle from o to 1 from the center
    :param point: single 2d point of the x,y position in the overhead
    :return:
    """
    centered_point = [point[i]-center[i] for i in range(2)]
    radius = np.linalg.norm(centered_point)+0.01
    norm_vec = centered_point/radius
    angle = np.arccos(np.dot(norm_vec, np.array([0, 1])))
    # return [radius, angle]
    return [radius, (np.pi-angle)]


def feature_extractor(trajectories):
    """
    gets the features used in classification by the model from a list of trajectories
    :param trajectories: array where each element is a trajectory
    :return: [radial_variance, angular_variance, radial_avg, angular_avg]*len(trajectories)
    """
    all_features = []
    for i, trajectory in enumerate(trajectories):
        features = []
        polar = [to_polar(point) for point in trajectory]
        vars = np.var(polar, axis=0)
        avgs = np.mean(polar, axis=0)
        features.append(vars[0])
        features.append(vars[1])
        features.append(avgs[0])
        features.append(avgs[1])

        f = np.array(features)
        if len(f[np.isnan(f)]) != 0:
            print('nan point removed ---------', f)
            continue
        all_features.append(features)
    return all_features



def write_to_csv(trajectory, filename):
    with open(filename, 'w') as f:
        for p in trajectory:
            line = [f'{x},' for x in p]
            line = ''.join(line)[:-1] + '\n'
            f.write(line)



def read_csv(fname):
    traj = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            values = []
            for cell in row:
                values.append(int(cell))
            traj.append(values)
    return traj



def plot_points(c, app, r,chase, title='', save_file=None):
    fig, axs = plt.subplots(ncols=2)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.suptitle(title)

    a = 0
    b = 1
    axs[0].scatter(c.T[a], c.T[b], color='red', s=1)
    axs[0].scatter(app.T[a], app.T[b], color='green', s=1)
    axs[0].scatter(r.T[a], r.T[b], color='blue', s=1)
    axs[0].scatter(chase.T[a], chase.T[b], color='black', s=1)
    axs[0].legend(types)
    axs[0].set_ylabel('angular variance')
    axs[0].set_xlabel('radial variance')

    a = 2
    b = 3
    axs[1].scatter(c.T[a], c.T[b], color='red', s=1)
    axs[1].scatter(app.T[a], app.T[b], color='green', s=1)
    axs[1].scatter(r.T[a], r.T[b], color='blue', s=1)
    axs[1].scatter(chase.T[a], chase.T[b], color='black', s=1)
    axs[1].legend(types)
    axs[1].set_ylabel('angular average')
    axs[1].set_xlabel('radial average')

    if save_file is not None:
        fig.savefig(save_file)
    fig.show()



def show_scores(scores, title='radial features model accuracy'):
    num_clusters = [10, 20, 30, 40, 50]
    cluster_sizes = [15, 25, 35, 45]
    print(scores.T[2])
    extent = [min(num_clusters), max(num_clusters), min(cluster_sizes), max(cluster_sizes)]

    plt.imshow(scores.T[2], extent=extent)
    plt.xticks(num_clusters)
    plt.yticks(cluster_sizes)
    plt.suptitle(title)
    plt.xlabel('number of points')
    plt.ylabel('sample frequency')
    plt.colorbar()
    plt.show()


def read_data(base_dir='../datafiles/csv_files/', max_len=50, cluster_size=25, min_=0, max_=500):
    meta_data = f'_{50}_{cluster_size}_'
    data_set = []
    for n, _type in enumerate(types):
        _type = _type + meta_data
        this_ds = []
        for i in range(min_, max_):
            # print(_type, i)
            points = read_csv(f'{base_dir}{_type}{i}.csv')
            if len(points) < 5:
                continue
            this_ds.append(points[:max_len])
        feats = np.array(feature_extractor(this_ds))
        data_set.append(feats)
    return np.array(data_set)



def compare_models():
    num_clusters = [10, 20, 30, 40, 50]
    cluster_sizes = [15, 25, 35, 45]

    # first verify all files exist, this will throw an error if not
    for nc in num_clusters:
        for cs in cluster_sizes:
            print(f'testing {nc}, {cs}')
            data_set = read_data(max_len=nc, cluster_size=cs)

    scores = []
    for nc in num_clusters:
        for cs in cluster_sizes:
            print('------------------------------------------------------------')
            print(f'training with {nc} clusters and {cs} cluster sizes')
            data_set = read_data(max_len=nc, cluster_size=cs)
            print('data_set shape: ', data_set.shape)
            fp = f'../datafiles/svc_models/svc_clusters_{nc}_clusterSize_{cs}.joblib'
            circling = data_set[0]
            approaching = data_set[1]
            random = data_set[2]
            chasing = data_set[3]

            l = min([len(x) for x in data_set])
            # plot_points(circling, approaching, random, chasing, title=f'num clusters={nc}, cluster size={cs}', save_file=fp[:-7]+'_fig.png')

            # organize data in form classifier wants
            X = np.append(circling[:l], approaching[:l], axis=0)
            X = np.append(X, random[:l], axis=0)
            X = np.append(X, chasing[:l], axis=0)

            # 0 for circling, 1 for approaching, 2 for random, 3 for chasing
            y = np.array([0] * l + [1] * l + [2] * l + [3] * l)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

            clf = ActivityClassifier(file_path=fp)
            clf.fit(X_train, y_train, types)
            test_score = clf.classifier.score(X_test, y_test)
            print('svc score test data: ', test_score)
            scores.append([nc, cs, test_score])
            print('------------------------------------------------------------')
    print("scores: ", scores)
    with open('../datafiles/model_scores.txt', 'w') as fff:
        fff.write('scores = ' + str(scores))

    scores = np.array(scores).reshape((len(num_clusters), len(cluster_sizes), 3))
    with open('../datafiles/model_scores.npy', 'wb') as fff:
        np.save(fff, scores)
    return scores



def train_single_model(cached=True, display=True, n_points=50, cluster_size=25, n_samples=500):
    '''
    trains a model with the given parameters
    :param cached: if True and a model with given parameters exists, it will only score the model
    :param display: show a scatter plot of the features of the trajectories for debugging
    :param n_points: path length
    :param cluster_size: sample frequency
    :param n_samples: number of samples to train on
    :return:
    '''
    a = time.time()
    array_name = f'../datafiles/svc_models/features_{n_points}_{cluster_size}_{n_samples}.nyp'
    if os.path.exists(array_name) and cached:
        with open(array_name, 'rb') as f:
            data_set = np.load(f, allow_pickle=True)
    else:
        data_set = read_data(max_len=n_points, cluster_size=cluster_size, max_=n_samples)
        with open(array_name, 'wb') as f:
            np.save(f, data_set)

    # data_set will be an array of arrays where each sub array is a array not of the same len of the rest
    # this is fixed later when data is placed in X and y
    print('data_set shape: ', data_set.shape)
    # print(data_set)

    # get the circling, approaching, and random data
    circling = data_set[0]
    approaching = data_set[1]
    random = data_set[2]
    chasing = data_set[3]

    l = min([len(x) for x in data_set])

    if display:
        plot_points(circling, approaching, random, chasing, title=f'features:  n_clusters={n_points}  cluster_size={cluster_size}', save_file='../experiments/all_feats.png')

    # organize data in form classifier wants
    X = np.append(circling[:l], approaching[:l], axis=0)
    X = np.append(X, random[:l], axis=0)
    X = np.append(X, chasing[:l], axis=0)

    # 0 for circling, 1 for approaching, 2 for random, 3 for chasing
    y = np.array([0]*l + [1]*l + [2]*l + [3]*l)

    for i, tp in enumerate(data_set):
        print(f"class: {types[i]}")
        print('avg radial variance ', np.average(tp.T[0]))
        print('avg angular variance ', np.average(tp.T[1]))
        print('avg radial avg ', np.average(tp.T[2]))
        print('avg angular avg ', np.average(tp.T[3]))
        print()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    print('len of each type:', l)
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)

    model_file = f'../datafiles/svc_models/svc_clusters_{n_points}_clusterSize_{cluster_size}.joblib'

    print('\ngetting svc')
    if os.path.exists(model_file) and cached:
        print('loading saved model')
        clf = load(model_file)
    else:
        print('model path does not exist, training new model')
        clf = ActivityClassifier(file_path=model_file)
        clf.fit(X_train, y_train, types)
    print('done.\n')

    print('score test data: ', clf.classifier.score(X_test, y_test))
    print('score circling: ', clf.classifier.score(circling, [0]*len(circling)))
    print('score approaching: ', clf.classifier.score(approaching, [1] * len(approaching)))
    print('score random: ', clf.classifier.score(random, [2] * len(random)))
    print('score chase: ', clf.classifier.score(chasing, [3] * l))
    b = time.time()
    t = b - a
    print(f'execution time: {t}')
    c, inc = 0, 0
    for i, t in enumerate(X_test):
        yy = clf.predict_sample(t)
        if yy[0] == types[y_test[i]]:
            c += 1
        else:
            inc += 1
    print(f'true score: {c/(c+inc)}')
    return clf



def results(cs=35, nump=50):
    """
    Gets the results and the confusion matrix of the model with the given parameters.
    Assumes model already exists
    :param cs: cluster size (sample frequency)
    :param nump: number of points (trajectory length)
    """
    data_set = read_data(max_len=nump, cluster_size=cs)
    print('data_set shape: ', data_set.shape)

    # get the circling, approaching, and random data
    circling = data_set[0]
    approaching = data_set[1]
    random = data_set[2]
    chasing = data_set[3]

    l = min([len(x) for x in data_set])

    # organize data in form classifier wants
    X = np.append(circling[:l], approaching[:l], axis=0)
    X = np.append(X, random[:l], axis=0)
    X = np.append(X, chasing[:l], axis=0)

    # 0 for circling, 1 for approaching, 2 for random, 3 for chasing
    y = np.array([0] * l + [1] * l + [2] * l + [3] * l)

    for i, tp in enumerate(data_set):
        print(f"class: {types[i]}")
        print('avg radial variance ', np.average(tp.T[0]))
        print('avg angular variance ', np.average(tp.T[1]))
        print('avg radial avg ', np.average(tp.T[2]))
        print('avg angular avg ', np.average(tp.T[3]))
        print()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    print('len of each type:', l)
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)

    model_file = f'../datafiles/svc_models/svc_clusters_{nump}_clusterSize_{cs}.joblib'
    clf = load(model_file)

    cir = list(clf.classifier.predict(circling))
    ap = list(clf.classifier.predict(approaching))
    ch = list(clf.classifier.predict(chasing))
    ran = list(clf.classifier.predict(random))

    print()

    a = 0
    t, f = cir.count(a), ap.count(a)+ran.count(a)+ch.count(a)
    print(f'predicted circling: correct = {t/(t+f)}, incorrect = {f/(t+f)}')

    a = 1
    t, f = ap.count(a), cir.count(a)+ran.count(a)+ch.count(a)
    print(f'predicted approaching: correct = {t/(t+f)}, incorrect = {f/(t+f)}')

    a = 2
    t, f = ran.count(a), ap.count(a)+cir.count(a)+ch.count(a)
    print(f'predicted random: correct = {t/(t+f)}, incorrect = {f/(t+f)}')

    a = 3
    t, f = ch.count(a), ap.count(a)+cir.count(a)+ran.count(a)
    print(f'predicted chase: correct = {t/(t+f)}, incorrect = {f/(t+f)}')
    print()

    m = [
        [cir.count(0), cir.count(1), cir.count(2), cir.count(3)],
        [ap.count(0), ap.count(1), ap.count(2), ap.count(3)],
        [ran.count(0), ran.count(1), ran.count(2), ran.count(3)],
        [ch.count(0), ch.count(1), ch.count(2), ch.count(3)],
    ]

    print('confusion matrix: ', m)
    print('score test data: ', clf.classifier.score(X_test, y_test))
    print('score circling: ', clf.classifier.score(circling, [0] * len(circling)))
    print('score approaching: ', clf.classifier.score(approaching, [1] * len(approaching)))
    print('score random: ', clf.classifier.score(random, [2] * len(random)))
    print('score chase: ', clf.classifier.score(chasing, [3] * l))



if __name__ == '__main__':
    # results(nump=50)
    # exit()
    # scores = compare_models()
    # scores = np.array(scores).reshape((5, 4, 3))
    # show_scores(scores, title='Radial Model Accuracy (moderate noise)')

    data = read_data()

    b = 25
    alpha = 0.6
    features = ['Radial Variance', 'Angular Variance', 'Radial Mean', 'Angular Mean']
    plts = []
    locs = [(0,0),(0,1),(1,0),(1,1)]
    s = (2,2)
    # plt.hist(data[type][:, feat], bins=b, alpha=alpha, label=types[type])

    for feat in range(4):
        plts.append(plt.subplot2grid(shape=s, loc=locs[feat]))
        for type in range(4):
            plts[feat].hist(data[type][:, feat], bins=b, alpha=alpha, label=types[type])
        plts[feat].set_title(features[feat])
        plts[feat].legend(loc='upper right', prop={'size': 6})
    plt.tight_layout()
    plt.show()
    exit()


    if not os.path.exists('../datafiles/svc_models/'):
        os.system('mkdir ../datafiles/svc_models')
    n_points = 50
    point_size = 45

    cclf = train_single_model(cached=False, n_points=n_points, cluster_size=point_size)
    # results(cs=point_size, nump=n_points)

    a = np.load('/home/george/PycharmProjects/real_path/overheadProjection/Experiments/Scene2bTake3/array.npy')

    feats = feature_extractor([a])
    print(feats)

    yy = cclf.predict_sample(feats[0])
    print(yy)















