import numpy as np
import matplotlib.pyplot as plt
import time
import random
import multiprocessing
from operator import itemgetter


'''
generates synthetic dataset of boat paths to '../datafiles/csv_files/' using multiprocessing
'''


def scale_points(points, scale):
    return [(int(scale * x), int(scale * y)) for x, y in points]


def shift_points(points, shift):
    return [(x + shift[0], y + shift[1]) for x, y in points]


def move_circle_trajectory(trajectory, domain=(1024, 1024), c1=1):
    """
    shifts trajectory points and scales them down if needed to keep all values positive and under 1024
    :param trajectory: 2d list of points
    :param domain: defines the max value of the points
    :param c1: coefficient for scaling points
    :return: new trajectory
    """
    # uncomment this to ensure all circling points are in the domain, (boat always seen)
    xs = [x[0] for x in trajectory]
    ys = [x[1] for x in trajectory]
    _max = max([max(xs), max(ys)])
    _min = min([min(xs), min(ys)])
    spread = _max - _min
    c2 = min([0.95 * max(domain) / spread, 1])

    scaled_points = scale_points(trajectory, c1 * c2)
    scaled_points = shift_points(scaled_points, (domain[0] // 2, domain[1] // 2))
    scaled_points = remove_bad_points(scaled_points, domain=domain)
    return scaled_points


def sort_by_distance_to_end(current_position, displacement_vectors, ending_position):
    # sorts list of moves in increasing order of the distance to the ending position of the
    # position gained from applying the move to the current position
    move_magnitude = []
    for move in displacement_vectors:
        position = (current_position[0] + move[0], current_position[1] + move[1])
        distance_vector = (ending_position[0] - position[0], ending_position[1] - position[1])
        distance_to_end = (distance_vector[0] ** 2 + distance_vector[1] ** 2) ** 0.5
        move_magnitude.append((move, distance_to_end))
    sorted_moves_plus_mag = sorted(move_magnitude, key=itemgetter(1))
    return [x[0] for x in sorted_moves_plus_mag]


def sort_by_distance_to_vector(target_vector, displacement_vectors):
    # sorts list of moves in increasing order of the distance to the target vector which
    # characterizes circular motion about the origin
    mag = sum(a ** 2 for a in target_vector) ** .5
    target_vector = tuple(a / (mag+0.0001) for a in target_vector)
    move_magnitude = []
    for move in displacement_vectors:
        move_mag = sum(a ** 2 for a in move) ** 0.5
        if move_mag == 0:
            normalized_displacement = move
        else:
            normalized_displacement = tuple(a / move_mag for a in move)

        distance_vector = tuple(target_vector[i] - normalized_displacement[i] for i in range(2))
        distance = sum(a ** 2 for a in distance_vector) ** 0.5
        move_magnitude.append((move, distance))
    sorted_moves_plus_mag = sorted(move_magnitude, key=itemgetter(1))
    return [x[0] for x in sorted_moves_plus_mag]



def generate_circling_points(domain, prob_distribution, start=None, cluster_size=25, n_clusters=50, scale=True):
    """
    creates a list of points of a "circling boat" moving around the "victim boat". points are first generated
    on a grid of positive and negative numbers with the origin as the center of the circle for simplicity then points
    are scaled and shifted so that the center is the center of the domain, (where the ship is).
    :param domain: 2d list or tuple that are the max x and y coordinates.
    :param prob_distribution: a list of ten non-negative integers, the first 9 sum up to 100, the last entry is
    less than 100. The 1st entry in prob_distribution refers to the probability the chasing boat moves to the grid
    position that is closest to the tangent vector to the victim boat, the 2nd entry refers to the probability
    the chasing boat moves towards the 2nd closest grid position that is the 2nd closest to the tangent vector
    to the victim boat and so on... the last entry of prob_distribution refers to to the probability the chasing boat disappears from view
    :param start: starting position of trajectory
    :param cluster_size: number of moves per cluster
    :param n_clusters: number of points in final trajectory
    :param scale: boolean to scale the final trajectory by a random constant
    :return: circling trajectory points
    """
    rotation_direction = random.choice(["CW", "CCW"])
    possible_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)]
    min_domain = min(domain)
    while True:
        if start is None:
            start_domain = [x for x in range(-min_domain, min_domain)]
            current_pos = [random.choice(start_domain), random.choice(start_domain)]
        else:
            current_pos = start

        clustered_trajectory = []
        for k in range(n_clusters):
            for j in range(cluster_size):
                if rotation_direction == "CW":
                    target_vector = (current_pos[1], -1 * current_pos[0])
                else:
                    target_vector = (-1 * current_pos[1], current_pos[0])
                sorted_moves = sort_by_distance_to_vector(target_vector, possible_moves)
                weighted_moves = []  # we randomly select our next move from here
                for i in range(len(sorted_moves)):
                    for l in range(prob_distribution[i]):
                        weighted_moves.append(sorted_moves[i])
                choice = random.choice(weighted_moves)
                current_pos[0] += choice[0]
                current_pos[1] += choice[1]
            clustered_trajectory.append(tuple(current_pos))

        if scale:
            rand_coef = random.uniform(0.5, 1.2)
            clustered_trajectory = move_circle_trajectory(clustered_trajectory, c1=rand_coef)
            clustered_trajectory = remove_bad_points(clustered_trajectory, domain=domain)
        if len(clustered_trajectory) > n_clusters - (n_clusters//5):
            break
    return clustered_trajectory



def generate_approaching_points(domain, prob_distribution, axes_padding=(50, 50), start=None, cluster_size=25, n_clusters=50):
    """
    generate a list of points of a that are approaching from one side of the grid towards the "victim boat"
    see generate_circling_points() for description of parameters
    :return: list of points of length n_points of the trajectory
    """
    center = (domain[0] // 2, domain[1] // 2)
    starting_side = random.choice(["down", "left", "right"])

    if start is None:
        if starting_side == "down":
            x_ = random.choice([x for x in range(axes_padding[0], domain[0] - axes_padding[0])])
            starting_pos = (x_, axes_padding[1])
        if starting_side == "left":
            y_ = random.choice([x for x in range(axes_padding[1], domain[1] - axes_padding[1])])
            starting_pos = (axes_padding[0], y_)
        if starting_side == "right":
            y_ = random.choice([x for x in range(axes_padding[1], domain[1] - axes_padding[1])])
            starting_pos = (domain[0] - axes_padding[0], y_)
    else:
        starting_pos = start

    current_pos = list(starting_pos)
    possible_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)]

    clustered_trajectory = []
    for k in range(n_clusters):
        for j in range(cluster_size):
            sorted_moves = sort_by_distance_to_end(current_pos, possible_moves, center)
            weighted_moves = []  # we randomly select our next move from here
            for i in range(len(sorted_moves)):
                for l in range(prob_distribution[i]):
                    weighted_moves.append(sorted_moves[i])
            choice = random.choice(weighted_moves)
            current_pos[0] += choice[0]
            current_pos[1] += choice[1]
        clustered_trajectory.append(tuple(current_pos))
    clustered_trajectory = remove_bad_points(clustered_trajectory, domain=domain)
    # double check the path is valid
    if not verify_points(clustered_trajectory):
        assert 0, 'verification failed'
    return clustered_trajectory



def generate_chase_points(domain, prob_distribution, axes_padding=(50, 50), cluster_size=25, n_clusters=50):
    """
    generate a list of points of a "chase boat" moving from one side of the grid towards the "victim boat"
    see generate_circling_points() for description of parameters
    :return: list of points of length n_points of the trajectory
    """
    center = (domain[0] // 2, domain[1] // 2)
    x_ = random.choice([x for x in range(center[0] // 2, domain[0] - center[0] // 2)])
    starting_pos = (x_, domain[1] - axes_padding[1])

    current_pos = list(starting_pos)
    possible_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)]

    # we now create the trajectory
    clustered_trajectory = []
    for k in range(n_clusters):
        current_cluster = []
        for j in range(cluster_size):
            sorted_moves = sort_by_distance_to_end(current_pos, possible_moves, center)
            weighted_moves = []  # we randomly select our next move from here
            for i in range(len(sorted_moves)):
                for l in range(prob_distribution[i]):
                    weighted_moves.append(sorted_moves[i])
            choice = random.choice(weighted_moves)
            current_pos[0] += choice[0]
            current_pos[1] += choice[1]
        clustered_trajectory.append(tuple(current_pos))


    clustered_trajectory = remove_bad_points(clustered_trajectory, domain=domain)
    # double check the path is valid
    if not verify_points(clustered_trajectory):
        assert 0, 'verification failed'
    return clustered_trajectory


def generate_random_path_points(domain, prob_distribution, axes_padding=(50, 50), start=None, cluster_size=25, n_clusters=50):
    """
    generates random path on overhead that is a does not pass within a certain distance of the center
    see generate_circling_points() for decription of parameters
    :return:
    """
    ignore_dist_thresh = False
    center = (domain[0] // 2, domain[1] // 2)
    dist_thresh = 150
    while True:
        if start is None:
            starting_side = random.choice(["up", "down", "left", "right"])
            if starting_side == "up":
                x_ = random.choice([x for x in range(axes_padding[0], domain[0] - axes_padding[0])])
                starting_pos = (x_, axes_padding[1])
            if starting_side == "down":
                x_ = random.choice([x for x in range(axes_padding[0], domain[0] - axes_padding[0])])
                starting_pos = (x_, domain[1] - axes_padding[1])
            if starting_side == "left":
                y_ = random.choice([x for x in range(axes_padding[1], domain[1] - axes_padding[1])])
                starting_pos = (axes_padding[0], y_)
            if starting_side == "right":
                y_ = random.choice([x for x in range(axes_padding[1], domain[1] - axes_padding[1])])
                starting_pos = (domain[0] - axes_padding[0], y_)
        else:
            if ((center[0] - start[0])**2 + (center[1] - start[1])**2)**.5 < dist_thresh+50:
                ignore_dist_thresh = True
            starting_pos = start


        ending_side = random.choice(["up", "down", "left", "right"])
        if ending_side == "up":
            x_ = random.choice([x for x in range(axes_padding[0], domain[0] - axes_padding[0])])
            ending_pos = (x_, axes_padding[1])
        if ending_side == "down":
            x_ = random.choice([x for x in range(axes_padding[0], domain[0] - axes_padding[0])])
            ending_pos = (x_, domain[1] - axes_padding[1])
        if ending_side == "left":
            y_ = random.choice([x for x in range(axes_padding[1], domain[1] - axes_padding[1])])
            ending_pos = (axes_padding[0], y_)
        if ending_side == "right":
            y_ = random.choice([x for x in range(axes_padding[1], domain[1] - axes_padding[1])])
            ending_pos = (domain[0] - axes_padding[0], y_)

        if not ignore_dist_thresh and point_to_line_dist(starting_pos, ending_pos, center=center) > dist_thresh:
            break

    current_pos = list(starting_pos)
    possible_moves = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, 1), (-1, -1)]

    clustered_trajectory = []
    for k in range(n_clusters):
        for j in range(cluster_size):
            sorted_moves = sort_by_distance_to_end(current_pos, possible_moves, ending_pos)
            weighted_moves = []  # we randomly select our next move from here
            for i in range(len(sorted_moves)):
                for l in range(prob_distribution[i]):
                    weighted_moves.append(sorted_moves[i])
            choice = random.choice(weighted_moves)
            current_pos[0] += choice[0]
            current_pos[1] += choice[1]
        clustered_trajectory.append(tuple(current_pos))

    # there is no need to ensure that all points are valid since a boat could easily come in and out of the
    # domain on a random walk, but these points do need to be removed
    if not verify_points(clustered_trajectory, domain=domain):
        assert 0, 'verification failed'
    return clustered_trajectory


def verify_points(points, domain=(1024, 1024)):
    """
    verifies all points are in domain
    """
    for x, y in points:
        if not ((x >= 0 and x < domain[0]) and (y >= 0 and y < domain[1])):
            print(x, y)
            return False
    return True


def remove_bad_points(points, domain=(1024, 1024), exclude_region=None):
    """
    removes points not in domain or in excluded region if provided
    :param points: trajectory
    :param domain: max values for x and y
    :param exclude_region: [tl corner, br corner]. ex: [(495, 430), (530, 600)]
    :return: updated list of points
    """
    new_points = []
    for x, y in points:
        if (x >= 0 and x < domain[0]) and (y >= 0 and y < domain[1]):
            if exclude_region is not None:
                tl = exclude_region[0]
                br = exclude_region[1]
                if (x >= tl[0] and x <= br[0]) and (y >= tl[1] and y <= br[1]):
                    continue
            new_points.append((x, y))
    return new_points


def display_img_trajectory(trajectory, title='', domain=(1024, 1024)):
    img = np.zeros(shape=domain)
    # print(trajectory)
    # for a in range(0, 100):
    #     for b in range(0, 50):
    #         img[a, b] += 2

    for x, y in trajectory:
        img[x, y] = 1

    # make the ship
    for a in range(495, 530):
        for b in range(430, 600):
            img[a, b] += 2

    plt.imshow(img)
    plt.suptitle(title)
    plt.show()


def display_trajectory(trajectory, title='', domain=(1024, 1024), many=False):
    if not many:
        trajectory = [trajectory]
        title = [title]

    plts = []
    locs = [(0,0),(0,1),(1,0),(1,1)]
    s = (1,1)
    if many:
        s = (2,2)
    for i in range(len(trajectory)):
        x = []
        y = []
        for x_, y_ in trajectory[i]:
            x.append(x_)
            y.append(y_)

        plts.append(plt.subplot2grid(shape=s, loc=locs[i]))
        # mark the location of the ship
        plts[i].plot(domain[0] // 2, domain[1] // 2, 'o', color='red')
        plts[i].set_xlim([0, domain[0]])
        plts[i].set_ylim([0, domain[1]])

        plts[i].scatter(x, y, color='black', s=3)
        plts[i].grid()
        plts[i].set_title(title[i])
    plt.tight_layout()
    if many:
        plt.savefig('../datafiles/all_types.png')
    plt.show()


def write_to_csv(trajectory, filename):
    with open(filename, 'w') as f:
        for p in trajectory:
            f.write(f'{p[0]},{p[1]}\n')


def point_to_line_dist(p1, p2, center):
    """"
    gets the distance from the line connecting p1 and p2 and finds the distance
    the line is from the center.
    https://brilliant.org/wiki/dot-product-distance-between-point-and-a-line/
    """
    dy = (p2[1] - p1[1])
    dx = (p2[0] - p1[0] + 0.1)
    m = dy / dx
    b = p1[1] - m * p1[0]
    # -dy*x  + dx*y - dx*b = 0
    return abs(-dy * center[0] + dx * center[1] - dx * b) / ((dy ** 2 + dx ** 2) ** .5)



def sample_all():
    """
    creates examples of each activity type for debugging
    """
    a = time.time()
    # prob_dist = (20, 20, 20, 10, 10, 10, 5, 5, 0, 0) less noise
    # prob_dist = (20, 10, 10, 10, 10, 10, 10, 10, 5, 5) more noise
    prob_dist = (100, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    domain = (1024, 1024)
    n_samples = 1
    n_clusters = 45
    cs = 30
    all_ = []

    for i in range(n_samples):
        t = generate_approaching_points(domain=domain, axes_padding=(50, 50), prob_distribution=prob_dist, n_clusters=n_clusters, cluster_size=cs)
        display_trajectory(t, title=f'approaching path {i}')
        print(f'approaching points: {i}/{n_samples - 1}')
        if i == 0:
            all_.append(t)

    for i in range(n_samples):
        t = generate_chase_points(domain=domain, axes_padding=(50, 50), prob_distribution=prob_dist, n_clusters=n_clusters, cluster_size=cs)
        display_trajectory(t, title=f'chasing path {i}')
        print(f'chasing points: {i}/{n_samples - 1}')
        if i == 0:
            all_.append(t)

    for i in range(n_samples):
        print('here')
        t = generate_circling_points(domain=domain, prob_distribution=prob_dist, n_clusters=n_clusters, cluster_size=cs)
        print(t)
        display_trajectory(t, title=f'circling path{i}')
        print(f'circling points: {i}/{n_samples - 1}')
        if i == 0:
            all_.append(t)

    for i in range(n_samples):
        t = generate_random_path_points(domain=domain, axes_padding=(50, 50), prob_distribution=prob_dist, n_clusters=n_clusters, cluster_size=cs)
        display_trajectory(t, title=f'random {i}')
        print(f'random points: {i}/{n_samples - 1}')
        if i == 0:
            all_.append(t)

    types = ['approach', 'chase', 'circling', 'random']

    display_trajectory(all_, title=types, many=True)
    print('sdfa')
    b = time.time()
    print(f'execution time: {b - a}')


def generate_dataset(directory='../datafiles/csv_files/', samples_per_proc=10, thid=0, domain=(1024, 1024),
                     n_clusters=50, cluster_size=25, view_sample=False):
    """
    generates a dataset of synthetic activity paths of circling, chasing, and random walk
    :param directory: directory where all csvs will be saved
    :param domain: max x and y values
    :param samples_per_proc: number of samples of each process will create
    :param cluster_size: number of pixel level moves per cluster
    :param thid: thread (process) id
    :param n_clusters: number of clusters, (relevant only if clustered)
    :param view_sample: bool display a sample of the first of each activity
    :return: none
    """
    prob_dist = (20, 15, 15, 10, 10, 10, 10, 10, 0)
    # prob_dist = (20, 10, 10, 10, 10, 10, 10, 10, 10)  # more noise
    # prob_dist = (100, 0, 0, 0, 0, 0, 0, 0, 0, 0) # no noise

    print(thid)
    offset = thid * samples_per_proc
    meta_data = f'_{n_clusters}_{cluster_size}_'

    for i in range(offset, offset + samples_per_proc):
        s = 'approach'
        s = s + meta_data
        t = generate_approaching_points(domain=domain, axes_padding=(50,50), prob_distribution=prob_dist, cluster_size=cluster_size, n_clusters=n_clusters)
        if view_sample and i == 0:
            display_trajectory(t, title=f'{s} sample')
        write_to_csv(t, directory + f'{s}{i}.csv')
        print(f'{s} points: {i} in [{offset},  {offset + samples_per_proc})')

    for i in range(offset, offset + samples_per_proc):
        s = 'chase'
        s = s + meta_data
        t = generate_chase_points(domain=domain, axes_padding=(50,50), prob_distribution=prob_dist, cluster_size=cluster_size, n_clusters=n_clusters)
        if view_sample and i == 0:
            display_trajectory(t, title=f'{s} sample')
        write_to_csv(t, directory + f'{s}{i}.csv')
        print(f'{s} points: {i} in [{offset},  {offset + samples_per_proc})')

    for i in range(offset, offset + samples_per_proc):
        s = 'circling'
        s = s + meta_data
        t = generate_circling_points(domain=domain, prob_distribution=prob_dist,
                                     cluster_size=cluster_size, n_clusters=n_clusters)

        if view_sample and i == 0:
            display_trajectory(t, title=f'{s} sample')
        write_to_csv(t, directory + f'{s}{i}.csv')
        print(f'{s} points: {i} in [{offset},  {offset + samples_per_proc})')

    for i in range(offset, offset + samples_per_proc):
        s = 'random'
        s = s + meta_data
        t = generate_random_path_points(domain=domain, axes_padding=(50,50), prob_distribution=prob_dist, cluster_size=cluster_size, n_clusters=n_clusters)
        if view_sample and i == 0:
            display_trajectory(t, title=f'{s} sample')

        write_to_csv(t, directory + f'{s}{i}.csv')
        print(f'{s} points: {i} in [{offset},  {offset + samples_per_proc})')


def make_combined_trajectory():
    prob_dist = (20, 15, 15, 10, 10, 10, 10, 10, 0)
    domain = (1024, 1024)

    t = generate_circling_points(domain=(1024, 1024), prob_distribution=prob_dist, cluster_size=25, n_clusters=50)
    display_trajectory(t, title='1')
    print(len(t))

    t.extend(
        generate_approaching_points(domain=domain, axes_padding=(50, 50), prob_distribution=prob_dist, cluster_size=25,
                                    n_clusters=25, start=t[-1]))
    display_trajectory(t, title='2')
    print(len(t))

    t.extend(
        generate_random_path_points(domain=domain, prob_distribution=prob_dist, axes_padding=(50, 50), cluster_size=25,
                                    n_clusters=40, start=t[-1]))
    display_trajectory(t, title='3')
    print(len(t))

    t.extend(
        generate_approaching_points(domain=domain, axes_padding=(50, 50), prob_distribution=prob_dist, cluster_size=25,
                                    n_clusters=25, start=t[-1]))
    display_trajectory(t, title='4')
    print(len(t))

    write_to_csv(t, 'test_trajectory.csv')


# ----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    a = np.load('/home/george/PycharmProjects/real_path/overheadProjection/Experiments/Scene2bTake3/array.npy')
    display_trajectory(a, title='IPATCH Scene2bTake3')
    exit()


    # tag = f'_{n_clusters}_{cluster_size}_'

    # sample_all()
    # exit()

    sample_frequencies = [15, 25, 35, 45]
    n_processes = 5
    samples_per_proc = 100

    a = time.time()
    for cs in sample_frequencies:
        processes = []
        for i in range(n_processes):
            args_ = {'samples_per_proc': samples_per_proc, 'thid': i, 'cluster_size': cs, 'n_clusters': 50}
            x = multiprocessing.Process(target=generate_dataset, kwargs=args_)
            x.start()
            processes.append(x)
        for j in processes: j.join()

    b = time.time()
    first = b - a
    print(f'execution time: {first}')

#     test single proc time
#     a = time.time()
#     for i in range(n_processes*samples_per_job):
#         generate_dataset(samples_per_job=1, thid=i)
#     b = time.time()
#     second = b - a
#     print(f'execution time: {first} -- {second}')



