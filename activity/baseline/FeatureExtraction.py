import csv
import math
import numpy as np
import sklearn.cluster as cluster

'''
Feature extractor for Hidden Markov Models. Gets the features from tracks in '../datafiles/csv_files/'
and saves them in '../datafiles/feature/'. This directory may need to be created if it does not already exist.
'''

# export hardcoded center definition number to the top 
center = (0, 0)


# Reads in one csv,  first two columns assumes x,y per row

def get_angle(x_diff, y_diff, offset):
    angle = 0
    if x_diff == 0 or y_diff == 0:
        angle = 0
    elif x_diff > 0:
        angle = math.atan(y_diff / x_diff)
    elif x_diff < 0 and y_diff > 0:
        angle = math.atan(y_diff / x_diff) + math.pi
    elif x_diff < 0 and y_diff < 0:
        angle = math.atan(y_diff / x_diff) - math.pi

    if angle < 0:
        angle = angle + 2 * math.pi

    angle = (angle + offset) % (2 * math.pi)
    return angle


def get_quadrant(angle):
    if angle >= 0 and angle < math.pi / 2:
        return 1
    elif angle >= math.pi / 2 and angle < math.pi:
        return 2
    elif angle >= math.pi and angle < 3 * math.pi / 2:
        return 3
    elif angle >= 3 * math.pi / 2 and angle < 2 * math.pi:
        return 4
    else:
        return -1


def OPT(trajectory, angle_offset):
    # Top-down approach for dynamic programming to avoid recursion
    max_size = len(trajectory)
    quadrant = []
    memory = []
    for row in range(max_size):
        quadrant.append([])
        memory.append([])
        for col in range(max_size):
            quadrant[row].append(0)
            memory[row].append([0] * 5)

    for col in reversed(range(max_size)):
        for row in reversed(range(col)):
            x_diff = trajectory[col][0] - trajectory[row][0]
            y_diff = trajectory[col][1] - trajectory[row][1]
            if x_diff == 0 and y_diff == 0:
                quadrant[row][col] = 0
            else:
                angle = get_angle(x_diff, y_diff, angle_offset)
                quadrant[row][col] = get_quadrant(angle)

            the_quadrant = quadrant[row][col]
            if col == max_size - 1:
                for k in range(5):
                    if the_quadrant == 0 or k == 0 or (the_quadrant == k or abs(the_quadrant - k) % 2 > 0):
                        memory[row][col][k] = 1
                    else:
                        memory[row][col][k] = 0
            else:
                for k in range(5):
                    if the_quadrant == 0:
                        memory[row][col][k] = max(memory[col][col + 1][k] + 1, memory[row][col + 1][k])
                    elif k == 0 or (the_quadrant == k or abs(the_quadrant - k) % 2 > 0):
                        memory[row][col][k] = max(memory[col][col + 1][the_quadrant] + 1, memory[row][col + 1][k])
                    else:
                        memory[row][col][k] = memory[row][col + 1][k]

    path_index = get_path(quadrant, memory)
    path = []
    for index in path_index:
        path.append(trajectory[index])

    return path


def get_path(quadrant, memory):
    max_size = len(memory)
    path = []
    maximum = -1
    maximum_index = -1
    last_k = -1
    k = 0
    row = 0
    path.append(0)
    while row < max_size:
        for col in range(row + 1, max_size):
            if quadrant[row][col] == 0:
                maximum = maximum - 1
                path.append(col)
            elif memory[row][col][k] >= maximum:
                maximum = memory[row][col][k]
                maximum_index = col
                last_k = quadrant[row][col]
            else:
                k = last_k
                row = maximum_index
                maximum = maximum - 1
                path.append(maximum_index)
                break
        if maximum == 0:
            break
        if maximum_index >= max_size - 1:
            path.append(max_size - 1)
            break
    return path


def read_coordinates(file_name):
    coordinates = []
    with open(file_name, "r") as file:
        next(file)
        reader = csv.reader(file)
        for row in reader:
            try:
                coordinates.append([float(row[0]), float(row[1])])
            except Exception as err:
                print(err)
    return coordinates


def remove_duplicates(coordinates):
    angle_offset = 0
    distance_offset = 0
    trajectory = []
    minimum = float("inf")
    last = []
    time = 0
    start_pos = (-1, -1)
    for row in coordinates:
        try:
            if start_pos == (-1, -1):
                start_pos = (row[0], row[1])
            if len(last) == 0 or (row[0] != last[0] and row[1] != last[1]):
                x = row[0]
                y = row[1]
                trajectory.append([])
                trajectory[-1].append(x)
                trajectory[-1].append(y)
                trajectory[-1].append(time)
                last = row[0:2]
                distance_offset = math.sqrt((trajectory[-1][0] - center[0]) ** 2 + (trajectory[-1][1] - center[1]) ** 2)
                if distance_offset != 0 and distance_offset < minimum:
                    minimum = distance_offset
            time = time + 1
        except Exception as err:
            print(err)
    angle_offset = math.pi / 2 - get_angle(center[0] - start_pos[0], center[1] - start_pos[1], 0)
    distance_offset = minimum

    trajectory = OPT(trajectory, angle_offset)
    return trajectory, distance_offset, angle_offset


# symmetric is a parameter  ... increases the chances of matching

def features_extraction(file_name, type_, ind, symmetric=False):
    coordiantes = read_coordinates(file_name)
    trajectory, distance_offset, angle_offset = remove_duplicates(coordiantes)

    states = []
    for i in range(1, len(trajectory)):
        try:
            if trajectory[i][0:2] == trajectory[i - 1][0:2]:
                continue
            x_diff = trajectory[i][0] - trajectory[i - 1][0]
            y_diff = trajectory[i][1] - trajectory[i - 1][1]
            t_diff = trajectory[i][2] - trajectory[i - 1][2]

            distance = math.sqrt(
                (trajectory[i][0] - center[0]) ** 2 + (trajectory[i][1] - center[1]) ** 2) / distance_offset
            velocity = math.sqrt(x_diff ** 2 + y_diff ** 2) / t_diff

            # get angle ... and here computes the offset

            angle = get_angle(x_diff, y_diff, angle_offset)
            if symmetric:
                angle = (math.pi - angle) % (2 * math.pi)
            angle = math.atan(math.tan(angle))

            angular_velocity = angle / t_diff

            states.append([])
            states[-1].append(distance * 20)
            states[-1].append(velocity)
            states[-1].append(angle * 20)
            states[-1].append(angular_velocity * 10)
        except Exception as err:
            print(err)

    if symmetric:
        np.savetxt(f"../datafiles/feature/{type_}_{ind}_symmetric_feature.csv", states, delimiter=',',fmt ='%s',
                header="distance,velocity,angle,angular_velocity", comments="")
    else:
        np.savetxt(f"../datafiles/feature/{type_}_{ind}_feature.csv", states, delimiter=',',fmt ='%s',
                header="distance,velocity,angle,angular_velocity", comments="")



if __name__ == "__main__":
    n_samples = 500

    # the data set we are working on
    n_points = 50
    point_size = 35
    tag = f'_{n_points}_{point_size}_'

    types = ['circling', 'approach', 'random', 'chase']
    for t in types:
        print(t)
        for x in range(n_samples):
            features_extraction(f'../datafiles/csv_files/{t}{tag}{x}.csv', type_=t, ind=x)


