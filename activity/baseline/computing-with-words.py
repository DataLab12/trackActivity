import csv
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

'''
this script will run activity detection with the computing with words method on trajectories in
 '../datafiles/csv_files' that were generated with the script '../radial/path_generator'.
'''


class ComputingWithWords:
    def __init__(self):
        # regions = ['b', 'c', 'd','Inv', 'A', Cir]
        self.n_regions = 4
        self.normal_regions = 'abcdefghi'
        self.special_regions = ['Inv', 'A']
        self.current_region = 'Inv'
        self.state = 'Default'
        self.last_states = []
        self.alert_flag = 'normal activity'
        self.trajectory_string = []


    def receive_points(self, points):
        """
        receives a trajectory of points creates the trajectory string to be processed by functions that detect activity
        :param points: trajectory path
        """
        for point in points:
            region, points = point[:2]
            if region != self.current_region:
                self.trajectory_string.append(region)
                self.trajectory_string.append(points)
                self.n_frames_in_region = 1
                self.current_region = region

            else:
                self.trajectory_string.append(points)
                self.n_frames_in_region += 1

        to_check = [self.check_circling, self.check_approaching_chasing_random]
        state_found = not (self.state=='Default')

        for func in to_check:
            if state_found:
                break
            state_found = func()


    def check_circling(self):
        """
        checks if the path is circling and returns true if it is.
        circling occurs when the trajectory passes through more quads than regions
        :return: boolean
        """
        q = set()
        r = set()
        current_region = ''
        for a in self.trajectory_string:
            if not (str(a) in self.normal_regions or str(a) in self.special_regions):
                if current_region not in self.special_regions:
                    q.add(a)
            else:  # a is a new region
                r.add(a)
                current_region = a

        if len(q) > len(r):
            self.state = 'circling'
            return True
        else:
            return False


    def check_approaching_chasing_random(self):
        """
        checks if conditions of approaching, chasing, or random path apply by going through
        the current trajectory string and returns a boolean if activity was detected.
        A path is approaching or chasing if it remains in the same quad and goes through most of the regions.
        A path is random it it goes through multiple quads and regions.
        :return: boolean if activity is detected
        """
        current_region = ''
        quads = set()
        regions = set()
        for a in self.trajectory_string:
            if str(a) in self.normal_regions or str(a) == 'A':
                current_region = a
                regions.add(a)
            else:
                if not (current_region in self.special_regions and 0 not in quads):
                    quads.add(a)

        if len(regions) >= self.n_regions - 2 and len(quads) == 1:
            if 0 in quads or 1 in quads:  # 0 and 1 are the quads approach points are in, chase in 2 and 3
                self.state = 'approach'
            else:
                self.state = 'chase'
            return True

        elif len(quads) > 1:
            self.state = 'random'
            return True

        # if no condition fits
        return False


    def clear(self):
        '''
        resets the variables of class object to process a new trajectory
        '''
        self.current_region = 'Inv'
        self.state = 'Default'
        self.last_states = []
        self.alert_flag = 'normal activity'
        self.trajectory_string = []
# end class: ComputingWithWords -------------------------


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


def generate_region_image(n_regions=3, width=120, domain=(1024, 1024), dynamic=False):
    """
    debug tool for generate_regions() that displays an image of the model
    """
    regions = list('Aabcdefghij')[:n_regions + 1]
    x_, y_ = domain
    a = np.zeros(shape=(x_, y_), dtype=np.uint8)
    for x in range(x_):
        for y in range(y_):
            try:
                r = decode_region((x, y), n_regions=n_regions, width=width, domain=domain, dynamic=dynamic)
                if dynamic: r.append(0)
                value = regions.index(r[0]) * n_regions + int(r[1])
                a[x, y] = value
            except Exception:
                # print(decode_region((x, y)))
                a[x, y] = 255

    plt.suptitle(f'n_regions={n_regions}   width={width}')
    plt.imshow(a)
    plt.savefig(f'../datafiles/CWW_{n_regions}_{width}.png')
    plt.show()


def decode_region(point, n_regions=4, width=75, domain=(1024, 1024), dynamic=True):
    """
    given a (x,y) point, find the region it belongs to based on the given parameters
    :param point: x,y point
    :return: region point belongs to
    """
    x, y = point
    if dynamic:
        regions = dynamically_generate_regions(n_regions=n_regions, width=width, domain=domain)
    else:
        regions = generate_regions()
    for reg in regions:
        tl = reg[0]
        br = reg[1]
        this_region = reg[2:]
        if (x >= tl[0] and x <= br[0]) and (y >= tl[1] and y <= br[1]):
            return this_region
    return ('Inv', 0)


@lru_cache(None)
def generate_regions():
    """
    basic CWW model defined as a list of sub-lists.
    each sub-list has 4 elements. First element is a tuple for the top right left of the region and
    the second is the same for the  bottom right. The third is the region char, a, b, etc, and 4th is the quadrant
    each region can have multiple elements specifying that region in the list
    keep in mind that if different regions overlap, the one that appears first in the list will be declared the
    region for that point
    intervals are closed
    :return: list containing 4 quadrants of 1 region.
    """
    a = [(0, 0), (511, 255), 'b', 2]
    b = [(0, 0), (255, 511), 'b', 2]

    c = [(512, 0), (1024, 255), 'b', 1]
    d = [(767, 0), (1024, 511), 'b', 1]

    e = [(767, 512), (1024, 1024), 'b', 4]
    f = [(512, 767), (1024, 1024), 'b', 4]

    g = [(0, 767), (511, 1024), 'b', 3]
    h = [(0, 512), (255, 1024), 'b', 3]

    # tmp fix, if its not in the previous regions then its in alrt
    alrt = [(0, 0), (1024, 1024), 'A', 0]
    return [a, b, c, d, e, f, g, h, alrt]


@lru_cache(1)
def dynamically_generate_regions(n_regions=3, width=120, domain=(1024, 1024)):
    """
    creates a model with multiple regions similar to generate_regions() with each region having a defined pixel width.
    the programmer calling this function is responsible of ensuring regions do not overlap.
    function decorator will cache the result since this only needs to be computed once.
    :return: list of regions, each in the form: (top_left, bottom_right, region_char, quadrant_num)
    """
    xmax, ymax = domain
    regions = list('abcdefghij')[:n_regions]
    w = width
    all_regions = []
    halfx = xmax // 2
    halfy = ymax // 2
    for i, reg in enumerate(regions):
        p = i * w
        a1 = [(p, p), (p + w, halfy), reg, 0]
        a2 = [(p, p), (halfx, p + w), reg, 0]
        b1 = [(halfx, p), (xmax - p, p + w), reg, 1]
        b2 = [(xmax - p - w, p), (xmax - p, halfy), reg, 1]
        c1 = [(xmax - p - w, halfy), (xmax - p, ymax - p), reg, 2]
        c2 = [(halfx, ymax - p - w), (xmax - p, ymax - p), reg, 2]
        d1 = [(p, ymax - p - w), (halfx, ymax - p), reg, 3]
        d2 = [(p, halfy), (p + w, ymax - p), reg, 3]
        all_regions.extend([a1, a2, b1, b2, c1, c2, d1, d2])
    alrt = [(w * n_regions, w * n_regions), (xmax - w * n_regions, ymax - w * n_regions), 'A', 0]
    all_regions.append(alrt)
    return all_regions


if __name__ == '__main__':
    types = ['circling', 'approach', 'random', 'chase']
    domain = (1024, 1024)

    # define the number of regions and their widths
    n_regions = 3
    width = 130
    dynamic = True

    # parameters of the data set we are working on
    n_points = 50
    point_size = 35
    tag = f'_{n_points}_{point_size}_'

    # show the regions in the overhead
    generate_region_image(n_regions=n_regions, width=width, domain=domain, dynamic=dynamic)

    tracker = ComputingWithWords()
    tracker.n_regions = n_regions
    predictions = {}

    n = 500
    correct = 0
    for _type in types:
        file_ = _type + tag
        predictions[_type] = []
        for i in range(n):
            tracker.clear()
            points = read_csv(f'../datafiles/csv_files/{file_}{i}.csv')
            trajectory = [decode_region(point=x, n_regions=n_regions, width=width, dynamic=dynamic) for x in points]
            tracker.receive_points(trajectory)
            predictions[_type].append(tracker.state)
            if _type == tracker.state:
                correct += 1


    print(f'number of times {types} were predicted for each class')
    for t in types:
        print(f'class {t} --- ', [predictions[t].count(tt) for tt in types])

    # show the confusion matrix
    for t in types:
        print([predictions[t].count(tt) for tt in types])

    for t in types:
        print(t, predictions[t])

    print('correct: ', correct)
    print(f'accuracy: {correct/(n*len(types))}')

