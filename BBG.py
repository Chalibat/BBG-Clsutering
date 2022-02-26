from matplotlib import pyplot as plt
import numpy as np
import time


def get_BBG_instance(n: int, k: int, bad_point: int):
    print(f'[INFO] generating {n} points in {k} clusters with {bad_point} bad points')
    np.random.seed(int(time.time()))
    points_x = np.random.rand(n)
    points_y = np.random.rand(n)
    points = np.array((points_x, points_y)).T

    offset_x = np.arange(k)
    offset_y = np.array([3*i % k for i in range(k)])    # damit die cluster auch well separated sind

    cluster_size = np.random.randint(n/k - k/3 + 1, n/k + k/3, size=k-1)
    cluster_size = np.append(cluster_size, n - cluster_size.sum())

    offset_x = np.repeat(offset_x, cluster_size)
    offset_y = np.repeat(offset_y, cluster_size)
    offset = np.array((offset_x, offset_y)).T

    points += offset

    for i in range(bad_point):
        points[np.random.randint(0, n)] = np.random.rand(2) * k

    return points, cluster_size


def calculate_opt(points: np.array, cluster_size, dist_func = None):
    if dist_func is None:
        dist_func = lambda x, y: np.sqrt(np.square(x[0] - y[0]) + np.square(x[1] - y[1]))

    centers = []
    start_index, end_index = 0, cluster_size[0]
    for i, c in enumerate(cluster_size[1:]):
        centers.append(points[start_index:end_index].mean(axis=0))
        start_index = end_index
        end_index += c
    centers.append(points[start_index:end_index].mean(axis=0))

    opt = 0
    for i, p in enumerate(points):
        min = dist_func(p, centers[0])
        for j, c in enumerate(centers[1:]):
            d = dist_func(p, c)
            if d < min:
                min = d
        if min > 1:
            print(f'{min}  {p[0]}  {p[1]}')
        opt += min

    print(f'[INFO] the optimal value of the {len(points)} points clustered into the {len(cluster_size)} given clusters is {opt}')
    return opt


class BBG:
    def __init__(self, points: np.array, k: int, opt: float, alp: float, eps: float, div: float, dist_func=None):
        self.k = k
        self.div = div
        self.opt = opt
        self.alp = alp
        self.eps = eps
        self.n = len(points)
        self.points = points
        self.k_mean = [] #für die optimumssuche
        self.edges = []     #adjazensliste
        self.clusters = []  #liste an liste von punktindices die zu cluster gehören

        if dist_func is None:
            self.dist_func = lambda x, y: np.sqrt(np.square(x[0] - y[0]) + np.square(x[1] - y[1]))
        else:
            self.dist_func = dist_func

    def cluster(self):
        if self.opt < 1:
            self.find_opt()
        self.step1()
        self.step2()
        self.step3()

    def find_opt(self):
        self.k_mean = k_means(self.k, self.points, self.dist_func)
        opt = 0

        clus = [0 for i in range(len(self.points))]
        for i, c in enumerate(self.k_mean):
            for j, poi in enumerate(c):
                clus[poi] = i

        for i, p in enumerate(self.points):
            # get distances to all points
            distances = [[] for _ in range(self.k)]
            for j, q in enumerate(self.points):
                distances[clus[j]].append(self.dist_func(p, q))
            # get minimum median distance
            mini = 10000
            for k in range(self.k):
                distances[k].sort()
                if len(distances[k]) > 1:
                    median = distances[k][len(distances[k]) // 2]
                if median < mini:
                    mini = median
            opt += mini

        self.opt = opt

    def step1(self):    # make edge if points are close
        tau = (self.opt * 2 * self.alp) / (self.n * 5 * self.eps)
        print(f'[INFO] working in step 1, tau={tau}')

        self.edges = [[] for _ in range(self.n)]

        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points[i + 1:]):
                if self.dist_func(p1, p2) < tau:
                    self.edges[i].append(i + j + 1)
                    self.edges[i + j + 1].append(i)

    def step2(self):    # prune edges
        new_edges = [[] for _ in range(len(self.edges))]

        b = self.eps * self.n * (1 + 5 / self.alp) / self.div
        print(f'[INFO] working in step 2, b={b}')

        for i, e in enumerate(self.edges):
            for j, n in enumerate(e):
                if len(set(e).intersection(set(self.edges[n]))) >= b:
                    new_edges[i].append(n)

        self.edges = new_edges

    def step3(self):    # find clusters and than reasign points -> no path between good points in diff clusters -> breadth first search
        print(f'[INFO] working in step 3')
        b = self.eps * self.n * (1 + 5 / self.alp) / self.div
        cluster = []
        bad_points = []
        available_points = [i for i in range(self.n)]

        i = 0
        while i < self.k:
            if len(available_points) == 0:
                break
            c_i = [available_points[0]]
            for j in c_i:
                c_i.extend(list(set(self.edges[j]).difference(set(c_i))))
            if len(c_i) > b:
                cluster.append(c_i)
                i += 1
            else:
                bad_points.extend(c_i)
            available_points = list(set(available_points).difference(set(c_i)))

        if len(bad_points) > 0:
            print(f'[INFO] found {len(bad_points)} bad points')
            cluster[0].extend(bad_points)

        new_cluster = [[] for _ in range(len(cluster))]
        for i, p in enumerate(self.points):
            min_median_dist = np.Inf
            min_median_cluster = -1
            for j, c in enumerate(cluster):
                distances = []
                for k, q in enumerate(c):
                    distances.append(self.dist_func(p, self.points[q]))
                distances.sort()
                if distances[len(distances)//2] < min_median_dist:
                    min_median_dist = distances[len(distances)//2]
                    min_median_cluster = j
            new_cluster[min_median_cluster].append(i)

        self.clusters = new_cluster

def plot(points, cluster, target):
    x = points.T[0]
    y = points.T[1]

    colors = ['yellow', 'orange', 'red', 'green', 'lime', 'blue', 'aqua', 'peru', 'silver', 'purple', 'black']
    found_colors = [0 for _ in range(len(points))]
    for i, c in enumerate(cluster):
        for j in c:
            found_colors[j] = colors[i if i < 10 else 10]

    target_colors = [0 for _ in range(len(points))]
    for i, c in enumerate(target):
        for j in c:
            target_colors[j] = colors[i if i < 10 else 10]

    plt.subplots(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=found_colors)
    plt.title(f'found')
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=target_colors)
    plt.title(f'target')
    plt.show()


def k_means(k:int, p:np.array, dist_func=None):
    print(f'[INFO] working in k-means clustering')
    if dist_func is None:
        dist_func = lambda x, y: np.sqrt(np.square(x[0] - y[0]) + np.square(x[1] - y[1]))

    means = [p[np.random.randint(0, len(p))] for _ in range(k)]
    centers = [p[np.random.randint(0, len(p))] for _ in range(k)]

    for i in range(20):
        # new clustering
        clustering = [[] for _ in range(k)]
        for i, c in enumerate(p):
            mini, min_c = dist_func(c, centers[0]), 0
            for j in range(1, k):
                if dist_func(c, centers[j]) < mini:
                    mini = dist_func(c, centers[j])
                    min_c = j
            clustering[min_c].append(i)

        # new centers
        for i in range(k):
            means[i] = np.array([0, 0])
        cnt = [len(clustering[i]) for i in range(k)]
        for i, c in enumerate(clustering):
            for j, cc in enumerate(c):
                means[i] = means[i] + p[cc]
        for i in range(k):
            centers[i] = means[i] / max(cnt[i], 1)

    return clustering


# Werte angepasst für die Cluster-Erstellung
n = 1000
k = 10
bad = 50

alp, eps, div = .95, .1, 14
if n == 1000:
    div = 8

p, target = get_BBG_instance(n, k, bad)
opt = calculate_opt(p, target)

c = BBG(points=p, k=k, opt=opt, alp=alp, eps=eps, div=div)
c.cluster()

km = k_means(k, p)

targ_clus = []
num = 0
for i in target:
    targ_clus.append([])
    for j in range(i):
        targ_clus[-1].append(num)
        num += 1

plot(c.points, c.clusters, targ_clus)
plot(c.points, c.clusters, km)
