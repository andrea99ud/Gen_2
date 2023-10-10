import random
import time

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB


def plot(data):
    figure, axis = plt.subplots(2, 3)

    axis[0][0].plot(data.cost)
    axis[0][0].set_title("cost transport")
    # axis[0].set_ylim((0, 2))

    axis[0][1].plot(data.time_child)
    axis[0][1].set_title("time to find a child")
    # axis[1].set_ylim((0, 0.01))

    axis[0][2].plot(data.time_model)
    axis[0][2].set_title("LP time")
    # axis[2].set_ylim((0, 0.5))

    axis[1][0].plot(data.n_children)
    axis[1][0].set_title("number sites")

    axis[1][1].boxplot(data.memory)
    axis[1][1].set_title("number of memory slot")

    axis[1][2].plot(np.array(data.cost[:-1]) - np.array(data.cost[1:]))
    axis[1][2].set_title("improvements")

    figure.set_size_inches(18.5, 10.5)

    plt.show()

    print(f"the final cost is {data.cost[-1]}")


def stats(data):
    print("obj_cost: ", data.cost[-1])
    print("average_memory: ", np.mean(data.memory))
    print("average_time_LP: ", np.mean(data.time_model))
    print("average_children_sampled: ", np.mean(data.children_sampled))


def create_array(orig_array, percentage):
    # Compute the number of unique rows we'll use
    num_rows = int(orig_array.shape[0] * percentage)

    # If num_rows is zero, we should at least select one row.
    num_rows = max(1, num_rows)

    # Randomly select 'num_rows' row indices without replacement
    unique_indices = np.random.choice(orig_array.shape[0], size=num_rows,
                                      replace=False)

    # Select the unique rows from the original array
    unique_rows = orig_array[unique_indices, :]

    # Now, we'll need to repeat these rows to fill the new array
    num_repeats = int(np.ceil(orig_array.shape[0] / float(num_rows)))

    # Repeat the unique rows to create a new array
    new_array = np.repeat(unique_rows, num_repeats, axis=0)

    # Make sure the new array has the same number of rows as the original array
    new_array = new_array[:orig_array.shape[0], :]

    return new_array


def initialize_omega(imgs, info, add_diag=False):
    # the north west rule is used to initialise omega
    # we start with the first pixel of each image
    N, dim, n_imgs = info['N'], info['dim'], info['n_imgs']
    indices = np.zeros(n_imgs)
    omega = np.array(indices.copy())
    b = np.array([img[0] for img in imgs])

    current_gamma = [np.min(b)]
    while np.min(indices) < N ** dim - 1:
        gamma = np.min(b)
        b -= gamma
        low = np.where(b < 1e-9)[0]
        indices[low] += 1
        indices = np.clip(indices, a_min=0, a_max=N ** dim - 1)
        b[low] = imgs[low, indices[low].astype('int')]

        omega = np.vstack((omega, indices.copy()))
        current_gamma.append(gamma)

    current_gamma = np.array(current_gamma)
    active_indices = np.array(omega).astype('int')
    if add_diag:
        current_gamma = np.concatenate((current_gamma, np.zeros(N ** dim)))
        active_indices = np.vstack((active_indices, np.repeat(
            np.arange(0, N ** dim)[:, np.newaxis], n_imgs, axis=1)))
    return current_gamma, active_indices


def initialize_model(imgs, info, current_gamma, active_indices, b_eq,
                     current_cost_vector, output=False, method=1):
    start = time.time()
    indices_row = np.array([])
    indices_col = np.array([])
    N, dim, n_imgs = info['N'], info['dim'], info['n_imgs']

    for i in range(n_imgs):
        for indices in range(N ** dim):
            gamma_indices = \
                np.where(active_indices.transpose()[i] == indices)[0]
            indices_row = np.concatenate((indices_row, gamma_indices))
            indices_col = np.concatenate((indices_col,
                                          np.ones(len(gamma_indices)) * (
                                                  indices + i * (
                                                  N ** dim))))

    A_eq = sp.csr_matrix(
        (np.ones(len(indices_col)), (indices_col, indices_row)), shape=(
            (N ** dim) * n_imgs, len(active_indices)))

    m = gp.Model("model")
    gamma = m.addMVar(shape=len(current_cost_vector),
                      vtype=GRB.CONTINUOUS)
    m.setObjective(current_cost_vector @ gamma, GRB.MINIMIZE)
    # print(len(self.current_cost_vector))
    m.Params.OutputFlag = output
    m.Params.Method = method

    m.Params.FeasibilityTol = 1e-8

    m.addConstr(A_eq @ gamma == b_eq)
    m.Params.LPWarmStart = 2
    gamma.PStart = current_gamma
    # self.constraints.DStart = self.current_kantorovich
    m.optimize()
    primal_solution = np.array(gamma.X)
    dual_solution = np.array(m.PI)
    time_model = time.time() - start
    cost = m.ObjVal
    return primal_solution, dual_solution, m, time_model, cost


def barycentric_distance(indices_list, info):
    # mean squared deviation from the classical barycenter of the xi
    # rescale x and y to be in [0,1]
    N, dim, n_imgs = info['N'], info['dim'], info['n_imgs']

    barycenter = np.sum(indices_list / N, axis=0) / n_imgs

    barycenter_cost = np.sum(
        [np.sum((x - barycenter) ** 2, axis=0) / n_imgs for x in
         indices_list / N], axis=0)
    return barycenter_cost


def get_cost(vector, info):
    N, dim, n_imgs = info['N'], info['dim'], info['n_imgs']

    # for each pair of active pixels, compute the cost of moving the first pixel to the second
    indices_list = []
    for i in range(n_imgs):
        indices_list.append(np.array(np.unravel_index(vector.transpose()[i],
                                                      tuple([N for i in
                                                             range(dim)]))))
    cost_vector = barycentric_distance(np.array(indices_list), N, n_imgs)
    return cost_vector


def compute_gain(cost, children, current_kantorovich, info):
    # compute the gain of moving the first pixel of each image to the second
    N, dim, n_imgs = info['N'], info['dim'], info['n_imgs']
    gain = np.sum([current_kantorovich[
                   i * (N ** dim):(i + 1) * (N ** dim)][
                       children.transpose()[i]] for i in
                   range(n_imgs)], axis=0) - cost
    return gain


def find_best_child(non_zero_indices, info, child_par, neighbours):
    N, dim, n_imgs = info['N'], info['dim'], info['n_imgs']

    turnover_rate = child_par['turnover_rate']
    parent_to_children = child_par['parent_to_children']
    random_selection = child_par['random_selection']
    parent_changed = child_par['parent_changed']
    add_neighbours = child_par['add_neighbours']

    parent = non_zero_indices.copy().transpose()

    parent = create_array(parent, parent_to_children)

    index = random.sample(range(0, n_imgs), parent_changed)
    parent[index] = np.random.randint(0, N ** dim,
                                      size=len(parent[1]))

    children = parent.copy().transpose()

    gain = compute_gain(get_cost(children, info), children, info)

    best_children = children[np.where(gain > 0)[0]]
    best_n = max(int(best_children.shape[0] * turnover_rate), 1)

    # now take the best turnover_rate% or just some random turnover_rate
    if not random_selection:
        # print(best_children.shape[0])

        best_children = children[np.argsort(gain)[-best_n:]]
        # print(best_children.shape[0])
    else:
        chosen_children = np.random.choice(best_children.shape[0],
                                           size=best_n, replace=False)
        best_children = best_children[chosen_children]

    if add_neighbours:
        for i in neighbours:
            parent[index] = np.clip(
                non_zero_indices.copy().transpose()[index] + i, 0, N ** 2 - 1)
            children = parent.copy().transpose()
            gain = compute_gain(get_cost(children, N, n_imgs, dim), children, N,
                                n_imgs, dim)
            proposed = children[np.where(gain > 0)[0]]
            best_n = max(int(proposed.shape[0] * turnover_rate), 1)
            best_children = np.vstack(
                (best_children, children[np.argsort(gain)[-best_n:]]))

    return best_children


def get_mean(par, imgs, N, n_imgs, active_indices, current_gamma):
    indices = np.array([[np.unravel_index(
        active_indices.transpose()[i], (N, N))[j] for i in
                         range(n_imgs)] for j in range(2)])
    indices = indices.transpose((1, 0, 2))
    mean = [np.sum([par[i] * indices[i][j] for i in range(n_imgs)],
                   axis=0).astype('int') for j in range(2)]
    mean = np.ravel_multi_index(mean, (N, N))
    gamma = sp.csr_matrix(
        (current_gamma, (active_indices.transpose()[0], mean)),
        shape=(N ** 2, N ** 2))
    return 1 - gamma.todense().transpose().dot(imgs[0]).reshape(N, N)


def get_neighbours(info, radius):
    N, dim, n_imgs = info['N'], info['dim'], info['n_imgs']
    neighbours = []

    for i in range(int(-N ** 2 / 2), int(N ** 2 / 2)):
        xx = np.sign(i) * i % N * np.sign(i)
        yy = int(i / N)
        mask = (xx) ** 2 + (yy) ** 2 < radius ** 2
        if not mask or i == 0:
            continue
        neighbours.append(i)

    return neighbours
