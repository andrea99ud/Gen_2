import utils.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
import scipy.sparse as sp
from gurobipy import GRB
import time
import random

class Genetic_Algorithm:
    def __init__(self, img_list):
        self.info = {}
        self.info['N'] = img_list[0].shape[0]
        self.info['dim'] = len(img_list[0].shape)

        self.imgs = np.array([img.flatten() for img in img_list])
        self.b_eq = np.concatenate(self.imgs)
        self.non_zero_indices = []
        self.info['n_imgs'] = len(self.imgs)


        # stats
        self.cost = []
        self.time_child = []
        self.children_sampled = []
        self.time_model = []
        self.len_children = []
        self.n_children = []
        self.memory = []

        self.current_gamma, self.active_indices = utils.initialize_omega(self.imgs, self.info, add_diag=True)
        self.current_cost_vector = utils.get_cost(self.active_indices, self.info)

        self.current_gamma, self.current_kantorovich, self.m, time_model, cost =\
            utils.initialize_model(self.imgs, self.info, self.current_gamma,
                                   self.active_indices, self.b_eq,
                                   self.current_cost_vector, output=False, method=1)

        self.time_model.append(time_model)
        self.cost.append(cost)

    def run(self, max_iter, max_sample, max_runtime, beta, turnover_rate,
            parent_to_children, random_selection, parent_changed,
            add_neighbours, radius):

        self.child_par = {}
        self.child_par['turnover_rate'] = turnover_rate
        self.child_par['parent_to_children'] = parent_to_children
        self.child_par['random_selection'] = random_selection
        self.child_par['parent_changed'] = parent_changed
        self.child_par['add_neighbours'] = add_neighbours
        self.child_par['radius'] = radius
        self.neighbours = utils.get_neighbours(self.info, radius)

        N, dim, n_imgs = self.info['N'], self.info['dim'], self.info['n_imgs']

        start_runtime = time.time()
        n_memory = 0
        for _ in range(max_iter):
            if time.time() - start_runtime > max_runtime:
                break
            start_iter = time.time()
            self.non_zero_indices = self.active_indices[
                np.nonzero(self.current_gamma)]

            # divide the non zero indices into chunks of size chunk_size
            best_children = utils.find_best_child(self.non_zero_indices,
                                                    self.info, self.child_par,
                                                    self.neighbours)

            self.children_sampled.append(best_children.shape[0])
            self.n_children.append(self.active_indices.shape[0])
            self.cost_children = utils.get_cost(best_children)

            self.active_indices = np.vstack(
                (self.active_indices, best_children))

            idx = best_children.copy().transpose()
            shift = np.array(
                [i * (N ** dim) for i in range(n_imgs)]).reshape(
                -1, 1)
            idx = (idx + shift).transpose()
            idx = idx.transpose()

            n_memory += 1

            self.time_child.append(time.time() - start_iter)

            if self.active_indices.shape[0] > int(
                    beta * ((N ** dim) * n_imgs)):
                self.memory.append(n_memory)
                n_memory = 0
                if beta > 3:
                    remove_value = beta - 1
                else:
                    remove_value = 1
                zero_indices = np.where(self.current_gamma == 0)[0][:int(
                    remove_value * (N ** dim) * n_imgs)]
                self.active_indices = np.delete(self.active_indices,
                                                zero_indices, axis=0)
                self.current_cost_vector = np.delete(self.current_cost_vector,
                                                     zero_indices)
                self.current_gamma = np.delete(self.current_gamma, zero_indices)

                self.current_gamma, self.current_kantorovich, self.m, time_model, cost =\
                    utils.initialize_model(self.imgs, self.info, self.current_gamma,
                                             self.active_indices, self.b_eq,
                                             self.current_cost_vector, output=False, method=1)
                self.time_model.append(time_model)
                self.cost.append(cost)
                self.len_children.append(self.active_indices.shape[0])

                continue

            constr = self.m.getConstrs()

            for i in range(len(best_children)):
                self.m.addVar(obj=self.cost_children[i], vtype=GRB.CONTINUOUS,
                              column=gp.Column([1] * n_imgs,
                                               [constr[j] for j in idx[i]]))

            self.m.optimize()

            self.runtime.append(self.m.Runtime)
            self.cost.append(self.m.ObjVal)
            #self.time_model.append(time.time() - start)
            self.current_gamma = np.array(
                self.m.getAttr("X", self.m.getVars())).copy()
            self.current_kantorovich = np.array(self.m.PI).copy()