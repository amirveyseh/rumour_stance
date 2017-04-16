import numpy as np
import scipy.sparse as sp
from itertools import izip


def spontaneousnode_count(
    infecting_vec,
    infected_vec,
    node_vec,
    D,
    ):

    spontaneous_nodes = node_vec[infecting_vec == infected_vec]
    updates = np.zeros((D, 1))
    for node in spontaneous_nodes:
        updates[int(node)] += 1
    return updates


def spontaneousmeme_count(
    infecting_vec,
    infected_vec,
    eventmemes,
    M,
    ):

    spontaneous_memes = eventmemes[infecting_vec == infected_vec]
    updates = np.zeros((M, 1))
    for meme in spontaneous_memes:
        updates[int(meme)] += 1
    return updates


def infecting_node(infected_vec, infecting_vec, node_vec):

    infecting_node_vec = []
    eventid_to_node = {}

    for (evid, inf_evid, nodeid) in izip(infected_vec, infecting_vec,
            node_vec):
        eventid_to_node[int(evid)] = nodeid
        infecting_node_vec.append(eventid_to_node[int(inf_evid)])
    infecting_node_vec = np.array(infecting_node_vec).flatten()
    return (infecting_node_vec, eventid_to_node)


def infections_count(
    infecting_node,
    infected_node,
    infecting_vec,
    infected_vec,
    D,
    ):

    infections_mat = sp.lil_matrix((D, D), dtype=np.int)
    for (infected_u, infecting_u, infected_e, infecting_e) in \
        izip(infected_node, infecting_node, infected_vec,
             infecting_vec):
        if infected_e != infecting_e:
            infections_mat[infecting_u, infected_u] += 1
    return infections_mat


def one_hot_sparse(index_array, num_values):
    m = sp.lil_matrix((num_values, index_array.shape[0]), dtype=np.bool)
    for i in range(index_array.shape[0]):
        m[index_array[i], i] = 1
    return m.tocsr()


def one_hot(index_array, num_values):
    m = np.zeros(shape=(num_values, index_array.shape[0]),
                 dtype=np.bool)
    m[index_array.astype(int), range(index_array.shape[0])] = 1
    return m


