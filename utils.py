import time
import numpy as np


class DisjointSet(object):
    """
    Disjoint-set data structure that supports union/find operations and do-not-merge constraints.
    """
    def __init__(self, values, neg_constraints=None):
        if isinstance(values, int) or (isinstance(values, float) and values % 1 == 0):
            values = range(values)
        self.values = set(values)
        self.parent = dict(zip(values, values))
        self.rank = dict(zip(values, [0] * len(values)))
        self.neg_constraints = neg_constraints

        # a dict with constraints indexed by each cluster
        if neg_constraints is not None:
            neg_constraints_dict = {}
            for v in self.values:
                neg_constraints_dict[v] = set()
            for (f, t) in self.neg_constraints:
                neg_constraints_dict[f].add(t)
                neg_constraints_dict[t].add(f)
            self.neg_constraints_dict = neg_constraints_dict

    def find(self, v):
        """Find the root of v.

        :param v: a node
        :return: the root node of v
        """
        p = self.parent[v]
        if p != v:
            self.parent[v] = self.find(p)
        return self.parent[v]

    def union(self, v1, v2):
        """Union the set that contains v1 and the set that contains v2.

        :param v1: node 1
        :param v2: node 2
        :return: bool, True if a union happens (the number of disjoint sets is reduced by 1)
        """
        rt1, rt2 = self.find(v1), self.find(v2)
        # v1 and v2 are already in the same cluster
        if rt1 == rt2:
            return False

        # v1 and v2 are in different clusters, but there is do-not-merge constraint
        if self.neg_constraints is not None and (min(rt1, rt2), max(rt1, rt2)) in self.neg_constraints:
            return False

        # v1 and v2 are in different clusters, merge them
        rk1, rk2 = self.rank[rt1], self.rank[rt2]
        if rk1 < rk2:
            self.parent[rt1] = rt2
            parent, child = rt2, rt1
        elif rk1 > rk2:
            self.parent[rt2] = rt1
            parent, child = rt1, rt2
        else:
            self.parent[rt2] = rt1
            self.rank[rt1] += 1
            parent, child = rt1, rt2

        # update constraints
        if self.neg_constraints is None:
            return True
        self.neg_constraints_dict[parent].update(self.neg_constraints_dict[child])
        for c in self.neg_constraints_dict[child]:
            self.neg_constraints_dict[c].remove(child)
            self.neg_constraints_dict[c].add(parent)
            self.neg_constraints.remove((min(c, child), max(c, child)))
            self.neg_constraints.add((min(c, parent), max(c, parent)))
        del self.neg_constraints_dict[child]

        return True

    def get_clusters(self):
        """List representation of the clusters.

        :return: a list of clusters, each contains its members in a list
        """
        clusters = {}
        for v in self.values:
            p = self.find(v)
            if p in clusters:
                clusters[p].append(v)
            else:
                clusters[p] = [v]
        return list(clusters.values())


class TimedBlock:
    """
    Context manager that times the execution of a block of code.
    """
    def __init__(self, msg='', verbose=False):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        if self.verbose and self.msg:
            print('{} ...'.format(self.msg), end='', flush=True)
        self.tic = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        toc = time.time()
        if self.verbose and self.msg:
            print(' done! ({:.2f} secs)'.format(toc - self.tic))


def load_ndarray(file_path, dtype=np.float32):
    """Load nd-array from a file."""
    if file_path.endswith('.mat'):
        try:
            from scipy.io import loadmat
            mat = loadmat(file_path)
            var_names = list(filter(lambda k: not k.startswith('__'), mat.keys()))
            if len(var_names) != 1:
                raise ValueError('There are {} variables in {}. 1 is expected.'.format(len(var_names), file_path))
            mat = mat[var_names[0]].astype(dtype=dtype)
        except NotImplementedError:
            import h5py
            f = h5py.File(file_path, 'r')
            var_names = list(f.keys())
            if len(var_names) != 1:
                raise ValueError('There are {} variables in {}. 1 is expected.'.format(len(var_names), file_path))
            mat = np.array(f[var_names[0]], dtype=dtype)
            mat = mat.transpose(range(mat.ndim-1, -1, -1))
    elif file_path.endswith('.npy'):
        mat = np.load(file_path).astype(dtype=dtype)
    else:
        raise ValueError('Cannot load data from this file type: {}.'.format(file_path))
    return mat


def save_ndarray(file_path, mat, var_name='A', oned_as='row'):
    """Save nd-array to a file."""
    if file_path.endswith('.mat'):
        from scipy.io import savemat
        savemat(file_path, {var_name: mat}, oned_as=oned_as)
    elif file_path.endswith('.npy'):
        np.save(file_path, mat)
    else:
        raise ValueError('Cannot write data to this file type: {}.'.format(file_path))
