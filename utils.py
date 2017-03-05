import time
import numpy as np


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
