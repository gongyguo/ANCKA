import cupy as cp
from sklearn.utils import check_random_state
from scipy.linalg import LinAlgError
from Cudafunc import ker_discrete,ker_norm
import time
import config


def gdiscretize(vectors, max_svd_restarts=3, n_iter_max=50,random_state=None):
    
    random_state = check_random_state(random_state)

    eps = cp.finfo(float).eps
    n_samples, n_components = vectors.shape

    norm = cp.linalg.norm(vectors,axis=0)
    sign = cp.sign(vectors[0])
    ker_norm((n_samples,),(n_components,),(vectors, norm, sign, n_samples,n_components))

    vectors = vectors / cp.sqrt((vectors ** 2).sum(axis=1))[:, cp.newaxis]
    vectors[cp.isnan(vectors)]=0
    vectors[cp.isinf(vectors)]=0

    svd_restarts = 0
    has_converged = False

    while (svd_restarts < max_svd_restarts) and not has_converged:

        rotation = cp.identity(n_components)

        last_objective_value = 0.0
        n_iter = 0

        max_label = None
    
        while not has_converged:

            n_iter += 1
            vectors = vectors.dot(rotation)
            labels = vectors.argmax(axis=1)
            labels_count = cp.bincount(labels)

            # user defined kernel
            vectors_discrete = cp.zeros(shape=(n_samples, n_components),order="F")
            ker_discrete((n_samples,),(n_components,),(vectors_discrete, labels, labels_count, n_samples,n_components))
           
            t_svd = vectors_discrete.T.dot(vectors)
            
            try:
                U, S, Vh = cp.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print("SVD did not converge")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if ncut_value > 0:
                max_label = labels
            if ((abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max)):
                has_converged = True
            else:
                last_objective_value = ncut_value
                rotation = cp.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError('SVD did not converge')
    
    return max_label, vectors_discrete

