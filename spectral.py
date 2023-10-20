import numpy as np
from sklearn.utils import check_random_state, as_float_array


def discretize(vectors, copy=True, max_svd_restarts=3, n_iter_max=50,
               random_state=None):

    from scipy.sparse import csc_matrix, csr_matrix
    from scipy.linalg import LinAlgError

    random_state = check_random_state(random_state)

    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) \
            * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]
    vectors[np.isnan(vectors)]=0
    vectors[np.isinf(vectors)]=0

    svd_restarts = 0
    has_converged = False

    while (svd_restarts < max_svd_restarts) and not has_converged:
        rotation = np.identity(n_components)

        last_objective_value = 0.0
        n_iter = 0

        max_label = None

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components)).toarray()
            
            vectors_f = vectors_discrete
            vectors_fs = np.sqrt(vectors_f.sum(axis=0))
            vectors_fs[vectors_fs==0]=1
            vectors_f = vectors_f*1.0/vectors_fs

            t_svd = vectors_f.T.dot(vectors)

            try:
                U, S, Vh = np.linalg.svd(t_svd)
                svd_restarts += 1
            except LinAlgError:
                print("SVD did not converge")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if ncut_value > 0:
                max_label = labels
            if ((abs(ncut_value - last_objective_value) < eps) or
                    (n_iter > n_iter_max)):
                has_converged = True
            else:
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError('SVD did not converge')

    return max_label, vectors_f
