import numpy as np
from itertools import product


# define number of letters to permute
N = 4
factorial = np.math.factorial
# number of elements in S_N
ORD = factorial(N)


# useful function for calculating cycle types from partitions
# e.g.: one_hot(3, 5) -> [0, 0, 0, 1, 0]
def one_hot(x, size):
    z = np.zeros(size, dtype="i4")
    z[x] = 1
    return z

# find all partitions for given integer
def get_partitions(n):
    if n == 0:
        yield np.array([], dtype="i4")
    if n < 0:
        return
    for p in get_partitions(n-1):
        yield np.concatenate((p, np.array([1])))
        l = len(p)
        if l == 1 or (l > 1 and p[-1] < p[-2]):
            yield p + one_hot(l-1, l)

# translate partition into cycle type
def get_cycle_type(p, l=None):
    if not l:
        l = max(p)
    # cycle type of one cycle
    kvec = lambda lam: one_hot(lam-1, l)
    # cycle types of all cycles in a list
    ks = np.vectorize(kvec, signature="()->(n)")(p)
    # return sum of all cycle types
    return np.sum(ks, axis=0)

# get all possible cycle types for elements in S_n
def get_all_cycle_types(n, l=None):
    for p in get_partitions(n):
        yield get_cycle_type(p, l)

# find all repartitions of a given partition
def get_repartitions(p, k):
    # cycle types of repartition should have the same length as input k
    l = len(k)
    # calculate all the cycle types of repartitions of each row
    reparts = [get_all_cycle_types(lam, l) for lam in p]
    # calculate cartesian product of the sets of repartitions of rows to
    # get the repartitions of p
    reparts = np.array(list(product(*reparts)))
    # calculate the cycle types of the new repartitions
    reparts_ks = np.sum(reparts, axis=1)
    # accept only the repartitions with repart_k = k
    mask = np.product(reparts_ks == k, axis=1, dtype="bool")
    reparts = reparts[mask]
    # order along second axis of repartitions is reversed, but does not matter
    # because of the product over j in (3.68) in the script
    return reparts

def psi(p, k):
    # calculate entries of Psi matrix
    res = np.array(0., dtype="f4")
    for r in get_repartitions(p, k):
        k_factorial = np.array([factorial(k_i) for k_i in k])
        r_factorial = np.array([[factorial(r_ij) for r_ij in r_j]
                                for r_j in r])
        res += np.product(k_factorial)/np.product(r_factorial)
    return res


if __name__ == "__main__":
    # build Psi matrix. The cycle types k need to be padded to length N
    Psi = np.array([[psi(p, np.pad(k, (0, N-len(k))))
                    for k in get_all_cycle_types(N)]
                    for p in get_partitions(N)], dtype="i4")
    print("Psi:\n", Psi, "\n")

    # calculate order of conjugacy class labeled by cycle type k
    def ord_C(k):
        # order of stabilizer
        ord_stab = np.product(np.array([(i+1)**k_i * factorial(k_i)
                                        for (i, k_i) in enumerate(k)]))
        return ORD / ord_stab

    # build Sigma matrix from (3.52) in the script
    Sigma = np.diag(np.array([ord_C(k) / ORD
                              for k in get_all_cycle_types(N)]))
    print("Sigma:\n", Sigma, "\n")

    # calculate Psi * Sigma * Psi.T
    PSPT = np.around(Psi @ Sigma @ Psi.T).astype("i4")
    print("Psi * Sigma * Psi.T:\n", PSPT, "\n")


    # initialize K with zeros
    K = np.zeros(shape=(len(PSPT), len(PSPT))).tolist()

    def fill_K_column(A, n):
        if len(A) == 0:
            return
        # fill the n'th column of K, note that indices start at 0 here...
        K[n-1][n-1] = float(np.sqrt(A[n-1, n-1]))
        for i in range(n):
            K[i][n-1] = float(A[i, n-1]/K[n-1][n-1])

        # build B matrix
        K_arr = np.array(K)
        B = np.fromfunction(lambda i, j: K_arr[i, n-1]*K_arr[j, n-1],
                             shape=(n, n), dtype="i4")

        # build A tilde and recurse
        A_tilde = (A - B)[:-1, :-1]
        return fill_K_column(A_tilde, n-1)

    # start of recursion
    fill_K_column(PSPT, len(PSPT))

    K = np.around(np.array(K)).astype("i4")
    print("K:\n", K, "\n")


    X = np.around(np.linalg.inv(K) @ Psi).astype("i4")
    print("X:\n", X, "\n")

    II = np.around(X @ Sigma @ X.T).astype("i4")
    success = not np.max(II - np.identity(len(II)))
    print("X * Sigma * X.T:\n", II, "\n")
    print("Is X * Sigma * X.T == I (after rounding to integers)?:",
          success, "\n")

    if success:
        print("We calculated the character table succesfully!")
    else:
        print("It seems like something went wrong :(")

