import numpy as np
import math
from dataclasses import dataclass
from random import randint
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

# We follow Szabo and Ostlund page 146 for the SCF procedure.

@dataclass
class Coordinate:
    x1: float
    x2: float
    x3: float


def to_basis_element_index(i, j, k, K):
    """
    Converts grid location (i, j, k) to a basis index i K^2 + j K + k.
    :param i: The index of the position in x1.
    :param j: The index of the position in x2.
    :param k: The index of the position in x3.
    :param K: The number of divisions of x1, x2 and x3.
    :return: The index of the corresponding product delta function in the finite basis list.
    """
    return i * K ** 2 + j * K + k


def from_basis_element_index(n, K):
    """
    Converts a basis index n to a location (i, j, k).
    :param n: The basis index.
    :param K: The divisions of x1, x2 and x3.
    :return: The location i, j, k.
    """
    i = math.floor(n / (K ** 2))
    j = math.floor((n % (K ** 2)) / K)
    k = (n % (K ** 2)) % K
    return i, j, k


def get_element_kinetic_part(n, m):
    """
    Returns the matrix element at col m and row n of H_kinetic.
    :param n: The row.
    :param m: The column.
    :return: The corresponding matrix element.
    """

    if (n == m):
        return 3 / delta ** 2
    elif (m == n - 1) or (m == n + 1) or (m == n - K) \
            or (m == n + K) or (m == n - K ** 2) or (m == n + K ** 2):
        return -1 / (2 * delta ** 2)
    else:
        return 0


def get_element_nuclear_attraction(n, m, nuclear_coordinates, atom_nums):
    """
    Returns the matrix element at col m and row n of H_nuclear_attraction.
    :param n: The row.
    :param m: The column.
    :param nuclear_coordinates: The coordinates of our nuclei.
    :param atom_nums: The atomic numbers.
    :return: The corresponding matrix element.
    """
    assert len(atom_nums) == len(nuclear_coordinates)

    if n == m:
        i, j, k = from_basis_element_index(n, K)

        attractions = 0
        for c, coord in enumerate(nuclear_coordinates):
            attractions -= atom_nums[c] / abs(np.sqrt((i * delta - coord.x1) ** 2 +
                                                   (j * delta - coord.x2) ** 2 +
                                                   (k * delta - coord.x3) ** 2))

        return attractions

    else:
        return 0


def get_element_coulomb_term(n, m, K, psi, delta):
    """
    Calculates the Coulomb term.
    :param n: The row.
    :param m: The column
    :param K: The number of divisions.
    :param psi: The wavefunction.
    :param delta: The grid spacing.
    :return: The Coulomb term.
    """

    if n == m:

        sum = 0
        i, j, k = from_basis_element_index(n, K)

        for p in range(K ** 3):
            r, s, t = from_basis_element_index(p, K)

            sum += psi[n].conjugate() \
                   * (1 / abs(np.sqrt((i * delta - r * delta) ** 2 +
                                   (j * delta - s * delta) ** 2 +
                                   (k * delta - t * delta) ** 2))) * psi[n]

        return sum
    else:
        return 0


def get_element_exchange_term(n, m, K, psi, delta):
    """
    Calculates an element of the exchange term.
    :param n: The row.
    :param m: The column.
    :param K: The number of divisions.
    :param psi: The wavefunction.
    :param delta: The size of the divisions.
    :return: The exchange term element.
    """
    sum = 0
    i, j, k = from_basis_element_index(n, K)
    r, s, t = from_basis_element_index(m, K)

    return psi[m].conjugate() \
           * (1 / abs(np.sqrt((i * delta - r * delta) ** 2 +
                           (j * delta - s * delta) ** 2 +
                           (k * delta - t * delta) ** 2))) * psi[n]


def get_charge_density_bond_order_matrix_element(n, m, electrons, coeffs):
    """
    Constructs a charge density bond order matrix. See Szabo and Ostlund page 139.
    :param electrons: Number of electrons.
    :param coeffs: The coefficients for each electron (column) and basis function (row).
    :return: The matrix.
    """
    sum = 0
    for a in range(math.floor(electrons / 2)):
        sum += coeffs[n][a] * coeffs[m][a].conjugate()

    sum = 2 * sum

    return sum


def clean_zoos(M):
    """
    Remove complex infinities.
    :param M: The matrix to clean.
    :return: The cleaned matrix.
    """
    for r in range(len(M[0])):
        for c in range(len(M[0])):
            if 'inf' in str(M[r][c]):
                M[r][c] = 10000
            elif 'nan' in str(M[r][c]):
                M[r][c] = 0

    return M

# H2 molecule

nuclear_coordinates = [Coordinate(0, 0, 0), Coordinate(0, 0, 0.74)]
atomic_numbers = [1, 1]
num_electrons = 2

size = 10  # size of space
K = 50  # number of divisions
delta = size / K  # division size

# We use symmetric orthogonalization per Szabo and Ostlund page 143.

S = np.diag([1 for i in range(K ** 3)]).astype(float)  # We know our basis is diagonal.

eigen_values, eigen_vectors = eigs(S, which="SM", k=6)
s_diag = np.diag(eigen_values ** -0.5)
ortho_matrix = np.dot(eigen_vectors, np.dot(s_diag, np.transpose(eigen_vectors)))

# We construct the core Hamiltonian.

kinetic = np.array([[get_element_kinetic_part(row, col)
                   for col in range(K ** 3)] for row in range(K ** 3)])

nuclear_attraction = np.array(
    [[get_element_nuclear_attraction(row, col, nuclear_coordinates, atomic_numbers)
       for col in range(K ** 3)] for row in range(K ** 3)])

H_core = kinetic + nuclear_attraction

H_core = clean_zoos(H_core)

# Now we solve for our two-electron integrals.

# Our initial guess is a poor one.

psi = [randint(0, 10) * 0.1 for o in range(K ** 3)]

# Construct the Coulomb matrix.

C = np.array([[get_element_coulomb_term(row, col, K, psi, delta) * 2
             for col in range(K ** 3)] for row in range(K ** 3)])

# Construct the Exchange matrix.

E = np.array([[get_element_exchange_term(row, col, K, psi, delta)
             for col in range(K ** 3)] for row in range(K ** 3)])

G = C + E

G = clean_zoos(G)

# Construct and transform the Fock matrix.

F = H_core + G

F_trans = np.matmul(np.matmul(np.transpose(np.conjugate(ortho_matrix)),
                              np.array(F).astype(np.float64)), ortho_matrix)

# Get an updated guess from the transformed Fock matrix's eigen vectors.

eigen_values, eigen_vectors = np.linalg.eig(F_trans)
g = np.dot(ortho_matrix, eigen_vectors)

new_psi = g[0]

# Convergence criteria.

inner_tolerance = 0.1
outer_tolerance = 0.5
converged = False

while not converged:

    old_eigen_values = eigen_values
    old_psi = new_psi

    # Construct the Coulomb matrix.

    C = np.array(
        [[get_element_coulomb_term(row, col, K, new_psi, delta) * 2
          for col in range(K ** 3)] for row in range(K ** 3)])

    # Construct the Exchange matrix.

    E = np.array([[get_element_exchange_term(row, col, K, new_psi, delta)
                 for col in range(K ** 3)] for row in range(K ** 3)])

    G = C + E

    G = clean_zoos(G)

    # Construct and transform the Fock matrix.

    F = H_core + G

    F_trans = np.matmul(np.matmul(np.transpose(np.conjugate(ortho_matrix)),
                                  np.array(F).astype(np.float64)), ortho_matrix)

    # Get an updated guess from the transformed Fock matrix's eigen vectors.

    eigen_values, eigen_vectors = np.linalg.eig(F_trans)
    g = np.dot(ortho_matrix, eigen_vectors)

    new_psi = g[0]

    # Test for convergence using the energy of the system (the eigenvalues).

    converged = (np.sum(eigen_values - old_eigen_values < inner_tolerance)
                 / eigen_values.size) > outer_tolerance


eigen_values, eigen_vectors = np.linalg.eig(np.array(F).astype(np.float64))

print("Energies:")
print(eigen_values)

# Plotting.

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = []
ys = []
zs = []
cs = []

for vec in eigen_vectors:
    for n in range(len(vec)):
        i, j, k = from_basis_element_index(n, K)
        xs.append(i * delta)
        ys.append(j * delta)
        zs.append(k * delta)
        cs.append(vec[n])

ax.scatter(xs, ys, zs, c=cs, cmap='gray')

plt.show()
