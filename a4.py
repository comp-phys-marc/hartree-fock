import numpy as np
import math
from dataclasses import dataclass
from sympy import symbols, Integer, Integral, \
    Matrix, Function, Float, Mul, Pow, sqrt, Add
from random import randint
import matplotlib.pyplot as plt

# We follow Szabo and Ostlund page 146 for the SCF procedure.

@dataclass
class Coordinate:
    x1: float
    x2: float
    x3: float


class DDelta(Function):
    """
    A class representing the Dirac Delta.
    """

    @classmethod
    def eval(cls, x, *args):
        """
        We do not immediately evaluate DDeltas. Instead, they are symbolically represented until doit() is called.
        """
        pass

    def __hash__(self):
        """
        Identifies a DDelta uniquely.
        :return: The string of its arguments.
        """
        return hash(self.args)

    def __eq__(self, other):
        """
        Allows for us to check whether two DDeltas are the same.
        :param other: The other DDelta.
        :return: Whether they are the same.
        """
        if isinstance(other, DDelta):
            return hash(self.args) == hash(other.args)
        else:
            return False

    def doit(self, deep=False, **hints):
        """
        Evaluates the DDelta at a specific point. The built-in DiracDelta doesn't evaluate at points.
        :param deep: Whether to evaluate nested expressions (I think).
        :param hints: Used by sympy.
        :return: The result of evaluating the DDelta at a given point.
        """
        x = self.args[0]
        if isinstance(x, Integer) or isinstance(x, Float):
            if x == 0:
                return Integer(1)
            else:
                return Integer(0)
        else:
            return DDelta(x)

    def integrate(self, *args, **kwargs):
        """
        Evaluates the integral of the DDelta.
        :param args: The var to integrate with respect to, as well as the limits.
        :param kwargs: Not used.
        :return: The evaluated integral.
        """
        var, low, high = args[0]

        if var not in self.args[0].free_symbols:
            return DDelta(self.args[0])
        # if our unit impulse is in the range
        elif low <= - self.args[0].subs(var, 0) <= high:
            return Integer(1)
        else:
            return Integer(0)


class Mul(Mul):
    """
    Extends the Mul so that we can test for DDelta equality in integrals of product delta functions.
    """

    def integrate(self, *args, **kwargs):
        """
        Evaluates the integral of a product delta function. If the DDeltas are the same and in range, it is 1.
        Otherwise, the integral is 0.
        :param args: The var to integrate with respect to, as well as the limits.
        :param kwargs: Not used.
        :return: The evaluated integral.
        """
        var, low, high = args[0]

        for arg in self.args:
            if not isinstance(arg, DDelta):
                return super().integrate(*args, **kwargs)

        if var not in self.args[0].free_symbols:
            return Mul(*self.args)
        elif low <= - self.args[0].args[0].subs(var, 0) <= high:
            prev = self.args[0]
            for arg in range(1, len(self.args)):
                if prev != self.args[arg]:
                    return Integer(0)
                prev = self.args[arg]
            return Integer(1)
        else:
            return Integer(0)


class Pow(Pow):
    """
    Extends the Pow so that we can test for DDelta equality in integrals of product delta functions.
    """

    def integrate(self, *args, **kwargs):
        """
        Evaluates the integral of a product delta function. If the DDeltas are the same and in range, it is 1.
        Otherwise, the integral is 0.
        :param args: The var to integrate with respect to, as well as the limits.
        :param kwargs: Not used.
        :return: The evaluated integral.
        """
        var, low, high = args[0]

        if isinstance(self.args[0], DDelta):
            if var not in self.args[0].free_symbols:
                return Pow(*self.args)
            # we don't have to check if the DDeltas are the same
            elif low <= - self.args[0].args[0].subs(var, 0) <= high:
                return Integer(1)
            else:
                return Integer(0)
        else:
            return super().integrate(*args, **kwargs)


def is_delta_kind(expr):
    """
    Determines whether the expr is like a DDelta.
    :param expr: The expr that may be like a DDelta.
    :return: Whether the expr is like a DDelta.
    """
    is_inst = isinstance(expr, DDelta)

    is_mul = False
    if 'Mul' in str(expr.__class__):
        for d in range(len(expr.args)):
            if is_delta_kind(expr.args[d]):
                is_mul = True
                break

    is_pow = 'Pow' in str(expr.__class__) and \
             isinstance(expr.args[0], DDelta)

    return is_inst or is_pow or is_mul


class Integral(Integral):
    """
    A class that integrates DDeltas.
    """

    def doit(self, deep=True, **hints):
        """
        Evaluates an integral. Defers to sympy excepts in the case we use a DDelta.
        :param deep: Whether to evaluate nested expressions (I think).
        :param hints: Tells sympy a strategy to use to evaluate the integral.
        :return: The evaluated integral.
        """

        function = self.function  # the function to integrate

        # Handle integrating DDeltas

        if isinstance(function, DDelta):
            return function.integrate(self.limits[0])  # call the user-defined DDelta integrate

        # Handle integrating product delta functions

        elif 'Mul' in str(function.__class__) and \
                isinstance(function.args[0], DDelta) \
                 and isinstance(function.args[1], DDelta) and len(function.args) == 3:
            # call the user-defined DDelta integrate
            return Mul(
                function.args[0],
                function.args[1],
                function.args[2]
            ).integrate(self.limits[0])
        elif 'Mul' in str(function.__class__) and \
                isinstance(function.args[0], DDelta) \
                and isinstance(function.args[1], DDelta) \
                and len(function.args) == 2:
            return Mul(
                function.args[0],
                function.args[1]
            ).integrate(self.limits[0])
        elif 'Pow' in str(function.__class__) and \
                isinstance(function.args[0], DDelta):
            return Pow(
                function.args[0],
                function.args[1]
            ).integrate(self.limits[0])
        elif 'Mul' in str(function.__class__):
            functions_to_return = []
            deltas = []
            var, low, high = self.limits[0]
            for arg in function.args:
                if not 'DDelta' in str(arg) and var in arg.free_symbols or \
                        ('DDelta' in str(arg) and var not in arg.free_symbols):
                    functions_to_return.append(arg)
                elif 'DDelta' in str(arg) and var in arg.free_symbols:
                    deltas.append(arg)
            if len(deltas) == 1 and 'Pow' in str(deltas[0].__class__):
                return Mul(*functions_to_return).subs(
                    var, - deltas[0].args[0].args[0].subs(var, 0))
            elif len(deltas) >= 2 and isinstance(deltas[0], DDelta):
                prev = deltas[0]
                for d in range(1, len(deltas)):
                    if not isinstance(deltas[d], DDelta):
                        # We only implement what we need for our calculation
                        raise Exception('NotImplementedError')
                    if prev != deltas[d]:
                        return Integer(0)
                    prev = deltas[d]
                return Mul(*functions_to_return).subs(
                    var, - deltas[0].args[0].args[0].subs(var, 0))
            elif len(deltas) >= 2 and 'Pow' in str(deltas[0].__class__) \
                    and isinstance(deltas[0].args[0], DDelta):
                prev = deltas[0].args[0]
                for d in range(1, len(deltas)):
                    if not (isinstance(deltas[d], DDelta) or
                            ('Pow' in str(deltas[0].__class__) and
                             isinstance(deltas[0].args[0], DDelta))):
                        raise Exception('NotImplementedError')
                    if isinstance(deltas[d], DDelta) and prev != deltas[d]:
                        return Integer(0)
                    elif 'Pow' in str(deltas[0].__class__) and \
                            isinstance(deltas[0].args[0], DDelta) and\
                            prev != deltas[d].args[0]:
                        return Integer(0)
                    prev = deltas[d]
                return Mul(*functions_to_return).subs(
                    var, - deltas[0].args[0].args[0].args[0].subs(var, 0))
            elif len(deltas) == 2 and isinstance(deltas[0], Add) and\
                    isinstance(deltas[1], DDelta):
                muls = []
                for d in range(len(deltas[0].args)):
                    if not is_delta_kind(deltas[0].args[d]):
                        raise Exception("NotImplementedError")
                    else:
                        muls.append(Integral(Mul(deltas[0].args[d],  deltas[1]),
                                             self.limits[0]).simplify().doit())
                return Add(*muls)
            elif len(deltas) == 2 and isinstance(deltas[1], Add) and\
                    isinstance(deltas[0], DDelta):
                muls = []
                for d in range(len(deltas[1].args)):
                    if not is_delta_kind(deltas[1].args[d]):
                        raise Exception("NotImplementedError")
                    else:
                        muls.append(Integral(Mul(deltas[1].args[d], deltas[0]),
                                             self.limits[1]).simplify().doit())
                return Add(*muls)
            elif len(deltas) == 1 and isinstance(deltas[0], Add):
                return Add(*[Integral(Mul(arg, *functions_to_return),
                                      self.limits[0]
                                      ).simplify().doit() for arg in deltas[0].args])
            elif len(deltas) == 1 and isinstance(deltas[0], DDelta):
                return Mul(*functions_to_return).subs(
                    var, - deltas[0].args[0].subs(var, 0))
            else:
                raise Exception('NotImplementedError')
        elif 'Add' in str(function.__class__):
            integrals = []
            for arg in function.args:
                integrals.append(Integral(arg, self.limits[0]).simplify().doit())
            return Add(*integrals)
        else:
            return super().doit(**hints)  # otherwise do what sympy usually does


def get_product_delta_function(x1, x2, x3, i, j, k, delta):
    """
    Allows for the retrieval of delta functions based on their index. The index is defined by i K^2 + j K + k,
    where the delta function indicates the wavefunction at (i * delta, j * delta, k * delta).
    :param x1: The symbol for the x1 position.
    :param x2: The symbol for the position in x2.
    :param x3: The symbol for the position in x3.
    :param i: The index of the delta function in x1.
    :param j: The index of the delta function in x2.
    :param k: The index of the delta function in x3.
    :param delta: The size of the grid spacing.
    :return: The delta function corresponding to the provided index.
    """
    return DDelta(x1 - i * delta) * DDelta(x2 - j * delta) * DDelta(x3 - k * delta)


def create_finite_basis(x1, x2, x3, N, delta):
    """
    Creates a list of product delta functions that define a basis.
    over our grid of x1, x2, x3 locations.
    :param N: The number of divisions of x1, x2 and x3 in our grid.
    :param delta: The size of the divisions.
    :return: The finite basis set.
    """
    finite_basis = []

    # build our finite basis
    for i in range(N):
        for j in range(N):
            for k in range(N):
                finite_basis.append(
                    get_product_delta_function(
                        x1,
                        x2,
                        x3,
                        i,
                        j,
                        k,
                        delta
                    ))
                # now we can index the list by i K^2 + j K + k

    return finite_basis


def eval_integrals(expr, debug=False):
    """
    Evaluates integrals until they are simplified completely.
    :param expr: The expression involving integrals.
    :param debug: Whether to be verbose.
    :return: The simplified expression.
    """
    while 'Integral' in str(expr):
        prev = expr
        if debug:
            print(prev)
        expr = expr.doit()

    return expr


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


def get_element_kinetic_part(n, m, x1, x2, x3, finite_basis, size):
    """
    Returns the matrix element at col m and row n of H_kinetic.
    :param n: The row.
    :param m: The column.
    :param x1: The symbol for the position in x1.
    :param x2: The symbol for the position in x2.
    :param x3: The symbol for the position in x3.
    :param basis: The finite basis.
    :param size: The size of the space to integrate over.
    :return: The corresponding matrix element.
    """

    i, j, k = from_basis_element_index(m, K)

    return eval_integrals(Integral(
        eval_integrals(Integral(
            eval_integrals(Integral( - (Integer(1) / Integer(2)) * finite_basis[n] *
                     (DDelta(x2 - j * delta) * DDelta(x3 - k * delta) *
                        (DDelta(x1 - (i - 1) * delta) - 2 *
                         DDelta(x1 - i * delta) + DDelta(x1 - (i + 1) * delta))
                      + DDelta(x1 - i * delta) * DDelta(x3 - k * delta) *
                        (DDelta(x2 - (j - 1) * delta) - 2 *
                         DDelta(x2 - j * delta) + DDelta(x2 - (j + 1) * delta))
                      + DDelta(x1 - i * delta) * DDelta(x2 - j * delta) *
                        (DDelta(x3 - (k - 1) * delta) - 2 *
                         DDelta(x3 - k * delta) + DDelta(x3 - (k + 1) * delta))),
                (x1, 0, size))),
            (x2, 0, size))),
        (x3, 0, size)))


def get_element_nuclear_attraction(n, m, x1, x2, x3, finite_basis, size, nuclear_coordinates, atom_nums):
    """
    Returns the matrix element at col m and row n of H_nuclear_attraction.
    :param n: The row.
    :param m: The column.
    :param x1: The symbol for the position in x1.
    :param x2: The symbol for the position in x2.
    :param x3: The symbol for the position in x3.
    :param basis: The finite basis.
    :param size: The size of the space to integrate over.
    :param nuclear_coordinates: The coordinates of our nuclei.
    :param atom_nums: The atomic numbers.
    :return: The corresponding matrix element.
    """
    assert len(atom_nums) == len(nuclear_coordinates)

    attractions = 0
    for c, coord in enumerate(nuclear_coordinates):
        attractions -= atom_nums[c] / abs(sqrt((x1 - coord.x1) ** 2 +
                                               (x2 - coord.x2) ** 2 +
                                               (x3 - coord.x3) ** 2))

    return eval_integrals(Integral(
        eval_integrals(Integral(
            eval_integrals(Integral(
                finite_basis[n] * attractions * finite_basis[m],
            (x1, 0, size))),
        (x2, 0, size))),
    (x3, 0, size)))


def get_integral_two_electron(mu, nu, lambd, sigma, basis_one, basis_two, x1, x2, x3, y1, y2, y3, size):
    """
    Returns the two-electron double-integral corresponding to the four basis elements provided.
    :param mu: The first basis element index.
    :param nu: The second basis element index.
    :param lambd: The third basis element index.
    :param sigma: The final basis element index.
    :param basis_one: The basis functions over the position of particle 1.
    :param basis_two: The basis functions over the position of particle 2.
    :param x1: The x1 position of the first electron.
    :param x2: The x2 position of the first electron.
    :param x3: The x3 position of the first electron.
    :param y1: The y1 position of the second electron.
    :param y2: The y2 position of the second electron.
    :param y3: The y3 position of the second electron.
    :param size: The space to integrate over.
    :return: The double integral.
    """
    return eval_integrals(Integral(
        eval_integrals(Integral(
            eval_integrals(Integral(
                eval_integrals(Integral(
                    eval_integrals(Integral(
                        eval_integrals(Integral(
                                basis_one[mu] * basis_one[nu] *
                                (Integer(1) / abs(sqrt((x1 - y1) ** 2 +
                                                       (x2 - y2) ** 2 +
                                                       (x3 - y3) ** 2))) *
                                basis_two[lambd] * basis_two[sigma],
                            (x1, 0, size))),
                        (x2, 0, size))),
                    (x3, 0, size))),
                (y1, 0, size))),
            (y2, 0, size))),
        (y3, 0, size)))


def get_element_two_electron(n, m, density_matrix, size, basis_one, basis_two, x1, x2, x3, y1, y2, y3):
    """
    Calculates an element of the two-electron interaction matrix.
    :param n: The row.
    :param m: The column
    :param density_matrix: The density matrix that contains the decomposition of psi.
    :param size: The size of the space to integrate over.
    :param basis_one: The basis functions over the position of particle 1.
    :param basis_two: The basis functions over the position of particle 2.
    :param x1: The x1 position of the first electron.
    :param x2: The x2 position of the first electron.
    :param x3: The x3 position of the first electron.
    :param y1: The y1 position of the second electron.
    :param y2: The y2 position of the second electron.
    :param y3: The y3 position of the second electron.
    :param size: The space to integrate over.
    :return: The matrix element.
    """
    # TODO: account for symmetries (optimization)
    sum = 0

    for lambd in range(len(basis_two)):
        for sigma in range(len(basis_two)):
            sum += density_matrix[lambd][sigma] * \
                (get_integral_two_electron(
                    n,
                    m,
                    sigma,
                    lambd,
                    basis_one,
                    basis_two,
                    x1,
                    x2,
                    x3,
                    y1,
                    y2,
                    y3,
                    size
                ) - (1 / 2)
                 * get_integral_two_electron(
                            n,
                            lambd,
                            sigma,
                            m,
                            basis_one,
                            basis_two,
                            x1,
                            x2,
                            x3,
                            y1,
                            y2,
                            y3,
                            size
                ))
    return sum


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

# H2 molecule

nuclear_coordinates = [Coordinate(0, 0, 0), Coordinate(0, 0, 0.74)]
atomic_numbers = [1, 1]
num_electrons = 2

x1, x2, x3 = symbols('x1 x2 x3')

size = 4  # size of space
K = 2  # number of divisions
delta = size / K  # division size

finite_basis = create_finite_basis(x1, x2, x3, K, delta)

# We use symmetric orthogonalization per Szabo and Ostlund page 143.

S = []

for mu, phi_mu in enumerate(finite_basis):
    S.append([])
    for phi_nu in finite_basis:
        S[mu].append(eval_integrals(Integral(
                eval_integrals(Integral(
                    eval_integrals(
                        Integral(phi_mu * phi_nu, (x1, 0, K))),
                    (x2, 0, K))),
            (x3, 0, K))))

S = np.array(S).astype('float64')

eigen_values, eigen_vectors = np.linalg.eigh(S)
s_diag = np.diag(eigen_values ** -0.5)
ortho_matrix = np.dot(eigen_vectors, np.dot(s_diag, np.transpose(eigen_vectors)))

# We construct the core Hamiltonian.

kinetic = Matrix(
    [[get_element_kinetic_part(row, col, x1, x2, x3, finite_basis, size)
      for col in range(K ** 3)] for row in range(K ** 3)])

nuclear_attraction = Matrix(
    [[get_element_nuclear_attraction(
        row, col, x1, x2, x3, finite_basis, size, nuclear_coordinates, atomic_numbers)
       for col in range(K ** 3)] for row in range(K ** 3)])

H_core = kinetic + nuclear_attraction

# Now we solve for our two-electron integrals. With K = 100 there are 12,753,775 of them! (Szabo, Ostlund page 141).

y1, y2, y3 = symbols('x1 x2 x3')
second_finite_basis = create_finite_basis(y1, y2, y3, K, delta)

# Our initial guess is a poor one.

coeffs = np.array([[randint(0, 10) * 0.1 for electron in range(2)] for row in range(len(finite_basis))])

P = np.array([[get_charge_density_bond_order_matrix_element(row, col, num_electrons, coeffs)
         for col in range(len(finite_basis))] for row in range(len(finite_basis))])

# Construct the two-electron matrix.

G = Matrix(
    [[get_element_two_electron(row, col, P, size, finite_basis, second_finite_basis, x1, x2, x3, y1, y2, y3)
        for col in range(K ** 3)] for row in range(K ** 3)])

# Construct and transform the Fock matrix.

F = H_core + G

for r in range(len(F.row(0))):
    for c in range(len(F.row(0))):
        if 'zoo' in str(F[r, c]):
            F[r, c] = 10000

F_trans = np.matmul(np.matmul(np.transpose(np.conjugate(ortho_matrix)),
                              np.array(F).astype(np.float64)), ortho_matrix)

# Get an updated guess from the transformed Fock matrix's eigen vectors.

eigen_values, eigen_vectors = np.linalg.eig(F_trans)
C = np.dot(ortho_matrix, eigen_vectors)


# Convergence criteria.

inner_tolerance = 0.1
outer_tolerance = 0.5
converged = False

new_P = P

while not converged:

    old_P = new_P

    # Construct the charge density bond order matrix.

    new_P = np.array([[get_charge_density_bond_order_matrix_element(row, col, num_electrons, C)
                     for col in range(len(finite_basis))] for row in range(len(finite_basis))])

    # Construct the two-electron matrix.

    G = Matrix(
        [[get_element_two_electron(
            row, col, new_P, size, finite_basis, second_finite_basis, x1, x2, x3, y1, y2, y3)
          for col in range(K ** 3)] for row in range(K ** 3)])

    # Construct and transform the Fock matrix.

    F = H_core + G

    for r in range(len(F.row(0))):
        for c in range(len(F.row(0))):
            if 'zoo' in str(F[r, c]):
                F[r, c] = 10000

    F_trans = np.matmul(np.matmul(np.transpose(np.conjugate(ortho_matrix)),
                                  np.array(F).astype(np.float64)), ortho_matrix)

    # Get an updated guess from the transformed Fock matrix's eigen vectors.

    eigen_values, eigen_vectors = np.linalg.eig(F_trans)
    C = np.dot(ortho_matrix, eigen_vectors)

    converged = (np.sum(old_P - new_P < inner_tolerance) / new_P.size) > outer_tolerance

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
