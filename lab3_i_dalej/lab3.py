import copy
import math
import numpy as np
import random

data = []
with open("australian.dat", "r") as f:
    for line in f:
        data.append(list(map(lambda x: float(x), line.split())))

# print(data[0:5])


def euclidean_metric(a, b):
    tmp = 0
    for i in range(len(a)):
        tmp += (b[i] - a[i]) ** 2

    return math.sqrt(tmp)


# print(metryka_euklidesowa(data[0], data[1]))

# lista
# y = lista[0]
# d(y, x) gdzie x = lista[1:]
# zrob {a: b} gdzie a = klasa decyzyjna (ostatni element), b = lista z odleglosciami

# zad 2: wyznacznik macierzy kwadratowej

metrics = {}

base = data[0]
for other in data[1:]:
    distance = euclidean_metric(base, other)
    d_class = int(other[-1])
    if d_class not in metrics:
        metrics[d_class] = [distance]
    metrics[d_class].append(distance)

# print(metrics[0][:10])
# print(metrics[1][:10])


def remove_row_and_column(matrix, row, column):
    result = []
    for i in range(len(matrix)):
        if i != row:
            result.append([])
            for j in range(len(matrix[i])):
                if j != column:
                    result[-1].append(matrix[i][j])
    return result


def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        result = 0
        for i in range(len(matrix)):
            result += (
                matrix[0][i]
                * (-1) ** i
                * determinant(remove_row_and_column(matrix, 0, i))
            )

    return result


# matrix = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
# print(determinant(matrix))


def get_distances_and_classes(vec, rest):
    distances = []
    for other in rest:
        distances.append((int(other[-1]), euclidean_metric(vec, other[:-1])))
    return distances


dists_classes = get_distances_and_classes(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], data
)

metrics = {}
for cls, distance in dists_classes:
    if cls not in metrics:
        metrics[cls] = [distance]
    metrics[cls].append(distance)


def sum_k_lowest_distances(metrics, k):
    sums = {}
    for cls, distances in metrics.items():
        sums[cls] = sum(sorted(distances)[:k])
    return sums


def knn(vector, data, k):
    dists_classes = get_distances_and_classes(vector, data)

    metrics = {}
    for cls, distance in dists_classes:
        if cls not in metrics:
            metrics[cls] = []
        metrics[cls].append(distance)

    sums = sum_k_lowest_distances(metrics, k)

    print(sums)

    minimum = min(sums, key=sums.get)
    if list(sums.values()).count(minimum) > 1:
        return None

    return minimum


def scalar_euclidean_metric(a, b, remove_last=False):
    if remove_last:
        a = a[:-1]
        b = b[:-1]
    a = np.array(a)
    b = np.array(b)
    dist = a - b
    return math.sqrt(np.dot(dist, dist))


# print(knn([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], data, 3))
# print(scalar_euclidean_metric([1, 2, 3], [4, 5, 6]))
# print(metryka_euklidesowa([1, 2, 3], [4, 5, 6]))

# dzielenie na grupy przez przypadkowe kolorowanie
# calkowanie monte carlo
# calkowanie przez sume prostokatow
# calkowanie przez sume trapezow


def split_points_into_groups(points, k):
    groups = {x: [] for x in range(k)}
    for point in points:
        groups[random.randint(0, k - 1)].append(point)
    return groups


def get_centroid_index(points):
    dists = {}
    for i, point in enumerate(points):
        dists[i] = sum(
            [scalar_euclidean_metric(point, other) for other in points]
        ) / len(points)

    return min(dists, key=dists.get)


def get_closest_centroid_group(point, points, centroids):
    dists = {}
    for i, centroid in centroids.items():
        dists[i] = scalar_euclidean_metric(point, points[centroid], True)

    lowest = min(dists, key=dists.get)
    return lowest


def k_means_clustering(data, k):
    groups = split_points_into_groups(data, k)

    for _ in range(5):
        centroids = {}
        for j, group in groups.items():
            centroids[j] = get_centroid_index(group)

        new_groups = {x: [] for x in range(k)}
        for point in data:
            closest_centroid_group = get_closest_centroid_group(point, data, centroids)
            new_groups[closest_centroid_group].append(point)

        groups = new_groups

    return groups


# print(k_means_clustering(data, 2))


def integral_monte_carlo(f, a, b, n):
    result = 0
    for i in range(n):
        result += f(random.uniform(a, b))
    return (result / n) * (b - a)


# print(integral_monte_carlo(lambda x: x, 0, 1, 5000))


def integral_rectangles(f, a, b, n):
    step = (b - a) / n
    result = 0
    for i in range(n):
        result += f(a + i * step) * step

    return result


# print(integral_rectangles(lambda x: x, 0, 1, 5000))


def integral_trapezoids(f, a, b, n):
    step = (b - a) / n
    result = 0
    for i in range(n):
        result += (f(a + i * step) + f(a + (i + 1) * step)) / 2 * step

    return result


# print(integral_trapezoids(lambda x: x**2, 0, 2, 5000))


# def average_of_vectors(vectors):
#    return [sum(x) / len(x) for x in zip(*vectors)]


def average_of_vector(vector):
    vector_np = np.array(vector)
    dot = np.dot(vector_np, np.ones(vector_np.shape))
    return dot / len(vector)


# print(average_of_vectors([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
# print(average_of_vector([1, 2, 3]))

# def variance_of_vectors(vectors):
#    avg = average_of_vectors(vectors)
#
#    return [
#        sum([(x - avg[i]) ** 2 for i, x in enumerate(vector)]) / len(vector)
#        for vector in vectors
#    ]


# def variance_of_vector(vector):
#    avg = average_of_vector(vector)
#    return sum([(x - avg) ** 2 for x in vector]) / len(vector)


def variance_of_vector(vector):
    vector_np = np.array(vector)
    avg = average_of_vector(vector)
    diff = vector_np - avg
    return np.dot(diff, diff) / len(vector)


# print(variance_of_vector([1, 2, 3]))


# def stddev_of_vectors(vectors):
#    return [math.sqrt(x) for x in variance_of_vectors(vectors)]


def stddev_of_vector(vector):
    return math.sqrt(variance_of_vector(vector))


# print(stddev_of_vectors([[1, 2, 3], [4, 5, 6], [1, 2, 3], [1, 2, 3]]))

# print(stddev_of_vector([1, 2, 3]))

# beta = (X^T * X)^-1 * X^T * Y


# def linear_regression(x, y):
#     x_array = np.array(x)
#     y_array = np.array(y)

#     beta =

#     print(x_array)


# print(linear_regression([1, 2, 3], [1, 2, 3]))


# prostopadlowanie:
# u1 = v1
# u2 = v2 - proj_u1(v2)
# u3 = v3 - proj_u1(v3) - proj_u2(v3)
# u_n = v_n - sum(i=1 to n-1)(proj_u_i(v_n))


def project_vector(vector, axis):
    return (np.dot(vector, axis) / np.dot(axis, axis)) * axis


def normalize_vector(vector):
    return vector / np.sqrt(np.dot(vector, vector))


def QR_decomposition(matrix):
    u_1 = matrix[0].copy()
    u_vectors = [u_1]
    num_columns = matrix.shape[0]
    for i in range(1, num_columns):
        v_i = matrix[i].copy()
        for u in u_vectors:
            v_i -= project_vector(v_i, u)
        u_vectors.append(v_i)

    Q = np.array([normalize_vector(u) for u in u_vectors])
    R = np.dot(Q, matrix.T)

    return Q, R


def next_A(A):
    Q, R = QR_decomposition(A)
    Q = Q.T
    return np.dot(np.dot(Q.T, A), Q)


def is_upper_triangular(M):
    for i in range(1, len(M)):
        for j in range(0, i):
            if M[i][j] > 0.001:
                return False
    return True


def eigenvalues(A):
    A_copy = copy.deepcopy(A)
    while not is_upper_triangular(A_copy):
        print(A_copy)
        A_copy = next_A(A_copy)
    return np.diag(A_copy)[::-1]


def gauss_jordan_elimination(A):
    size = np.shape(A)[1]
    result = []

    for i in range(size):
        if A[i][i] == 0:
            raise Exception("Zero!!!")

        for j in range(size):
            if i != j:
                ratio = A[j][i] / A[i][i]

                for k in range(size):
                    A[j][k] -= ratio * A[i][k]

    for i in range(size):
        result.append(A[i][size - 1] / A[i][i])

    return result


def eigenvector(A, eigenvalue):
    A_copy = copy.deepcopy(A)
    size = np.shape(A)[1]

    for i in range(size):
        for j in range(size):
            A_copy[i][j] = A[i][j] - eigenvalue

    return gauss_jordan_elimination(A_copy)


def eigenvectors(A):
    evalues = eigenvalues(A)
    eigenvectors = []
    for eigenvalue in evalues:
        eigenvectors.append(eigenvector(A, eigenvalue))
    return eigenvectors


A = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
# print(A)
Q, R = QR_decomposition(A)
# print(Q)
# print(R)
Q2, R2 = np.linalg.qr(A.T)
# print(Q2.T)
# print(R2)
# print(np.dot(Q.T, R).round(2))

B = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.float32)
print(np.linalg.eig(B)[0].round(2))

print(eigenvectors(B))
