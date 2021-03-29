from numpy import dot
from numpy.linalg import norm


def cosine_simularity(a, b):
    return dot(a, b) / (norm(a) * norm(b))
