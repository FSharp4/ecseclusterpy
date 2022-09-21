from copy import deepcopy

import numpy
import os

"""
matrix_logger.py
---------------
Stores iterations of matrices in dictionary for debugging inspection.
Used for debugging ward method implementations
"""

series_iteration = 0
series_name = "series"

store = {}


# def log_matrix(matrix: numpy.ndarray):
#     global series_iteration
#     global series_name
#     if (len(matrix.shape) == 2):
#         if os.path.exists("out") == False:
#             os.makedirs("out")
#
#         numpy.savetxt(f"out/{series_name}_{series_iteration}.csv", matrix, delimiter=", ")
#         series_iteration += 1

def store_matrix(matrix: numpy.ndarray):
    global store
    if series_name not in store:
        store[series_name] = []

    store[series_name] += [deepcopy(matrix)]


def change_series(name: str):
    global series_name
    global series_iteration
    series_name = name
    series_iteration = 0


def store_matrix_by_series(matrix: numpy.ndarray, series: str):
    global store
    if series not in store:
        store[series] = []

    store[series] += [deepcopy(matrix)]
