# [score on score: 71.04878048780488] [mutations: 13]
# Scores 71.04878048780488

# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
# EVOLVE-BLOCK-START
"""AlphaEvolve experiment codebase for maximizing the d-invariant of permutations."""
import itertools
import logging
import time
from scipy import integrate
import numpy as np
from scipy import optimize
import warnings
import random
import re
from typing import Any, Callable, Mapping, List, Tuple, Dict
import scipy.linalg as la
import collections
import copy
import math
import numba
from functools import lru_cache

njit = numba.njit
# EVOLVE-BLOCK-END

# PREVIOUS CONSTRUCTIONS START HERE


best_perms_10_iqhd = ((1, 5, 3, 7, 2, 9, 4, 8, 6, 10), (10,4, 6, 2, 8, 3, 9, 5, 7, 1))
best_score_10_iqhd = 16.0
best_perms_11_iqhd = ((1, 5, 3, 8, 2, 10,6, 9, 4, 11, 7), (11, 4, 6, 2, 9, 7, 10,3, 8, 5, 1))
best_score_11_iqhd = 18.0
best_perms_12_iqhd = ((1, 5, 3, 9, 2, 11, 7, 10,4, 8, 6, 12), (12, 4, 8, 2, 10,6, 11, 3, 9, 5, 7, 1))
best_score_12_iqhd = 21.0
best_perms_13_iqhd = ((1, 7, 4, 10,2, 8, 5, 12, 3, 9, 6, 11, 13), (13, 6, 9, 3, 11, 4, 7, 1, 12, 5, 8, 2, 10))
best_score_13_iqhd = 22.0
best_perms_14_iqhd = ((1, 8, 4, 11, 2, 9, 6, 13, 3, 10,5, 12, 7, 14), (14, 7, 10,3, 12, 5, 8, 1, 13, 6, 9, 2, 11, 4))
best_score_14_iqhd = 25.0
best_perms_15_iqhd = ((1, 8, 4, 12, 2, 10,6, 14, 3, 9, 5, 13, 7, 11, 15), (15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10,2, 12, 4, 8))
best_score_15_iqhd = 28.0
best_perms_16_iqhd = ((1, 9, 5, 13, 3, 11, 7, 15, 2, 10,6, 14, 4, 12, 8, 16), (16, 8, 12, 4, 14, 6, 10,2, 15, 7, 11, 3, 13, 5, 9, 1))
best_score_16_iqhd = 32.0
best_perms_17_iqhd = ((1, 9, 5, 13, 3, 11, 7, 15, 2, 10,6, 14, 4, 12, 8, 16, 17), (17, 8, 12, 4, 14, 6, 10,2, 15, 7, 11, 3, 13, 5, 9, 1, 16))
best_score_17_iqhd = 33.0
best_perms_18_iqhd = ((1, 10,5, 14, 3, 12, 7, 16, 2, 11, 6, 15, 4, 13, 8, 17, 9, 18), (18, 9, 13, 4, 15, 6, 11, 2, 16, 7, 12, 3, 14, 5, 10,1, 17, 8))
best_score_18_iqhd = 35.0
best_perms_19_iqhd = ((1, 10,5, 15, 3, 12, 7, 17, 2, 11, 6, 16, 4, 13, 8, 18, 9, 14, 19), (19, 9, 14, 4, 16, 6, 11, 2, 17, 7, 12, 3, 15, 5, 10,1, 18, 8, 13))
best_score_19_iqhd = 37.0
best_perms_20_iqhd = ((1, 11, 6, 16, 3, 13, 8, 18, 2, 12, 7, 17, 4, 14, 9, 19, 5, 15, 10,20), (20,10,15, 5, 17, 7, 12, 2, 18, 8, 13, 3, 16, 6, 11, 1, 19, 9, 14, 4))
best_score_20_iqhd = 40.0
best_perms_21_iqhd = ((1, 11, 6, 16, 3, 13, 8, 19, 2, 12, 7, 17, 4, 14, 9, 20,5, 15, 10,18, 21), (21, 10,15, 5, 18, 7, 12, 2, 19, 8, 13, 3, 16, 6, 11, 1, 20,9, 14, 4, 17))
best_score_21_iqhd = 42.0
best_perms_22_iqhd = ((1, 12, 6, 17, 3, 14, 9, 20,2, 13, 7, 18, 4, 15, 10,21, 5, 16, 8, 19, 11, 22), (22, 11, 16, 5, 19, 8, 13, 2, 20,9, 14, 3, 17, 6, 12, 1, 21, 10,15, 4, 18, 7))
best_score_22_iqhd = 45.0
best_perms_23_iqhd = ((1, 12, 6, 18, 3, 15, 9, 21, 2, 13, 7, 19, 4, 16, 10,22, 5, 14, 8, 20,11, 17, 23), (23, 11, 17, 5, 20,8, 14, 2, 21, 9, 15, 3, 18, 6, 12, 1, 22, 10,16, 4, 19, 7, 13))
best_score_23_iqhd = 48.0
best_perms_24_iqhd = ((1, 13, 7, 19, 4, 16, 10,22, 2, 14, 8, 20,5, 17, 11, 23, 3, 15, 9, 21, 6, 18, 12, 24), (24, 12, 18, 6, 21, 9, 15, 3, 22, 10,16, 4, 19, 7, 13, 1, 23, 11, 17, 5, 20,8, 14, 2))
best_score_24_iqhd = 52.0
best_perms_25_iqhd = ((1, 13, 7, 19, 4, 16, 10,22, 2, 14, 8, 20,5, 17, 11, 24, 3, 15, 9, 21, 6, 18, 12, 23, 25), (25, 12, 18, 6, 21, 9, 15, 3, 23, 10,16, 4, 19, 7, 13, 1, 24, 11, 17, 5, 20,8, 14, 2, 22))
best_score_25_iqhd = 54.0
best_perms_26_iqhd = ((1, 14, 7, 20,4, 17, 10,23, 2, 15, 8, 21, 5, 18, 12, 25, 3, 16, 9, 22, 6, 19, 11, 24, 13, 26), (26, 13, 19, 6, 22, 9, 16, 3, 24, 11, 17, 4, 20,7, 14, 1, 25, 12, 18, 5, 21, 8, 15, 2, 23, 10))
best_score_26_iqhd = 57.0
best_perms_27_iqhd = ((1, 14, 7, 21, 4, 17, 10,24, 2, 15, 8, 22, 5, 19, 12, 26, 3, 16, 9, 23, 6, 18, 11, 25, 13, 20,27), (27, 13, 20,6, 23, 9, 16, 3, 25, 11, 18, 4, 21, 7, 14, 1, 26, 12, 19, 5, 22, 8, 15, 2, 24, 10,17))
best_score_27_iqhd = 60.0
best_perms_28_iqhd = ((1, 15, 8, 22, 4, 18, 11, 25, 2, 16, 9, 23, 6, 20,13, 27, 3, 17, 10,24, 5, 19, 12, 26, 7, 21, 14, 28), (28, 14, 21, 7, 24, 10,17, 3, 26, 12, 19, 5, 22, 8, 15, 1, 27, 13, 20,6, 23, 9, 16, 2, 25, 11, 18, 4))
best_score_28_iqhd = 64.0
best_perms_29_iqhd = ((1, 15, 8, 22, 4, 18, 11, 26, 2, 16, 9, 24, 6, 20,13, 28, 3, 17, 10,23, 5, 19, 12, 27, 7, 21, 14, 25, 29), (29, 14, 21, 7, 25, 10,17, 3, 27, 12, 19, 5, 23, 8, 15, 1, 28, 13, 20,6, 24, 9, 16, 2, 26, 11, 18, 4, 22))
best_score_29_iqhd = 67.0
best_perms_30_iqhd = ((1, 16, 8, 23, 4, 19, 12, 27, 2, 17, 10,25, 6, 21, 14, 29, 3, 18, 9, 24, 5, 20,13, 28, 7, 22, 11, 26, 15, 30), (30,15, 22, 7, 26, 11, 18, 3, 28, 13, 20,5, 24, 9, 16, 1, 29, 14, 21, 6, 25, 10,17, 2, 27, 12, 19, 4, 23, 8))
best_score_30_iqhd = 71.0
best_perms_31_iqhd = ((1, 16, 8, 24, 4, 20,12, 28, 2, 18, 10,26, 6, 22, 14, 30,3, 17, 9, 25, 5, 21, 13, 29, 7, 19, 11, 27, 15, 23, 31), (31, 15, 23, 7, 27, 11, 19, 3, 29, 13, 21, 5, 25, 9, 17, 1, 30,14, 22, 6, 26, 10,18, 2, 28, 12, 20,4, 24, 8, 16))
best_score_31_iqhd = 75.0
best_perms_32_iqhd = ((1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 2, 18, 10,26, 6, 22, 14, 30,4, 20,12, 28, 8, 24, 16, 32), (32, 16, 24, 8, 28, 12, 20,4, 30,14, 22, 6, 26, 10,18, 2, 31, 15, 23, 7, 27, 11, 19, 3, 29, 13, 21, 5, 25, 9, 17, 1))
best_score_32_iqhd = 80.0
best_perms_33_iqhd = ((1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 2, 18, 10,26, 6, 22, 14, 30,4, 20,12, 28, 8, 24, 16, 32, 33), (33, 16, 24, 8, 28, 12, 20,4, 30,14, 22, 6, 26, 10,18, 2, 31, 15, 23, 7, 27, 11, 19, 3, 29, 13, 21, 5, 25, 9, 17, 1, 32))
best_score_33_iqhd = 81.0
best_perms_34_iqhd = ((1, 18, 9, 26, 5, 22, 13, 30,3, 20,11, 28, 7, 24, 15, 32, 2, 19, 10,27, 6, 23, 14, 31, 4, 21, 12, 29, 8, 25, 16, 33, 17, 34), (34, 17, 25, 8, 29, 12, 21, 4, 31, 14, 23, 6, 27, 10,19, 2, 32, 15, 24, 7, 28, 11, 20,3, 30,13, 22, 5, 26, 9, 18, 1, 33, 16))
best_score_34_iqhd = 83.0
best_perms_35_iqhd = ((1, 18, 9, 27, 5, 22, 13, 31, 3, 20,11, 29, 7, 24, 15, 33, 2, 19, 10,28, 6, 23, 14, 32, 4, 21, 12, 30,8, 25, 16, 34, 17, 26, 35), (35, 17, 26, 8, 30,12, 21, 4, 32, 14, 23, 6, 28, 10,19, 2, 33, 15, 24, 7, 29, 11, 20,3, 31, 13, 22, 5, 27, 9, 18, 1, 34, 16, 25))
best_score_35_iqhd = 85.0
best_perms_36_iqhd = ((1, 19, 10,28, 5, 23, 14, 32, 3, 21, 12, 30,7, 25, 16, 34, 2, 20,11, 29, 6, 24, 15, 33, 4, 22, 13, 31, 8, 26, 17, 35, 9, 27, 18, 36), (36, 18, 27, 9, 31, 13, 22, 4, 33, 15, 24, 6, 29, 11, 20,2, 34, 16, 25, 7, 30,12, 21, 3, 32, 14, 23, 5, 28, 10,19, 1, 35, 17, 26, 8))
best_score_36_iqhd = 88.0
best_perms_37_iqhd = ((1, 19, 10,28, 5, 23, 14, 33, 3, 21, 12, 30,7, 25, 16, 35, 2, 20,11, 29, 6, 24, 15, 34, 4, 22, 13, 31, 8, 26, 17, 36, 9, 27, 18, 32, 37), (37, 18, 27, 9, 32, 13, 22, 4, 34, 15, 24, 6, 29, 11, 20,2, 35, 16, 25, 7, 30,12, 21, 3, 33, 14, 23, 5, 28, 10,19, 1, 36, 17, 26, 8, 31))
best_score_37_iqhd = 90.0
best_perms_38_iqhd = ((1, 20,10,29, 5, 24, 15, 34, 3, 22, 12, 31, 7, 26, 17, 36, 2, 21, 11, 30,6, 25, 16, 35, 4, 23, 13, 32, 8, 27, 18, 37, 9, 28, 14, 33, 19, 38), (38, 19, 28, 9, 33, 14, 23, 4, 35, 16, 25, 6, 30,11, 21, 2, 36, 17, 26, 7, 31, 12, 22, 3, 34, 15, 24, 5, 29, 10,20,1, 37, 18, 27, 8, 32, 13))
best_score_38_iqhd = 93.0
best_perms_39_iqhd = ((1, 20,10,30,5, 25, 15, 35, 3, 22, 12, 32, 7, 27, 17, 37, 2, 21, 11, 31, 6, 26, 16, 36, 4, 23, 13, 33, 8, 28, 18, 38, 9, 24, 14, 34, 19, 29, 39), (39, 19, 29, 9, 34, 14, 24, 4, 36, 16, 26, 6, 31, 11, 21, 2, 37, 17, 27, 7, 32, 12, 22, 3, 35, 15, 25, 5, 30,10,20,1, 38, 18, 28, 8, 33, 13, 23))
best_score_39_iqhd = 96.0
best_perms_40_iqhd = ((1, 21, 11, 31, 6, 26, 16, 36, 3, 23, 13, 33, 8, 28, 18, 38, 2, 22, 12, 32, 7, 27, 17, 37, 4, 24, 14, 34, 9, 29, 19, 39, 5, 25, 15, 35, 10,30,20,40), (40,20,30,10,35, 15, 25, 5, 37, 17, 27, 7, 32, 12, 22, 2, 38, 18, 28, 8, 33, 13, 23, 3, 36, 16, 26, 6, 31, 11, 21, 1, 39, 19, 29, 9, 34, 14, 24, 4))
best_score_40_iqhd = 100.0
best_perms_41_iqhd = ((1, 21, 11, 31, 6, 26, 16, 36, 3, 23, 13, 33, 8, 28, 18, 39, 2, 22, 12, 32, 7, 27, 17, 37, 4, 24, 14, 34, 9, 29, 19, 40,5, 25, 15, 35, 10,30,20,38, 41), (41, 20,30,10,35, 15, 25, 5, 38, 17, 27, 7, 32, 12, 22, 2, 39, 18, 28, 8, 33, 13, 23, 3, 36, 16, 26, 6, 31, 11, 21, 1, 40,19, 29, 9, 34, 14, 24, 4, 37))
best_score_41_iqhd = 102.0
best_perms_42_iqhd = ((1, 22, 11, 32, 6, 27, 16, 37, 3, 24, 13, 34, 8, 29, 19, 40,2, 23, 12, 33, 7, 28, 17, 38, 4, 25, 14, 35, 9, 30,20,41, 5, 26, 15, 36, 10,31, 18, 39, 21, 42), (42, 21, 31, 10,36, 15, 26, 5, 39, 18, 28, 7, 33, 12, 23, 2, 40,19, 29, 8, 34, 13, 24, 3, 37, 16, 27, 6, 32, 11, 22, 1, 41, 20,30,9, 35, 14, 25, 4, 38, 17))
best_score_42_iqhd = 105.0
best_perms_43_iqhd = ((1, 22, 11, 33, 6, 27, 16, 38, 3, 24, 13, 35, 8, 30,19, 41, 2, 23, 12, 34, 7, 28, 17, 39, 4, 25, 14, 36, 9, 31, 20,42, 5, 26, 15, 37, 10,29, 18, 40,21, 32, 43), (43, 21, 32, 10,37, 15, 26, 5, 40,18, 29, 7, 34, 12, 23, 2, 41, 19, 30,8, 35, 13, 24, 3, 38, 16, 27, 6, 33, 11, 22, 1, 42, 20,31, 9, 36, 14, 25, 4, 39, 17, 28))
best_score_43_iqhd = 108.0
best_perms_44_iqhd = ((1, 23, 12, 34, 6, 28, 17, 39, 3, 25, 14, 36, 9, 31, 20,42, 2, 24, 13, 35, 7, 29, 18, 40,4, 26, 15, 37, 10,32, 21, 43, 5, 27, 16, 38, 8, 30,19, 41, 11, 33, 22, 44), (44, 22, 33, 11, 38, 16, 27, 5, 41, 19, 30,8, 35, 13, 24, 2, 42, 20,31, 9, 36, 14, 25, 3, 39, 17, 28, 6, 34, 12, 23, 1, 43, 21, 32, 10,37, 15, 26, 4, 40,18, 29, 7))
best_score_44_iqhd = 112.0
best_perms_45_iqhd = ((1, 23, 12, 34, 6, 28, 17, 40,3, 25, 14, 37, 9, 31, 20,43, 2, 24, 13, 35, 7, 29, 18, 41, 4, 26, 15, 38, 10,32, 21, 44, 5, 27, 16, 36, 8, 30,19, 42, 11, 33, 22, 39, 45), (45, 22, 33, 11, 39, 16, 27, 5, 42, 19, 30,8, 36, 13, 24, 2, 43, 20,31, 9, 37, 14, 25, 3, 40,17, 28, 6, 34, 12, 23, 1, 44, 21, 32, 10,38, 15, 26, 4, 41, 18, 29, 7, 35))
best_score_45_iqhd = 115.0
best_perms_46_iqhd = ((1, 24, 12, 35, 6, 29, 18, 41, 3, 26, 15, 38, 9, 32, 21, 44, 2, 25, 13, 36, 7, 30,19, 42, 4, 27, 16, 39, 10,33, 22, 45, 5, 28, 14, 37, 8, 31, 20,43, 11, 34, 17, 40,23, 46), (46, 23, 34, 11, 40,17, 28, 5, 43, 20,31, 8, 37, 14, 25, 2, 44, 21, 32, 9, 38, 15, 26, 3, 41, 18, 29, 6, 35, 12, 24, 1, 45, 22, 33, 10,39, 16, 27, 4, 42, 19, 30,7, 36, 13))
best_score_46_iqhd = 119.0
best_perms_47_iqhd = ((1, 24, 12, 36, 6, 30,18, 42, 3, 27, 15, 39, 9, 33, 21, 45, 2, 25, 13, 37, 7, 31, 19, 43, 4, 28, 16, 40,10,34, 22, 46, 5, 26, 14, 38, 8, 32, 20,44, 11, 29, 17, 41, 23, 35, 47), (47, 23, 35, 11, 41, 17, 29, 5, 44, 20,32, 8, 38, 14, 26, 2, 45, 21, 33, 9, 39, 15, 27, 3, 42, 18, 30,6, 36, 12, 24, 1, 46, 22, 34, 10,40,16, 28, 4, 43, 19, 31, 7, 37, 13, 25))
best_score_47_iqhd = 123.0
best_perms_48_iqhd = ((1, 25, 13, 37, 7, 31, 19, 43, 4, 28, 16, 40,10,34, 22, 46, 2, 26, 14, 38, 8, 32, 20,44, 5, 29, 17, 41, 11, 35, 23, 47, 3, 27, 15, 39, 9, 33, 21, 45, 6, 30,18, 42, 12, 36, 24, 48), (48, 24, 36, 12, 42, 18, 30,6, 45, 21, 33, 9, 39, 15, 27, 3, 46, 22, 34, 10,40,16, 28, 4, 43, 19, 31, 7, 37, 13, 25, 1, 47, 23, 35, 11, 41, 17, 29, 5, 44, 20,32, 8, 38, 14, 26, 2))
best_score_48_iqhd = 128.0
best_perms_49_iqhd = ((1, 25, 13, 37, 7, 31, 19, 43, 4, 28, 16, 40,10,34, 22, 46, 2, 26, 14, 38, 8, 32, 20,44, 5, 29, 17, 41, 11, 35, 23, 48, 3, 27, 15, 39, 9, 33, 21, 45, 6, 30,18, 42, 12, 36, 24, 47, 49), (49, 24, 36, 12, 42, 18, 30,6, 45, 21, 33, 9, 39, 15, 27, 3, 47, 22, 34, 10,40,16, 28, 4, 43, 19, 31, 7, 37, 13, 25, 1, 48, 23, 35, 11, 41, 17, 29, 5, 44, 20,32, 8, 38, 14, 26, 2, 46))
best_score_49_iqhd = 130.0
best_perms_50_iqhd = ((1, 26, 13, 38, 7, 32, 19, 44, 4, 29, 16, 41, 10,35, 22, 47, 2, 27, 14, 39, 8, 33, 20,45, 5, 30,17, 42, 11, 36, 24, 49, 3, 28, 15, 40,9, 34, 21, 46, 6, 31, 18, 43, 12, 37, 23, 48, 25, 50), (50,25, 37, 12, 43, 18, 31, 6, 46, 21, 34, 9, 40,15, 28, 3, 48, 23, 35, 10,41, 16, 29, 4, 44, 19, 32, 7, 38, 13, 26, 1, 49, 24, 36, 11, 42, 17, 30,5, 45, 20,33, 8, 39, 14, 27, 2, 47, 22))
best_score_50_iqhd = 133.0
best_average_score_found_iqhd = np.float64(71.04878048780488)


# PREVIOUS CONSTRUCTIONS END HERE


# EVOLVE-BLOCK-START
def search_for_best_permutations(
    n: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
  """Searches for a pair of permutations (x, y) in S_n maximizing d(x,y)."""

  if n == 0:
    return (tuple(), tuple())
  if n == 1:
    return ((1,), (1,))

  e = tuple(range(1, n + 1))

  # Construction for x=e, y=w_0 (anti-identity), which yields d(e,w_0) = n-1.
  # This is the maximum possible value for d(e,y) for any y.
  w0 = tuple(range(n, 0, -1)) # Anti-identity permutation
  initial_best_perms = (e, w0)
  initial_best_score = get_score(initial_best_perms, n)

  # Preserve the definition of 'optimal_y' from the previous code, as it's used in 'hybrid_perms'.
  # This 'optimal_y' is a specific "hook" permutation, not necessarily w_0.
  optimal_y_hook_construction = None
  if n % 2 != 0: # n is odd, n = 2m + 1
      m_hook = (n - 1) // 2
      # y = (n, 1, 2, ..., m, n-1, n-2, ..., m+1)
      y_hook_list = [n] + list(range(1, m_hook + 1)) + list(range(n - 1, m_hook, -1))
      optimal_y_hook_construction = tuple(y_hook_list)
  else: # n is even, n = 2m
      m_hook = n // 2
      # y = (n, n-1, ..., m+1, 1, 2, ..., m) -- A standard hook permutation for even n
      y_hook_list = list(range(n, m_hook, -1)) + list(range(1, m_hook + 1))
      optimal_y_hook_construction = tuple(y_hook_list)


  start_time = time.time()
  best_perms = initial_best_perms
  best_score = initial_best_score
  # Set a generous time limit for the whole process.
  time_limit = 9.8

  # --- Haiman-type construction for maximal d-invariant ---
  # This specific construction is known to achieve d(x,y) = 2n-4 (even n)
  # or d(x,y) = 2n-5 (odd n) and satisfy x <= y in Bruhat order.
  # This complements the x=e construction by using a non-identity 'x'.

  # m is n//2 for both even and odd n in this context of x_haiman
  m_haiman = n // 2 

  # x_haiman = (1, 2, ..., m, n, n-1, ..., m+1)
  # For n=1, m_haiman=0, this gives empty list. Handled by n=1 check.
  # For n>=2, m_haiman>=1.
  haiman_x_list = list(range(1, m_haiman + 1)) + list(range(n, m_haiman, -1))
  haiman_x = tuple(haiman_x_list)

  if n % 2 == 0:  # n is even, n = 2m
      # y_haiman = (1, 3, ..., 2m-1, 2m, 2m-2, ..., 2)
      haiman_y_list = [i for i in range(1, n + 1) if i % 2 != 0] + \
                      [i for i in range(n, 0, -1) if i % 2 == 0]
      haiman_y = tuple(haiman_y_list)
  else:  # n is odd, n = 2m + 1
      # y_haiman = (1, 3, ..., 2m-1, 2m+1, 2m, 2m-2, ..., 2)
      haiman_y_list = [i for i in range(1, n, 2)] + [n] + \
                      [i for i in range(n - 1, 0, -2)]
      haiman_y = tuple(haiman_y_list)

  haiman_perms = (haiman_x, haiman_y)
  score_haiman = get_score(haiman_perms, n)
  if score_haiman > best_score:
      best_score = score_haiman
      best_perms = haiman_perms

  # --- Hybrid Construction: Haiman_x with previous optimal_y_hook_construction ---
  # This explores combining the 'haiman_x' with the "hook" permutation ('optimal_y_hook_construction').
  # We reuse the previously calculated haiman_x and optimal_y_hook_construction.
  hybrid_perms = (haiman_x, optimal_y_hook_construction)
  score_hybrid = get_score(hybrid_perms, n)
  if score_hybrid > best_score:
      best_score = score_hybrid
      best_perms = hybrid_perms

  # Check time after adding a significant construction to avoid timeout during long computation of d_xy
  if time.time() - start_time > time_limit:
      return best_perms


  # --- Fractal/Recursive Constructions ---
  # This "crazy" idea builds permutations recursively. We start with {1..n}
  # and recursively partition the set, building up permutations of the subsets
  # and combining them. This creates self-similar or "fractal" structures.

  @lru_cache(maxsize=None)
  def _build_recursive(numbers: Tuple[int, ...], split_type: str, combine_type: str):
      n_local = len(numbers)
      if n_local <= 1:
          return (numbers, numbers)

      if split_type == 'contiguous':
          m = n_local // 2
          part1 = numbers[:m]
          part2 = numbers[m:]
      else:  # 'interleaved'
          part1 = numbers[::2]
          part2 = numbers[1::2]

      # Recursive calls
      x1, y1 = _build_recursive(part1, split_type, combine_type)
      x2, y2 = _build_recursive(part2, split_type, combine_type)

      # Combine results
      x = x1 + x2
      if combine_type == 'swap':
          y = y2 + y1
      elif combine_type == 'swap_reverse':
          y = tuple(reversed(y2)) + tuple(reversed(y1))
      elif combine_type == 'alternating_merge': # New combine type
          # Helper for alternating merge, specific to this recursive level
          def _merge_alternating_local(p1: Tuple[int, ...], p2: Tuple[int, ...]) -> Tuple[int, ...]:
              res = []
              i, j = 0, 0
              while i < len(p1) or j < len(p2):
                  if i < len(p1):
                      res.append(p1[i])
                      i += 1
                  if j < len(p2):
                      res.append(p2[j])
                      j += 1
              return tuple(res)

          # Combine x and y using alternating merge
          x = _merge_alternating_local(x1, x2)
          # For y, try combining the swapped sub-results with alternating merge
          y = _merge_alternating_local(y2, y1)
      else:
          # Should not happen with defined combine_types, but added for safety
          raise ValueError(f"Unknown combine_type: {combine_type}")

      return x, y

  # The initial set of numbers is {1, ..., n}.
  if n > 1: # Recursive construction is trivial for n<=1 and handled already.
    initial_numbers = tuple(range(1, n + 1))

    for split_t in ['contiguous', 'interleaved']:
        # Added 'alternating_merge' to the list of combine_types to explore
        for combine_t in ['swap', 'swap_reverse', 'alternating_merge']:
            if time.time() - start_time > time_limit:
                break

            _build_recursive.cache_clear()

            try:
              # _build_recursive handles the combination logic based on combine_t
              x_frac, y_frac = _build_recursive(initial_numbers, split_t, combine_t)
              perms_frac = (x_frac, y_frac)
              score_frac = get_score(perms_frac, n)

              if score_frac > best_score:
                  best_score = score_frac
                  best_perms = perms_frac
            except (ValueError, RecursionError):
              # Ignore errors from this experimental construction.
              pass

  # --- Local Search (Simulated Annealing) ---
  # Use the remaining time to refine the best construction found.

  time_remaining = time_limit - (time.time() - start_time)

  # Only run SA if we have a reasonable amount of time left and n is large enough.
  if time_remaining > 0.5 and n >= 2:
    current_perms = best_perms
    current_score_val = best_score

    # Simulated Annealing parameters
    T_initial = 1.0
    T_final = 1e-4

    sa_start_time = time.time()

    while time.time() - start_time < time_limit:
      # Temperature schedule: exponential decay based on overall time elapsed.
      elapsed_time_ratio = (time.time() - start_time) / time_limit

      T = T_initial * (T_final / T_initial)**elapsed_time_ratio
      # Ensure T does not drop below T_final, which can happen with float precision.
      T = max(T, T_final)

      perm_to_mutate_idx = random.randint(0, 1)
      current_perm_list = list(current_perms[perm_to_mutate_idx])

      # Choose mutation type
      mutation_type = random.choice(['swap', 'reverse_segment']) # Add more types here if desired

      if mutation_type == 'swap':
          # Swap two random elements
          if n >= 2:
              i, j = random.sample(range(n), 2)
              current_perm_list[i], current_perm_list[j] = current_perm_list[j], current_perm_list[i]
      elif mutation_type == 'reverse_segment':
          # Reverse a random subsegment
          if n >= 2:
              i, j = sorted(random.sample(range(n), 2))
              current_perm_list[i:j+1] = current_perm_list[i:j+1][::-1]

      mutated_perms_list = list(current_perms)
      mutated_perms_list[perm_to_mutate_idx] = tuple(current_perm_list)
      new_perms = tuple(mutated_perms_list)

      new_score = get_score(new_perms, n)

      # Simulated Annealing acceptance criteria
      if new_score > best_score:
        best_score = new_score
        best_perms = new_perms
        current_perms = new_perms
        current_score_val = new_score
      else:
        delta = new_score - current_score_val 

        # Accept if improvement (delta >= 0) or with probability if worsening (delta < 0).
        if delta >= 0 or (T > T_final and T > 0):
            try:
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_perms = new_perms
                    current_score_val = new_score
            except (OverflowError, ZeroDivisionError):
                # Handle potential overflow in exp or division by zero if T is extremely small
                pass

  return best_perms


# EVOLVE-BLOCK-END


# Scoring functions provided for the d-invariant problem.
# These are outside the evolve block as they define the problem.
@lru_cache(maxsize=None)
def length(w):
  """Calculates the number of inversions in a permutation."""
  inv = 0
  n = len(w)
  for i in range(n):
    for j in range(i + 1, n):
      if w[i] > w[j]:
        inv += 1
  return inv


def right_multiply(w, i):
  """Multiplies on the right by s_i = (i, i+1). i is 1-based."""
  w2 = list(w)
  w2[i - 1], w2[i] = w2[i], w2[i - 1]
  return tuple(w2)


@lru_cache(maxsize=None)
def right_descents(w):
  """Returns a tuple of i (1-based) such that l(w s_i) < l(w)."""
  n = len(w)
  D = []
  lw = length(w)
  for i in range(1, n):
    ws = right_multiply(w, i)
    if length(ws) < lw:
      D.append(i)
  return tuple(D)


@lru_cache(maxsize=None)
def bruhat_leq(x, y):
  """Checks if x <= y in the Bruhat order for S_n using rank tables."""
  n = len(x)
  if n != len(y):
    return False

  def rank_table(w):
    rt = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
      wi = w[i - 1]
      for j in range(1, n + 1):
        rt[i][j] = rt[i - 1][j] + (1 if wi <= j else 0)
    return rt

  Rx = rank_table(x)
  Ry = rank_table(y)
  for i in range(1, n + 1):
    for j in range(1, n + 1):
      if Rx[i][j] < Ry[i][j]:
        return False
  return True


@lru_cache(maxsize=None)
def d_xy(x, y):
  """Recursively computes d_{x,y} for x <= y in Bruhat (S_n)."""
  if x == y:
    return 0
  if not bruhat_leq(x, y):
    raise ValueError('Requires x <= y in Bruhat order.')

  D = right_descents(y)
  if not D:
    raise RuntimeError('y has no right descents; cannot continue.')
  i = D[0]
  ys = right_multiply(y, i)
  xs = right_multiply(x, i)

  if length(xs) < length(x):  # Case 1: xs < x
    return d_xy(xs, ys)
  else:
    # Here xs > x and ys < y. By lifting, x <= ys is always true.
    if not bruhat_leq(xs, ys):  # Case 2: xs not <= ys
      return d_xy(x, ys) + 1
    else:  # Case 3: xs <= ys
      return d_xy(x, ys)


def get_score(perms: Tuple[Tuple[int, ...], Tuple[int, ...]], n: int) -> float:
  """Calculates the score for a pair of permutations."""
  x, y = perms

  # Basic validation
  if len(x) != n or len(y) != n:
    return -1.0
  if set(x) != set(range(1, n + 1)) or set(y) != set(range(1, n + 1)):
    return -1.0

  # d_xy is only defined for x <= y in the Bruhat order.
  # The provided d_xy function will raise an error otherwise.
  try:
    score = d_xy(x, y)
    return float(score)
  except (ValueError, RuntimeError):
    # If x is not <= y or another error occurs, the score is 0.
    return 0.0


def format_feedback_repr(feedback: Mapping[str, Any]) -> Mapping[str, str]:
  """Formats feedback dictionary for representation in code."""
  formatted_feedback = {}
  np.set_printoptions(threshold=np.inf)
  for key, value in feedback.items():
    if isinstance(value, np.ndarray):
      repr_str = repr(value)
      cleaned_repr_str = re.sub(r'[\n\s]+', ' ', repr_str)
      array_content = cleaned_repr_str[6:-1]
      if np.iscomplexobj(value):
        formatted_feedback[key] = (
            f'np.array({array_content}, dtype=np.complex128)'
        )
      elif not np.issubdtype(value.dtype, np.inexact):
        formatted_feedback[key] = f'np.array({array_content}, dtype=np.float64)'
      else:
        formatted_feedback[key] = f'np.array({array_content})'
    elif isinstance(value, (list, tuple)):
      formatted_feedback[key] = repr(value)
    else:
      formatted_feedback[key] = repr(value)
  return formatted_feedback


def evaluate(params: int) -> Tuple[Dict[str, float], Dict[str, str]]:
  """Evaluates a pair of permutations for the d-invariant problem."""
  result = {}
  feedback = {}
  del params

  scores = []
  for n in range(10, 51):
    best_perms = search_for_best_permutations(n)
    feedback['best_perms_%d' % n] = best_perms
    score = get_score(best_perms, n)
    feedback['best_score_%d' % n] = score
    scores.append(score)

  average_score = np.mean(scores)
  print(f'Average score for n={n}: {average_score}')

  result['score'] = average_score
  feedback['best_average_score_found'] = average_score
  feedback = format_feedback_repr(feedback)
  return result, feedback

# score: 71.04878048780488