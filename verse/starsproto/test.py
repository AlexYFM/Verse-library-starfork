import numpy as np
from starset import StarSet
from starset import HalfSpace

basis = np.array([[1, 0], [0, 1]])
center = np.array([3,3])

def pred(alpha_vec):
    C = np.array([[1,-1,0,0],[0,0,1,-1]])
    g = np.array([1,1,1,1])
    if np.less_equal(np.multiply(C, alpha_vec),g):
        return True
    return False 

def sim(vec, t):
    A = np.array([[0.1,0],[0,0.1]])
    i = 0
    while i <= t:
        i += 1
        vec = np.matmul(A, vec)
    return vec

test = StarSet(center,basis, pred)
test.show()

new_test = test.post_cont(sim, 1)
new_test.show()


#new = test.superposition([2], [[4]])
#new.show()

#new = new.superposition([2], [[5,6]])

test_half = HalfSpace(np.array([1,1]), 2)

#foo = np.array([3,3])
#bar = 
test_star = StarSet(np.array([3,3]), np.array([[0,5],[0,5]]), pred)

result = test_star.intersection_halfspace(test_half)
result.show()
