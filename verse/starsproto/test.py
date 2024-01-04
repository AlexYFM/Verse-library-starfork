import numpy as np
from starset import StarSet
from starset import HalfSpace

basis = np.array([[1, 0], [0, 1]])
center = np.array([3,3])

def pred(alpha_vec):
    print("in predicate")
    C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
    g = np.array([1,1,1,1])
    intermediate = C @ alpha_vec
    #print(alpha_vec)
    #print(np.multiply(C, alpha_vec))
    #print(intermediate)
    if (np.less_equal(intermediate,g)).all() == True:
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

print("results!")

print(test.contains_point(np.array([1,0])))
print(test.contains_point(np.array([3,3])))
print(test.contains_point(np.array([2,2])))
print(test.contains_point(np.array([4,2])))
print(test.contains_point(np.array([4,1])))


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
