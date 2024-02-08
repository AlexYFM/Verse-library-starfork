import numpy as np
from starset import StarSet
from starset import HalfSpace

basis = np.array([[2, 0], [0, 2]])
center = np.array([3,3])

#def pred(alpha_vec):
    #print("in predicate")
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
    # intermediate = C @ alpha_vec
    #print(alpha_vec)
    #print(np.multiply(C, alpha_vec))
    #print(intermediate)
    #if (np.less_equal(intermediate,g)).all() == True:
    #    return True
    #return False 

def sim(vec, t):
    A = np.array([[0.1,0],[0,0.1]])
    i = 0
    while i <= t:
        i += 1
        vec = np.matmul(A, vec)
    return vec

#basis_rot = np.array([[0.707,0.707],[-0.707,0.707]])

test = StarSet(center,basis, C, g)


test.contains_point(np.array([3,2]))
print("done with contains test")
exit()
test.plot()


test.is_empty()
print("orig")
test.show()

print(test.satisfies(np.array([1,0]), -2))
print(test.satisfies(np.array([1,0]), 10))

print("test star set after")
test.show()

test.intersection_halfspace(np.array([5,5]), 3)
print("add single const")
test.show()
print("add multiple constr")
test.intersection_poly(np.array([[8,8],[9,9]]), np.array([4,5]))
test.show()

#print("results!")

#print(test.contains_point(np.array([1,0])))
#print(test.contains_point(np.array([3,3])))
#print(test.contains_point(np.array([2,2])))
#print(test.contains_point(np.array([4,2])))
#print(test.contains_point(np.array([4,1])))


new_test = test.post_cont(sim, 1)
new_test.show()


print("from poly test")
new_star = StarSet.from_poly(np.array([[8,8],[9,9]]), np.array([4,5]))
new_star.show()


#print("to poly test")
polystar = StarSet(center, basis,C, g)
#mat, rhs = polystar.to_poly()
#print(mat)
#print(rhs)
#print(np.matmul(mat, [3,3]))

#print(polystar.satisfies(np.array([[1,1]]),np.array([7])))

#print("verts test")
#print(StarSet.get_verts(polystar))

#new = test.superposition([2], [[4]])
#new.show()

#new = new.superposition([2], [[5,6]])

#test_half = HalfSpace(np.array([1,1]), 2)

#foo = np.array([3,3])
#bar = 
#test_star = StarSet(np.array([3,3]), np.array([[0,5],[0,5]]), C, g)

#result = test_star.intersection_halfspace(test_half)
#result.show()
