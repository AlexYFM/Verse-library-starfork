import numpy as np
import copy
from scipy.optimize import linprog
import matplotlib.pyplot as plt

class StarSet:
    """
    StarSet
    a one dimensional star set

    Methods
    -------
    superposition
    from_polytope
    intersection_halfspace
    intersection_poly
    contains_point
    satisfies
    is_empty
    """

    def __init__(
        self,
        center,
        basis,
        #predicate,
        C,
        g
    ):
        """
        

        Parameters
        ----------
        center : np array of numbers
            center of the starset.
        basis : nparray  of nparray of numbers
            basis of the starset
        predicate: boolean function
            function that gives the predicate
        """
        self.n = len(center)
        self.m = len(basis)
        self.center = np.copy(center)
        for vec in basis:
            if len(vec) != self.n:
                raise Exception("Basis for star set must be the same dimension as center")
        self.basis = np.copy(basis)
        if C.shape[1] != self.m:
            raise Exception("Width of C should be equal to " + str(m))
        if len(g) !=  len(C):
            raise Exception("Length of g vector should be equal length of C")
        self.C = np.copy(C)
        self.g = np.copy(g)


    def superposition(self, new_center, new_basis):
        """
        superposition
        produces a new starset with the new center and basis but same prediate

        Parameters
        ----------
        new_center : number
            center of the starset.
        new_basis : number
            basis of the starset
        """
        if len(new_basis) == len(self.basis):
            return StarSet(new_center, new_basis, self.C, self.g)
        raise Exception("Basis for new star set must be the same")

    '''
   prototype function for now. Will likley need more args to properly run simulation
    '''
    def post_cont(self, simulate, t):
        new_center = simulate(self.center,t)
        new_basis = np.empty_like(self.basis) 
        for i in range(0, len(self.basis)):
            vec = self.basis[i]
            new_x = simulate(np.add(self.center, vec), t)
            new_basis[i] = np.subtract(new_x, new_center)
        return self.superposition(new_center, new_basis)


    def show(self):
        print(self.center)
        print(self.basis)
        print(self.C)
        print(self.g)

    def get_halfspace_intersection(starset, constraint_vec, rhs_val):
        #starset.show()
        star_copy = StarSet(starset.center, starset.basis, starset.C, starset.g)
        #star_copy.show()
        star_copy.intersection_halfspace(constraint_vec, rhs_val)
        return star_copy

    '''
    starset intsersection of this star set and a halfspace
    '''
    def intersection_halfspace(self,constraint_vec, rhs_val):
        if not (constraint_vec.ndim == 1) or not (len(constraint_vec == self.n)):
            raise Exception("constraint_vec should be of length n")
        self.intersection_poly(np.array([constraint_vec]), np.array([rhs_val]))

    def intersection_poly(self, constraint_mat, rhs):
        #constraint mat should be length of n and heigh of j and rhs should have length of j
        if not (len(constraint_mat[0] == self.n)):
            raise Exception("constraint_mat vectors should be of length n")
        if not (len(rhs) == len(constraint_mat)):
            raise Exception("constraint_mat should be length of rhs")
        new_c = np.matmul(constraint_mat, self.basis)
        conj_c = np.vstack((self.C, new_c))
        new_g = np.subtract(rhs, np.matmul(constraint_mat, self.center))
        conj_g = np.append(self.g, new_g) 
        self.C = conj_c 
        self.g = conj_g 
#        return None




    def contains_point(self, pt):
        raise Exception("not implemented")
        if not (pt.ndim == 1) and not (len(pt) == self.n):
            raise Exception("pt should be n dimensional vector")
        #affine transformation of point with baisis as generator and cneter as offset
        #then check if 
        #print(self.basis)
        intermediate = np.matmul(np.transpose(self.basis), pt) 
        #print(intermediate)
        p_prime = np.add(intermediate,self.center)
        #print("this is alpha!!!")
        #print(p_prime)
        print(p_prime)
        print(self.C)
        return False #self.predicate(p_prime)

    def from_poly(constraint_mat, rhs):
        if not (len(rhs) == len(constraint_mat)):
            raise Exception("constraint_mat should be length of rhs")
        n = len(constraint_mat[0])
        center = np.zeros(n)
        basis = np.zeros((n, n))
        for index in range(0,n):
            basis[index][index] = 1.0
        return StarSet(center,basis, constraint_mat, rhs)

    def to_poly(self):
        raise Exception("to_poly not implemented")
        #new_constraint_mat =np.matmul(self.C,np.linalg.inv(self.basis)) - self.center
        #new_rhs = self.g
        new_constraint_mat =np.matmul(self.C,self.basis) + self.center
        new_rhs = self.g

        return (new_constraint_mat, new_rhs)

    '''
   returns true if entire star set is contained within the half_space A*x <= b
    '''
    def satisfies(self,constraint_vec, rhs):
        #check if the intersection Ax > b is emtpy. We can only check <= in scipy
        #TODO: improve to find check -Ax < -b instead of -Ax <= -b
        new_star = StarSet.get_halfspace_intersection(self, -1 * constraint_vec,-1*rhs)
        #new_star.show()
        #if new star is empty, this star is entirely contained in input halfspace
        return new_star.is_empty()

    def union():
        return None

    def intersect(star1, star2):
        return None

    def plot(self):
        xs, ys = StarSet.get_verts(self)
        #print(verts)
        plt.plot(xs, ys)
        plt.show()

    #stanley bak code
    def get_verts(stateset):
        """get the vertices of the stateset"""

        verts = []
        x_pts = []
        y_pts = []
        # sample the angles from 0 to 2pi, 100 samples
        for angle in np.linspace(0, 2*np.pi, 100):
            x_component = np.cos(angle)
            y_component = np.sin(angle)
            direction = np.array([[x_component], [y_component]])

            pt = stateset.maximize(direction)

            verts.append(pt)
            #print(pt)
            x_pts.append(pt[0][0])
            print(pt[0][0])
            y_pts.append(pt[1][0])
            print(pt[1][0])
        return (x_pts, y_pts)

#stanley bak code
    def maximize(self, opt_direction):
        """return the point that maximizes the direction"""

        opt_direction *= -1

        # convert opt_direction to domain
        domain_direction = opt_direction.T @ self.basis

        # use LP to optimize the constraints
        res = linprog(c=domain_direction, 
                A_ub=self.C, 
                b_ub=self.g, 
                bounds=(None, None))
        
        # convert point back to range
        #print(res.x)
        domain_pt = res.x.reshape((res.x.shape[0], 1))
        range_pt = self.center + self.basis @ domain_pt
        
        # return the point
        return range_pt


#    '''
#   returns true if star set intersects the half space
#    '''
#    def intersects():
#        return None

    def is_empty(self):
        feasible = StarSet.is_feasible(self.n,self.C,self.g)
        if feasible:
            return False
        return True

    def is_feasible(n, constraint_mat, rhs, equal_mat=None, equal_rhs=None):
        results = linprog(c=np.zeros(n),A_ub=constraint_mat,b_ub=rhs,A_eq=equal_mat, b_eq=equal_rhs)
        if results.status == 0:
            return True
        if results.status == 2:
            return False
        raise Exception("linear program had unexpected result")


class HalfSpace:
    '''
    Parameters
    --------------------
    H : vector in R^n
    g : value in R 
    '''
    def __init__(self, H, g):
        self.H = H
        self.g = g
