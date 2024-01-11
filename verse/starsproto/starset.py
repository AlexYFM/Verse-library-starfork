import numpy as np
import copy

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


    def contains_point_redo(self,pt):
        if not (pt.ndim == 1) and not (len(pt) == self.n):
            raise Exception("pt should be n dimensional vector")


    def contains_point(self, pt):
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


    '''
   returns true if entire star set is contained within the half_space
    '''
    def satisfies():
        #union with the polytope and check if empty
        return None

#    '''
#   returns true if star set intersects the half space
#    '''
#    def intersects():
#        return None

    def is_empty():
        return None


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
