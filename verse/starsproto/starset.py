import numpy as np


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
        predicate,
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
        self.center = center
        for vec in basis:
            if len(vec) != self.n:
                raise Exception("Basis for star set must be the same dimension as center")
        self.basis = basis
        self.predicate = predicate


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
            return StarSet(new_center, new_basis, self.predicate)
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
        print(self.predicate)


    '''
    starset intsersection of this star set a halfspace
    '''
    def intersection_halfspace(self,hspace):
        def new_pred(alpha):
            left = np.matmul(np.matmul(hspace.H, self.basis), alpha)
            right = np.subtract(hspace.g, np.matmul(hspace.H, self.center))
            return np.less(left, right)
        def conjunction_pred(alpha):
            return new_pred(alpha) and self.predicate(alpha)
        return StarSet(self.center, self.basis, conjunction_pred)


    def intersection_poly():
        return None


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
        p_prime = intermediate #np.add(intermediate,self.center)
        #print("this is alpha!!!")
        #print(p_prime)
        return self.predicate(p_prime)


    '''
   returns true if entire star set is contained within the half_space
    '''
    def satisfies():
        return None

    '''
   returns true if star set intersects the half space
    '''
    def intersects():
        return None

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
