import numpy as np


class StarSet:
    """
    StarSet
    a one dimensional star set

    Methods
    -------
    superposition
    
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

    def Post_cont(self, simulate, t):
        new_center = simulate(self.center,t)
        new_basis = np.empty_like(self.basis) 
        for i in range(0, len(self.basis)):
            vec = self.basis
            new_x = simulate(np.add(self.center, vec), t)
            new_basis[i] = np.subtract(new_x, new_center)
        return superposition(self, new_center, new_basis)


    def show(self):
        print(self.center)
        print(self.basis)
        print(self.predicate)
