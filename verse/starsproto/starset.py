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
        center : number
            center of the starset.
        basis : nparray of numbers
            basis of the starset
        predicate: boolean function
            function that gives the predicate
        """
        self.center = center
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
        raise Exception("Basis for star set must be the same dimension")


    def show(self):
        print(self.center)
        print(self.basis)
        print(self.predicate)
