import numpy as np


class StarSet:
    """
    StarSet
    a one dimensional star set

    Methods
    -------
    update_center
    update_bais
    
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
        basis : number
            basis of the starset
        predicate: boolean function
            function that gives the predicate
        """
        self.center = center
        self.basis = basis
        self.predicate = predicate

