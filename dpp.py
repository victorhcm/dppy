# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import logging
import sys, os

class DPP:
    def __init__(self, grid_points, sigma_ = 0.1):
        """
        Implements simple grid points using Gaussian kernel
        """
        self.n = len(grid_points)
        self.elements = range(0, self.n*self.n)
        self.sigma = sigma_
        
        # create L-ensemble
        self.xx, self.yy = np.meshgrid(grid_points, grid_points)
        self.xx = self.xx.flatten()[np.newaxis].T
        self.yy = self.yy.flatten()[np.newaxis].T
        self.L = np.exp(-0.5 * ((self.xx - self.xx.T)**2 + (self.yy - self.yy.T)**2) / self.sigma**2)

        # converts index i to equivalent point (a,b)
        self.idx_to_point = np.hstack((self.xx, self.yy))

    def _idxs(self, items):
        idxs_list = [[a] for a in items]
        return idxs_list, items

    def _rev_idxs(self, items):
        # selects the opposite of the items specified
        pass
    
    def prob(self, items):
        return np.linalg.det( self.L[ self._idxs(items) ] )
    
    def cond_prob(self, new_items, sampled = []):
        """TODO implement the conditioning function"""
        if not sampled:
            return self.prob(new_items)
        
        items = new_items + sampled
        conj = np.linalg.det( self.L[ self._idxs(items) ] )
        
        # creating Y - A, i.e. Y_omega - sampled
        Y_omega = range(0, self.n)
        Y_inv = [y for y in Y_omega if y not in sampled]
        
        # invA or inv_sampled
        I_invA = np.zeros_like(self.L)
        I_invA[Y_inv, Y_inv] = 1
        
        # normalization 
        norm = np.linalg.det( self.L + I_invA )
        return conj / float(norm)
    
    def dummy_sampling(self, num_samples, verbose = False):      
        Y = list(self.elements)
        sampled = []
        for it in xrange(num_samples):
            if verbose: print "Iteration: %d" % it
            
            prbs  = [self.cond_prob([y], sampled) for y in Y]
            prbs /= np.sum(prbs)
            
            elem_sampled = np.random.choice(Y, 1, p=prbs)
            sampled.append(elem_sampled)
            
            # creating Y - A, i.e. Y_omega - sampled
            Y.remove(elem_sampled)
            
            if verbose:
                print "Y", Y
                print "p", prbs
                print "s", sampled
                print 
            
        return map(int, sampled)
    
    def sample_dpp(self, k = None):
        logging.basicConfig(level=logging.DEBUG, format='%(name)s (%(levelname)s): %(message)s')
        module = sys.modules['__main__'].__file__
        log = logging.getLogger(module)

        # decomposes kernel
        D, V = np.linalg.eig(self.L) # D: eigenvalues, V: eigenvectors

        if k == None:
            # choose eigenvectors randomly
            D = D / (1 + D)
            v_idx = np.where(np.random.rand(len(D)) <= D)[0]
        else:
            # k-DPP
            print "not implemented yet"
            sys.exit(-1)

        k = len(v_idx)
        V = V[:,v_idx]

        Y = np.zeros(k)
        for i in xrange(k-1, -1, -1): # from k-1 until 0

            # compute probabilities for each item
            P = np.sum(V**2, axis = 1)
            P = P / np.sum(P)
            log.debug("P:\n%s", P)

            # choose a new item to include
            Y[i] = np.nonzero(np.random.rand(1) <= np.cumsum(P))[0][0] # takes the first non-zero element
            log.debug("cumsum P: %s", np.cumsum(P))
            log.debug("Y: %s", Y)

            # choose a vector to eliminate
            j = np.nonzero( V[Y[i],:] )[0][0]
            log.debug( "js (idx vec to eliminate) = %s", np.nonzero(V[Y[i],:]) )
            Vj = V[:,j]
            V = np.delete(V, j, 1)
            log.debug("j: %s, Vj: %s, V: %s", j, Vj.shape, V.shape)

            # update V
            log.debug("Shapes => Vj %s, V[Y[i],:] %s", Vj.shape, V[ Y[i],: ].shape)
            # Vj = Vj[:,np.newaxis]         # Vj must be a column vector, eg. (100, 1)
            # V_Yi = V[Y[i], np.newaxis, :] # V at Y[i] must be row-vector, eg. (36, 1)
            log.debug(Vj[:,np.newaxis].shape)
            log.debug(V[ Y[i], np.newaxis, : ].shape)
            log.debug((Vj[:, np.newaxis] * V[ Y[i], np.newaxis, : ]).shape)
            V = V - (Vj * V[ Y[i],: ]) / Vj[ Y[i] ]



if __name__ == "__main__":
    n = 10
    grid_points = np.arange(n) / float(n)
    dpp_grid = DPP(grid_points)

    print "sanity check"
    print "i = {0,1,2}:", dpp_grid.prob([0,1,2])
    print "i = {1,2,3}:", dpp_grid.prob([1,2,3])
    print 
    print "sampling distance points in the grid:"
    print "i = {0,50,99}:", dpp_grid.prob([0,50,99]) # selecting distant points in the diagonal
    print "i = {0,33,77}:", dpp_grid.prob([0,33,77])
    print "i = {0,10,20}:", dpp_grid.prob([0,10,20])
    print "i = {0,5,10}:", dpp_grid.prob([0,5,10])
    print
    print "conditional probabilities"
    print "i = {0,1,2}:", dpp_grid.prob([0,1,2])
    print "B = {3} | A = {0,1,2}:", dpp_grid.cond_prob([3], [0,1,2])
    print "B = {4} | A = {0,1,2}:", dpp_grid.cond_prob([4], [0,1,2])
    print "B = {9} | A = {0,1,2}:", dpp_grid.cond_prob([9], [0,1,2])
    print 
    print "B = {2} | A = {0,1}:", dpp_grid.cond_prob([2], [0,1])
    print "B = {3} | A = {0,1}:", dpp_grid.cond_prob([3], [0,1])
    print "B = {99} | A = {0,50}:", dpp_grid.cond_prob([99], [0,50])
    # p.s: you need to convert idx_to_point to see which point it is

    dpp_grid.sample_dpp()


