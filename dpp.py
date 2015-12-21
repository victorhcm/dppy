# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import logging
import sys, os, math

class DPP:
    def __init__(self, grid_points, sigma_ = 0.1):
        """
        Implements simple grid points using a Gaussian kernel
        """
        self.n = len(grid_points)
        self.itemset = range(0, self.n*self.n)

        # parameters
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
    
    def L_sel(self, items):
        """returns L_Y, that is, L indexed by elements Y"""
        return self.L[ self._idxs(items) ]

    # def L_sel(self, L_Y, items):
    #     """returns L_Y, that is, L indexed by elements Y"""
    #     return L_Y[ self._idxs(items) ]

    def prob(self, items):
        return np.linalg.det( self.L[ self._idxs(items) ] )
    
    def cond_prob(self, new_items, sampled = []):
        """implement the conditioning function"""
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
        Y = list(self.itemset)
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
        logging.basicConfig(level=logging.INFO, format='%(name)s (%(levelname)s): %(message)s')
        # module = sys.modules['__main__'].__file__
        log = logging.getLogger()

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
            log.debug("Shapes => j: %s, Vj: %s, V: %s", j, Vj.shape, V.shape)

            # update V
            log.debug("Shapes => Vj %s, V[Y[i],:] %s", Vj.shape, V[ Y[i],: ].shape)
            Vj = Vj[:,np.newaxis]           # Vj must be a column vector, eg. (100, 1)
            # V_Yi = V[Y[i], np.newaxis, :] # V at Y[i] must be row-vector, eg. (36, 1)
                                            # but it's not required, it's already row
            log.debug(Vj.shape)
            log.debug(V[Y[i],:].shape)
            log.debug((Vj * V[Y[i], :]).shape)

            V = V - (Vj * V[ Y[i],: ]) / Vj[ Y[i] ]

            # orthogonalize
            for a in xrange(0,i):
                for b in xrange(0,a):
                    V[:,a] = V[:,a] - V[:,a].T * V[:,b] * V[:,b]
                log.debug(np.linalg.norm(V[:,a]))
                V[:,a] = V[:,a] / np.linalg.norm(V[:,a])

        return sorted(map(int, Y))

    def mh_sampler(self, epsilon=0.01):
        """
        Metropolis-Hastings algorithm as proposed by [Fast Determinantal Point
        Process Sampling with Application to Clustering][1].

        Inputs:
        - epsilon: parameter used to determine the number of iterations. Authors
                   used 0.01
        Outputs:
        - Y: list of samples from the itemset

        [1]: http://papers.nips.cc/paper/\
            5008-fast-determinantal-point-process-\
            sampling-with-application-to-clustering.pdf
        """
        logging.basicConfig(level=logging.DEBUG, format='%(name)s (%(levelname)s): %(message)s')
        log = logging.getLogger()

        # randomly initialize state $Y \subseteq S$.  it's recommended to
        # initialize with a small size, as $o(n^{1/3})$ to avoid initial
        # expensive inverse computations.
        ini_sample_sz = self.n**(float(1)/3)
        Y = list(np.random.choice(self.itemset, ini_sample_sz, replace=False))
        log.debug("Initial sample size: %s", ini_sample_sz)
        log.debug("Sampled elements: %s", Y)

        # mixing time is $O(n log(n/epsilon)$
        niter = int(self.n * math.log(self.n / epsilon))
        log.debug("Number of iterations: %s", niter)

        for i in xrange(niter):
            u = np.random.choice(self.itemset, 1, replace=False)

            # temporary lists to compute the probabilities
            Ypu = list(Y); Ynu = list(Y)
            if u not in Y:
                Ypu.append(u)
            else:
                Ynu.remove(u)

            log.debug("Y:\t%s", Y)
            log.debug("Ypu:\t%s", Ypu)
            log.debug("Ynu:\t%s", Ynu)

            # probabilities of inclusion/removal of item u
            prob_Y = self.prob(Y)
            pu_pos = min(1, self.prob(Ypu) / prob_Y)
            pu_neg = min(1, self.prob(Ynu) / prob_Y)

            log.debug("pu_pos:\t%s", pu_pos)
            log.debug("pu_neg:\t%s", pu_neg)

            # TODO check if rand(1) >= pu_pos is correct
            if u not in Y:
                # includes u with probability pu_pos
                add_elem = np.random.rand(1) <= pu_pos
                if add_elem: 
                    previous_sz = len(Y)
                    Y.append(int(u))
                    log.debug('adding elem u: %s. '
                              'Y: %s to %s elements', u, previous_sz, len(Y))
            else:
                # removes u with probability pu_neg
                rem_elem = np.random.rand(1) <= pu_neg
                if rem_elem:
                    previous_sz = len(Y)
                    Y.remove(int(u))
                    log.debug('removing elem u: %s. '
                              'Y: %s to %s elements', u, previous_sz, len(Y))

        return Y


    def mh_fast_sampler(self, epsilon=0.01):
        """
        Faster Metropolis-Hastings algorithm as proposed by [Fast Determinantal
        Point Process Sampling with Application to Clustering][1]. Instead of
        computing the determinant every time, which is expensive, computes a
        determinant-free ratio. It still requires to compute the inverse of
        L^-1_Y, but it is updated every time in O(Y^2) instead of O(Y^3).

        Inputs:
        - epsilon: parameter used to determine the number of iterations. Authors
                   used 0.01
        Outputs:
        - Y: list of samples from the itemset

        [1]: http://papers.nips.cc/paper/\
            5008-fast-determinantal-point-process-\
            sampling-with-application-to-clustering.pdf
        """
        logging.basicConfig(level=logging.DEBUG, format='%(name)s (%(levelname)s): %(message)s')
        log = logging.getLogger()

        # randomly initialize state $Y \subseteq S$.  it's recommended to
        # initialize with a small size, as $o(n^{1/3})$ to avoid initial
        # expensive inverse computations.
        ini_sample_sz = self.n**(float(1)/3)
        Y = list(np.random.choice(self.itemset, ini_sample_sz, replace=False))
        log.debug("Initial sample size: %s", ini_sample_sz)
        log.debug("Sampled elements: %s", Y)

        # mixing time is $O(n log(n/epsilon))$
        niter = int(self.n * math.log(self.n / epsilon))
        log.debug("Number of iterations: %s", niter)

        # precomputing the inverse of L_Y, which will be updated iteratively
        L_Y = self.L_sel(Y)
        L_Y_inv = np.linalg.inv(L_Y)

        for i in xrange(niter):
            # randomly selects one element
            u = np.random.choice(self.itemset, 1, replace=False)

            # probabilities of inclusion/removal of item u
            b_u = self.L[Y,u][:, np.newaxis]
            c_u = self.L[u,u]

            log.debug('b_u: %s', b_u.shape)
            log.debug('c_u: %s', c_u.shape)
            log.debug('Y: %s', len(Y))
            log.debug('L_Y:\n%s', L_Y.shape)

            ratio = np.dot(np.dot(b_u.T, L_Y_inv), b_u)
            d_u   = c_u - ratio
            pu_pos = min(1, d_u)
            pu_neg = min(1, d_u**(-1))

            log.info("pu_pos:\t%s", pu_pos)
            log.info("pu_neg:\t%s", pu_neg)

            if u not in Y:
                # includes u with probability pu_pos
                add_elem = np.random.rand(1) <= pu_pos
                if add_elem: 
                    previous_sz = len(Y)
                    Y.append(int(u))
                    log.debug('adding elem u: %s. '
                              'Y: %s to %s elements', u, previous_sz, len(Y))

                    # updates the inverse
                    log.debug('previous L_Y_inv: %s', L_Y_inv.shape)

                    upper11 = L_Y_inv + (np.dot(np.dot(np.dot(L_Y_inv, b_u), b_u.T), L_Y_inv) / d_u)
                    upper12 = -(np.dot(L_Y_inv, b_u) / d_u)
                    under11 = -(np.dot(b_u.T, L_Y_inv) / d_u)
                    under12 = d_u
                    upper = np.hstack((upper11, upper12))
                    under = np.hstack((under11, under12))
                    L_Y_inv = np.vstack((upper, under))

                    log.debug('after L_Y_inv: %s', L_Y_inv.shape)
            else:
                # removes u with probability pu_neg
                rem_elem = np.random.rand(1) <= pu_neg
                if rem_elem:
                    previous_sz = len(Y)
                    Y.remove(int(u))
                    log.debug('removing elem u: %s. '
                              'Y: %s to %s elements', u, previous_sz, len(Y))

                    # updates the inverse
                    log.debug('previous L_Y_inv: %s', L_Y_inv.shape)
                    L_Y = self.L_sel(Y)

                    # FIXME u is a item from the whole matrix L. I must compute the equivalent u'
                    # for the submatrix L_Y
                    Y_sorted = Y.sort()
                    sub_u = Y_sorted.index(u)
                    log.debug('Y_sorted: %s', Y_sorted)
                    log.debug('u: %s => sub_u: %s', u, sub_u)

                    D = np.delete(np.delete(L_Y, sub_u, axis=0), sub_u, axis=1) # removes column and row u

                    log.debug('sub_u: %s %s', sub_u, b_u.shape)
                    e = np.delete(b_u, sub_u, axis = 0)

                    log.debug('D: %s', D.shape)
                    log.debug('e: %s', e.shape)

                    L_Y_inv = D - np.dot(e, e.T) / c_u
                    log.debug('after L_Y_inv: %s', L_Y_inv.shape)

        return Y




if __name__ == "__main__":
    # n = 10
    # grid_points = np.arange(n) / float(n)
    # dpp_grid = DPP(grid_points)

    # print "sanity check"
    # print "i = {0,1,2}:", dpp_grid.prob([0,1,2])
    # print "i = {1,2,3}:", dpp_grid.prob([1,2,3])
    # print 
    # print "sampling distance points in the grid:"
    # print "i = {0,50,99}:", dpp_grid.prob([0,50,99]) # selecting distant points in the diagonal
    # print "i = {0,33,77}:", dpp_grid.prob([0,33,77])
    # print "i = {0,10,20}:", dpp_grid.prob([0,10,20])
    # print "i = {0,5,10}:", dpp_grid.prob([0,5,10])
    # print
    # print "conditional probabilities"
    # print "i = {0,1,2}:", dpp_grid.prob([0,1,2])
    # print "B = {3} | A = {0,1,2}:", dpp_grid.cond_prob([3], [0,1,2])
    # print "B = {4} | A = {0,1,2}:", dpp_grid.cond_prob([4], [0,1,2])
    # print "B = {9} | A = {0,1,2}:", dpp_grid.cond_prob([9], [0,1,2])
    # print 
    # print "B = {2} | A = {0,1}:", dpp_grid.cond_prob([2], [0,1])
    # print "B = {3} | A = {0,1}:", dpp_grid.cond_prob([3], [0,1])
    # print "B = {99} | A = {0,50}:", dpp_grid.cond_prob([99], [0,50])
    # # p.s: you need to convert idx_to_point to see which point it is

    # # sampling only 10 elements
    # sampled_idxs = dpp_grid.sample_dpp()
    # sampled_points = dpp_grid.idx_to_point[sampled_idxs]
    # plt.scatter(sampled_points[:,0], sampled_points[:,1])
    # plt.show()

    # metropolis-hastings sampling is much faster, so I can quickly sample 100
    # elements
    n = 100
    grid_points = np.arange(n) / float(n)
    dpp_grid = DPP(grid_points)

    sampled_idxs = dpp_grid.mh_fast_sampler()
    sampled_points = dpp_grid.idx_to_point[sampled_idxs]
    plt.scatter(sampled_points[:,0], sampled_points[:,1])
    plt.show()


