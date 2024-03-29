
# TODO

- [x] Compute the Kernel $K$ given the $L$-ensemble $L$ that you defined in your experiment

- [x] Conditioning experiment: given that group $i = {...}$ was already selected, what is the probability of selecting another element/set. We see that it's possible to do a sampling algorithm by using conditioning. Given that one element was already selected, we condition on this set and recompute the marginal probabilities of every other set, then we sample a new element with its probability $P(\mathbf{Y} = A \cup B | A \subseteq \mathbf{Y})$. Of course this is not efficient, but aids at better understanding the algorithm based on the eigendecomposition.

- [x] Reimplement sampling for a grid. It seems that I'm sampling for a line of elements. What I need to do is sample twice, considering two lines. But, I need to check, because in this way, I would be sampling one axis independent of the other one


## Ideas

- Compare DPP with Markov Random Fields
- We can use DPP as a prior, e.g., for Latent Dirichlet Allocation
- Sampling can be improved by sampling several points and then taking the mean of the sampling
- Sampling can be even more improved using Metropolis-Hastings or Gibbs Sampler
- Testing DPP to select pedestrians

## Notes

- Notice that we are really defining the ensemble $L$, as the sampling algorithm requires the eigendecomposition of $L$, not $K$.
