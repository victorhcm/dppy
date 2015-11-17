
# TODO

- Conditioning experiment: given that group $i = {...}$ was already selected, what is the probability of selecting another element/set. We can notice that we may try to do a sampling algorithm by using conditioning. Given that one element was already selected, we condition on this set and recompute the marginal probabilities of every other set, then we sample a new element with its probability $P(\mathbf{Y} = A \cup B | A \subseteq \mathbf{Y})$. Of course this is not efficient, but aids at better understanding the algorithm based on the eigendecomposition.

- Compute the Kernel $K$ given the $L$-ensemble $L$ that you defined in your experiment

## Notes

- Notice that we are really defining the ensemble $L$, as the sampling algorithm requires the eigendecomposition of $L$, not $K$.
