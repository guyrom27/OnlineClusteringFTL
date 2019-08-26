This is the code used to generate the figure for my dissertation on online kmeans clustering.
 
The jupyter notebook FTL.ipynb contains the simulation and plotting of the caounter example that shows that Follow-The-Leader (FTL) has linear regret for this problem. The following is the regret for an online algorithm was halted at different time steps T, of both FTL and MWUA (Multiplicative Weights Update Algorithm) that is run on the the historical FTL leaders.

![linear regret](https://github.com/guyrom27/OnlineClusteringFTL/blob/master/figures/FTL_vs_sim_and_MNIST.pdf)

The jupyter notebook figures.ipynb contains the plotting of the different FTL natural simulations. Gaussians.py contains the GMM simulations. random_order.py contains the MNIST simulations.

The following is the regret halted at T vs. log(T) for FTL on MNIST, and on Gaussian Mixture Models using the correct k value for kmeans or incorrect. It shows that in these cases FTL obtains logarithmic regret.

![logarithmic regret](https://github.com/guyrom27/OnlineClusteringFTL/blob/master/figures/ftl_counterexample.pdf)

The Dissertation itself can be found here- [Thesis](https://github.com/guyrom27/OnlineClusteringFTL/blob/master/Thesis-Online Clustering.pdf)

The paper version can be found here- [Paper](https://github.com/guyrom27/OnlineClusteringFTL/blob/master/paper-online_clustering.pdf)

