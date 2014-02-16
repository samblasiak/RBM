RBM
===

A Restricted Boltzmann Machine implementation in Python suitable for small-scale applications.

Two RBM versions are currently implemented
    -BasicRBM - conditional distributions on both the hidden and observed variables are logistic
    -GausRBM  - the conditional distribution on the observed variables is Gaussian
                -GausRBM is either incorrect or is much tricker to train than the BasicRBM

Gradients for both RBM versions are approximated using contrastive divergence with one sampling iteration (CD-1)
