# astroSABER

## About
**S**elf-**A**bsorption **B**aseline **E**xtracto**R** (astroSABER) is a baseline fitting routine originally developed to extract the baselines of self-absorption features in HI spectra. The routine makes use of asymmetric least squares smoothing first proposed by [Eilers and Boelens 2005](https://www.researchgate.net/publication/228961729_Baseline_Correction_with_Asymmetric_Least_Squares_Smoothing). The basic principle is to find a solution that minimizes the regularized least squares function:

![\begin{align*}
    F(\mathbf{z}) = (\mathbf{y} - \mathbf{z})^\top (\mathbf{y} - \mathbf{z}) + \lambda \mathbf{z}^\top \, \mathbf{D}^\top \mathbf{D} \, \mathbf{z} \: .
\end{align*}](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A++++F%28%5Cmathbf%7Bz%7D%29+%3D+%28%5Cmathbf%7By%7D+-+%5Cmathbf%7Bz%7D%29%5E%5Ctop+%28%5Cmathbf%7By%7D+-+%5Cmathbf%7Bz%7D%29+%2B+%5Clambda+%5Cmathbf%7Bz%7D%5E%5Ctop+%5C%2C+%5Cmathbf%7BD%7D%5E%5Ctop+%5Cmathbf%7BD%7D+%5C%2C+%5Cmathbf%7Bz%7D+%5C%3A+.%0A%5Cend%7Balign%2A%7D)

## Dependencies
You will need the following packages to run `astroSABER`. We list the version of each package which we know to be compatible with `astroSABER`:

* [python3.6](https://www.python.org/) 
* [astropy (v4.0.2)](https://www.astropy.org/)
* [numpy (v1.19.2)](https://numpy.org/)
* [scipy (v1.5.2)](https://www.scipy.org/)
* [tqdm (v4.56.2)](https://tqdm.github.io/)

## Download astroSABER
Download `astroSABER` using git `$ git clone https://github.com/astrojoni89/astroSABER.git`

## Getting started
You can find an example script for an HI self-absorption (HISA) baseline extraction run in the `example` directory. The data used in this example are taken from The HI/OH Recombination line survey of the inner Milky Way (THOR; [Beuther et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...595A..32B/abstract), [Wang et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..83W/abstract)).
