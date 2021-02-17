# astroSABER

## About
astroSABER (**S**elf-**A**bsorption **B**aseline **E**xtracto**R**) is a baseline fitting routine originally developed to extract the baselines of self-absorption features in HI spectra. The routine makes use of asymmetric least squares smoothing first proposed by [Eilers & Boelens 2005](https://www.researchgate.net/publication/228961729_Baseline_Correction_with_Asymmetric_Least_Squares_Smoothing). The basic principle is to find a solution that minimizes the penalized least squares function:

![\begin{align}
    F(\mathbf{z}) = (\mathbf{y} - \mathbf{z})^\top \mathbf{W} (\mathbf{y} - \mathbf{z}) + \lambda \mathbf{z}^\top \, \mathbf{D}^\top \mathbf{D} \, \mathbf{z} \: ,
\end{align}](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Balign%7D%0A++++F%28%5Cmathbf%7Bz%7D%29+%3D+%28%5Cmathbf%7By%7D+-+%5Cmathbf%7Bz%7D%29%5E%5Ctop+%5Cmathbf%7BW%7D+%28%5Cmathbf%7By%7D+-+%5Cmathbf%7Bz%7D%29+%2B+%5Clambda+%5Cmathbf%7Bz%7D%5E%5Ctop+%5C%2C+%5Cmathbf%7BD%7D%5E%5Ctop+%5Cmathbf%7BD%7D+%5C%2C+%5Cmathbf%7Bz%7D+%5C%3A+%2C%0A%5Cend%7Balign%7D)

where ![\mathbf{y}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7By%7D) is the signal and ![\mathbf{z}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7Bz%7D) is the smoothed signal to be found. The first and second term in Eq.(1) express the fitness of the data and smoothness of ![\mathbf{z}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7Bz%7D) defined by the second order differential matrix ![\mathbf{D}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7BD%7D), respectively.  The parameter ![\lambda](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Ctextstyle+%5Clambda%0A) adjusts the balance between these two terms.
In order to correct the baseline with respect to peaks and dips in the spectrum, the weighting matrix ![\mathbf{W} = \mathrm{diag}(\mathbf{w})
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7BW%7D+%3D+%5Cmathrm%7Bdiag%7D%28%5Cmathbf%7Bw%7D%29%0A) is introduced. The weighting matrix ![\mathbf{W}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Ctextstyle+%5Cmathbf%7BW%7D%0A) is initialized to have ![w_i = 1
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Ctextstyle+w_i+%3D+1%0A). Depending on the deviation of ![\mathbf{z}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7Bz%7D) from ![\mathbf{y}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Ctextstyle+%5Cmathbf%7By%7D) in after the first iteration, the new weights are assigned as follows:

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
