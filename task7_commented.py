def calc_Dw(pval, step=0.0001):
    '''
    Parameters
    ----------
    pval : float
        The p-value for the waveguide.
    step : float, optional
        How granular you want the step to be for the central difference scheme. The default is 0.0001.

    Returns
    -------
    float
        The result of the numerical differentiation using the central difference scheme.

    '''
    c = 3e10 # Speed of Light
    h = wavelength * step # Define h to be in terms of the original wavelength
    
    wave0 = wavelength - wavelength * step # f(x - h)
    wave1 = wavelength # f(x)
    wave2 = wavelength + wavelength * step #f(x + h)
    
    # Calculate the effective index for all wavelengths
    neff_0 = calc_neff(pval, (2 * np.pi) / (wave0 * n1))
    neff_1 = calc_neff(pval, (2 * np.pi) / (wave1 * n1))
    neff_2 = calc_neff(pval, (2 * np.pi) / (wave2 * n1))
    
    return - (wavelength / c) * ((neff_2 - (2 * neff_1) + neff_0) / h**2) # central differentiation scheme

def calc_neff(pval, k):
    # Simple function for calculating the effective index
    beta = np.sqrt(n1 ** 2 * k ** 2 - pval ** 2) # Formula for beta using p value
    return beta / k # Effective index
