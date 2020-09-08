# python 3.6
# author Weikang Lin

#### Define my own Background functions:


import numpy as np
from scipy import integrate

# fitting constants for the massive neutrino transitions:
n_w = 1.8367
alpha_w = 3.1515
Tv = 1.945/1.160451812E4  # massless neutrino temperature in eV


# some background function:
# Flat LCDM with neutrinos
# first take normal hierarchy with one massless, needs to be generalized into the other hierarchy with another one massive as well
def Ez_flat_LCDM(z,Om,h=0.7, m_v1=0.0):  # Integrand of the comoving distance in the unit of 1/H0
    a=1./(1.+z)
    Ogamma = 2.472E-5/h**2  # O_gamma = 2.472/h^2 * 10^-5
    # neutrinos: take 1 massless neutrinos, normal hierarchy, ignore mass^2 difference uncertainties:
    if m_v1==0.0:
        Ov1 = 5.617E-6/h**2  # massless Ov_massless = 5.617/h^2 * 10^-6
        E_v1 = Ov1*(1.0+z)**4
    else:
        Ov1 = m_v1/93.14/h**2  # these treatments are good for m_v1=0 or m_v1>2K
        aT_v1 = Tv/m_v1*alpha_w
        E_v1 = Ov1/a**4*((a**n_w+aT_v1**n_w)/(1.+aT_v1**n_w))**(1./n_w)

    m_v2 = (7.9E-5+m_v1**2)**0.5
    Ov2 = m_v2/93.14/h**2  # massive Ov_massive = m(eV)/93.14/h^2
    aT_v2 = Tv/m_v2*alpha_w
    E_v2 = Ov2/a**4*((a**n_w+aT_v2**n_w)/(1.+aT_v2**n_w))**(1./n_w)
    
    m_v3 = (2.2E-3+m_v1**2)**0.5
    Ov3 = m_v3/93.14/h**2 #0.001028  # massive Ov_massive = m(eV)/93.14/h^2
    aT_v3 = Tv/m_v3*alpha_w
    E_v3 = Ov3/a**4*((a**n_w+aT_v3**n_w)/(1.+aT_v3**n_w))**(1./n_w)
    
    return (1.0-Om-Ogamma-Ov1-Ov2-Ov3 + Om*(1.0+z)**3 + Ogamma*(1.0+z)**4 + E_v1 + E_v2 + E_v3 )**0.5

def Hz(z,H0,Om, m_v1=0.0):
    h=H0/100
    return Ez_flat_LCDM(z,Om,h, m_v1)*H0

def dageH0_LCDM_postrebom(z,Om,h=0.7, m_v1=0.0):      # post recombination, so radiation is ignored
    return 1/Ez_flat_LCDM(z,Om,h, m_v1)/(1+z)

def ageH0_LCDM_postrebom(z,Om,h=0.7, m_v1=0.0):
    temp = integrate.quad(dageH0_LCDM_postrebom, 0, z, args=(Om,h, m_v1,))
    return temp[0]

def age_LCDM_postrebom(z,H0,Om, m_v1=0.0):
    h=H0/100
    return 978.5644/H0*ageH0_LCDM_postrebom(z,Om,h, m_v1)     # 978.5644 is 1/(km/s/Mpc) in Gyrs

def fC_LCDM(z,Om,h=0.7, m_v1=0.0):
    nll = lambda *args: 1/Ez_flat_LCDM(*args)
    temp = integrate.quad(nll, 0, z, args=(Om,h, m_v1))
    return temp[0]

def ft_LCDM(zs,zl,Om,h=0.7,m_v1=0.0):
    fM_s_temp = fC_LCDM(zs,Om,h, m_v1)
    fM_l_temp = fC_LCDM(zl,Om,h, m_v1)
    return fM_l_temp*fM_s_temp/(fM_s_temp-fM_l_temp)


# relation of H0 and Om in LCDM given the age:
def H0_Om_age_bound(H0,age_star,z_star,m_v1=0.0):
    h=H0/100
    Om_low = 0
    Om_high = 1
    while (Om_high-Om_low)>1E-4:
        Om_try = (Om_high+Om_low)/2
        if age_LCDM_postrebom(z_star,H0,Om_try,m_v1) > age_star:
            Om_low = Om_try
        else:
            Om_high = Om_try
    return Om_try



# Observations for late-time BAO
# AP= delta(z)/delta(\theta)
def AP(z,Om,h=0.7, m_v1=0.0):
    return Ez_flat_LCDM(z,Om,h, m_v1)*fC_LCDM(z,Om,h, m_v1)

def fV_LCDM(z,Om,h=0.7, m_v1=0.0):
    return (z/Ez_flat_LCDM(z,Om,h, m_v1)*(fC_LCDM(z,Om,h, m_v1))**2)**(1/3.0)




# For sound speed and sound horizon:
def BAO_sound_speed(z, omegabh2):
    return 1./( 3*(1+1/(1+z)*3*omegabh2/4/(2.472E-5) ) )**0.5     # sound speed, needs to make it general

def sound_horizon_integrand(z, Om, h, omegabh2_fit, m_v1=0.0):
    return BAO_sound_speed(z, omegabh2_fit)/Ez_flat_LCDM(z, Om, h, m_v1)
#return BAO_sound_speed(z)/np.sqrt(1-Om-4.15E-5/h**2+Om*(1+z)**3+4.15E-5/h**2*(1+z)**4)

def Delta_rH0(zstar, Deltaz, Om_fit, h_fit, omegabh2_fit, m_v1_fit=0.0):
    nll = lambda *args: sound_horizon_integrand(*args)
    temp = integrate.quad(nll, zstar-Deltaz, zstar, args=(Om_fit, h_fit, omegabh2_fit, m_v1_fit))
    return temp[0]

def rH0(zstar, Om_fit, h_fit, m_v1_fit=0.0):
    nll = lambda *args: sound_horizon_integrand(*args)
    temp = integrate.quad(nll, zstar, np.inf, args=(Om_fit, h_fit, omegabh2_fit, m_v1_fit))
    return temp[0]
