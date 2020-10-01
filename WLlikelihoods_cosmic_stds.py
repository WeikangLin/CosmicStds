# python 3.6
# author Weikang Lin

#### Define my own Likelihood functions
#### call with WL_PS(path to Pk)


import numpy as np
from scipy import integrate
import WLbackground_cosmic_stds as BG
import sys, os


# constant for fast algorithm of fM

# one-term coefficients:
a = 1.
alpha = 0.5

# two-term coefficients:
b1 = 0.5
b2 = 0.5
beta1 = (3.+np.sqrt(3.))/6.
beta2 = (3.-np.sqrt(3.))/6.

# three-term coefficients:

c1 = 4./9.
c2 = 5./18.
c3 = 5./18.
gamma1 = 0.5
gamma2 = (5.+np.sqrt(15.))/10.
gamma3 = (5.-np.sqrt(15.))/10.




# prior on Om and H0:
def lnprior_SN(theta): # keep the order of theta to be C, Om, h, and m_v1
    if len(theta)==2:
        if 17<theta[0]<30 and 0.001<theta[1]<0.99:
            return 0.0
    if len(theta)==3:
        if 17<theta[0]<30 and 0.001<theta[1]<0.99 and 0.3<theta[2]<1.1:
            return 0.0
    if len(theta)==4:
        if 17<theta[0]<30 and 0.001<theta[1]<0.99 and 0.3<theta[2]<1.1 and 0<theta[3]<0.1:
            return 0.0
    return -np.inf



# SN pantheon likelihood:
def lnlike_SN(theta, PAN,invC): # to be modify
    if len(theta)==2:
        C_fit, Om_fit = theta
        h_fit = 0.7
        m_v1_fit=0.0
    if len(theta)==3:
        C_fit, Om_fit, h_fit = theta
        m_v1_fit=0.0
    if len(theta)==4:
        C_fit, Om_fit, h_fit, m_v1_fit = theta
    fMs = []
    for z_ob in PAN.zcmb:
        temp = BG.fC_LCDM(z_ob,Om_fit,h_fit,m_v1_fit)
        fMs.append(temp)
    m_pre = 5*np.log10((1+PAN.zcmb)*fMs)+C_fit
    return -0.5*np.dot((m_pre-PAN.mb),np.dot(invC,(m_pre-PAN.mb)))


def lnlike_SN_fast(theta, PAN, invC):  #data need to be sorted in a z ascending order
    if len(theta)==2:   # need to make parameter selction thing smarter
        C_fit, Om_fit = theta
        h_fit = 0.7
        m_v1_fit=0.0
    if len(theta)==3:
        C_fit, Om_fit, h_fit = theta
        m_v1_fit=0.0
    if len(theta)==4:
        C_fit, Om_fit, h_fit, m_v1_fit = theta
    fMs = []
    nll = lambda *args: 1/BG.Ez_flat_LCDM(*args)
    FirstZ = True
    for z_ob in PAN.zcmb:
        if FirstZ:
            temp = BG.fC_LCDM(z_ob,Om_fit,h_fit,m_v1_fit)
            fMs.append(temp)
            FirstZ = False
        else:
            Deltaz = z_ob-z_previous
            if Deltaz>0.1:
                temp = BG.fC_LCDM(z_ob,Om_fit,h_fit,m_v1_fit)
            else:
                # one term, O^3h
                #temp1 = 1/BG.Ez_flat_LCDM(z_previous+Deltaz*0.5,Om_fit,h_fit,m_v1_fit)*Deltaz
                
                # two terms, O^5h
                temp1 = b1/BG.Ez_flat_LCDM(z_previous+Deltaz*beta1,Om_fit,h_fit,m_v1_fit)*Deltaz
                temp2 = b2/BG.Ez_flat_LCDM(z_previous+Deltaz*beta2,Om_fit,h_fit,m_v1_fit)*Deltaz
                
                # three terms, O^7h
                #temp1 = c1/BG.Ez_flat_LCDM(z_previous+Deltaz*gamma1,Om_fith_fit,m_v1_fit)*Deltaz
                #temp2 = c2/BG.Ez_flat_LCDM(z_previous+Deltaz*gamma2,Om_fit,h_fit,m_v1_fit)*Deltaz
                #temp3 = c3/BG.Ez_flat_LCDM(z_previous+Deltaz*gamma3,Om_fit,h_fit,m_v1_fit)*Deltaz
                
                temp = fM_previous + temp1 + temp2 #+ temp3
            fMs.append(temp)
        z_previous = z_ob
        fM_previous = temp
    m_pre = 5*np.log10((1+PAN.zcmb)*fMs)+C_fit
    return -0.5*np.dot((m_pre-PAN.mb),np.dot(invC,(m_pre-PAN.mb)))


def lnprob_SN(theta, PAN, invC, use_fast):
    lp = lnprior_SN(theta)
    if not np.isfinite(lp):
        return -np.inf
    if use_fast:
        return lp + lnlike_SN_fast(theta, PAN, invC)
    else:
        return lp + lnlike_SN(theta, PAN, invC)


## BAO likelihoods:
def lnlike_BAO_fV(theta, zeff, obs):
    if len(theta)==2:
        rdH0_fit, Om_fit = theta
        h_fit = 0.7
        m_v1_fit=0.0
    if len(theta)==3:
        rdH0_fit, Om_fit, h_fit = theta
        m_v1_fit=0.0
    if len(theta)==4:
        rdH0_fit, Om_fit, h_fit, m_v1_fit = theta
    predict = rdH0_fit/BG.fV_LCDM(zeff, Om_fit, h_fit, m_v1_fit)
    return -0.5*(predict-obs[0])**2/obs[1]**2

def lnlike_BAO_theta_dz(theta, zeff, obs, invC_BAO):
    if len(theta)==2:
        rdH0_fit, Om_fit = theta
        h_fit = 0.7
        m_v1_fit=0.0
    if len(theta)==3:
        rdH0_fit, Om_fit, h_fit = theta
        m_v1_fit=0.0
    if len(theta)==4:
        rdH0_fit, Om_fit, h_fit, m_v1_fit = theta
    predict = []
    if zeff.size==1:
        predict.append(rdH0_fit/BG.fC_LCDM(zeff,Om_fit,h_fit,m_v1_fit))
        predict.append(rdH0_fit*BG.Ez_flat_LCDM(zeff,Om_fit,h_fit,m_v1_fit))
    else:
        for z in zeff:
            predict.append(rdH0_fit/BG.fC_LCDM(z,Om_fit,h_fit,m_v1_fit))
            predict.append(rdH0_fit*BG.Ez_flat_LCDM(z,Om_fit,h_fit,m_v1_fit))
    predict=np.array(predict)
    return -0.5*np.dot((predict-obs),np.dot(invC_BAO,(predict-obs)))



def lnlike_BAO(theta, BAO_data_path, BAO_datasets):
    if len(theta)==2:
        rdH0_fit, Om_fit = theta
        h_fit = 0.7
        m_v1_fit=0.0
    if len(theta)==3:
        rdH0_fit, Om_fit, h_fit = theta
        m_v1_fit=0.0
    if len(theta)==4:
        rdH0_fit, Om_fit, h_fit, m_v1_fit = theta
    BAO_like = 0
    for BAO_dataset in BAO_datasets:
        zeff = np.loadtxt(BAO_data_path+BAO_dataset+'_zeff.txt')
        obs = np.loadtxt(BAO_data_path+BAO_dataset+'_measurement.txt')
        if os.path.isfile(BAO_data_path+BAO_dataset+'_C.txt'):
            Cov = np.loadtxt(BAO_data_path+BAO_dataset+'_C.txt')
            invC_BAO = np.linalg.inv(Cov)
            temp = lnlike_BAO_theta_dz([rdH0_fit,Om_fit,h_fit,m_v1_fit],zeff,obs,invC_BAO)
        else:
            temp = lnlike_BAO_fV([rdH0_fit,Om_fit,h_fit,m_v1_fit],zeff,obs)
        BAO_like += temp
    return BAO_like


def lnprior_BAO(theta): # keep the order of theta to be rdH0, Om, h, and m_v1
    if len(theta)==2:
        if 0.01<theta[0]<0.1 and 0.001<theta[1]<0.99:
            return 0.0
    if len(theta)==3:
        if 0.01<theta[0]<0.1 and 0.001<theta[1]<0.99 and 0.3<theta[2]<1.1:
            return 0.0
    if len(theta)==4:
        if 0.01<theta[0]<0.1 and 0.001<theta[1]<0.99 and 0.3<theta[2]<1.1 and 0<theta[3]<0.1:
            return 0.0
    return -np.inf


def lnprob_BAO(theta, BAO_data_path, BAO_datasets):
    lp = lnprior_BAO(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_BAO(theta, BAO_data_path, BAO_datasets)





# CMB theta likelihood:

def lnprior_CMB(theta): # keep the order of theta to be zstar rdH0, Om, h, and m_v1
    if len(theta)==3:
        if 1070<theta[0]<1110 and 0.01<theta[1]<0.1 and 0.01<theta[2]<0.99:
            return 0.0
    if len(theta)==4:
        if 1070<theta[0]<1110 and 0.01<theta[1]<0.1 and 0.01<theta[2]<0.99 and 0.01<theta[3]<2.0:
            return 0.0
    if len(theta)==5:
        if 1070<theta[0]<1110 and 0.01<theta[1]<0.1 and 0.01<theta[2]<0.99 and 0.1<theta[3]<2.0 and 0<theta[3]<0.1:
            return 0.0
    return -np.inf


def lnlike_CMB(theta, CMB_means, CMB_invC):
    if len(theta)==3:
        zstar_fit, rsH0_fit, Om_fit = theta
        h_fit = 0.7
        m_v1_fit=0.0
    if len(theta)==4:
        zstar_fit, rsH0_fit, Om_fit, h_fit = theta
        m_v1_fit=0.0
    if len(theta)==5:
        zstar_fit, rsH0_fit, Om_fit, h_fit, m_v1_fit = theta
    theta_pre = rsH0_fit/BG.fC_LCDM(zstar_fit,Om_fit,h_fit,m_v1_fit)
    if CMB_means.size == 2:
        CMB_pre = np.array([theta_pre,zstar_fit])
    else:
        Rshift_pre = Om_fit**0.5*BG.fC_LCDM(zstar_fit,Om_fit,h_fit,m_v1_fit)
        CMB_pre = np.array([theta_pre,zstar_fit,Rshift_pre])
    return -0.5*np.dot((CMB_pre-CMB_means),np.dot(CMB_invC,(CMB_pre-CMB_means)))


def lnprob_CMB(theta, CMB_means, CMB_invC):
    lp = lnprior_CMB(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_CMB(theta, CMB_means, CMB_invC)
