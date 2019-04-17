import os
from scipy.linalg import solve, inv
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.special import zeta
from constants import *
import time
import cython
cimport numpy as cnp
import numpy as np
#import statsmodels.api as sm
#from matrix_build import *
#import warnings
#warnings.filterwarnings("error", category=UserWarning)

path = os.getcwd()

cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)
    double log10(double)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.cdivision(True)
@cython.nonecheck(False)
class Universe(object):

    def __init__(self, k, omega_b, omega_cdm, omega_g, omega_L, accuracy=1e-3,
                 stepsize=0.01, lmax=5, testing=False, hubble_c=67.66, zreion=10,
                 m_nu=0., T_nu=0.71599):

        self.tcmb = 2.7255
        self.omega_b = omega_b
        self.omega_cdm = omega_cdm
        self.omega_g = omega_g
        self.omega_M = omega_cdm + omega_b
        self.omega_R = omega_g
        self.omega_L = 1. - self.omega_M - self.omega_R
        self.H_0 = hubble_c / 2.998e5 # units Mpc^-1
        self.little_h = hubble_c / 1e2
        self.zreion = zreion
        self.m_nu = m_nu
        self.T_nu = T_nu * self.tcmb * kboltz

        self.n_bary = self.omega_b * rho_critical  * (hubble_c / 100.)**2.

        self.Lmax = lmax
        self.stepsize = stepsize
        self.k = k

        self.accuracy = accuracy
        self.nu_q_bins = 100
        self.q_list = np.linspace(1e-3, 20, self.nu_q_bins)
#        self.nu_q_bins = Nlaguerre
#        self.q_list = q_i_Lag
        self.TotalVars = 5 + 2*(self.Lmax+1) + (self.Lmax+1)*self.nu_q_bins

        self.step = 0

        self.Theta_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        self.Theta_P_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        self.Neu_Dot = np.zeros(((self.Lmax+1)*self.nu_q_bins) ,dtype=object)

        self.combined_vector = np.zeros(self.TotalVars ,dtype=object)
        self.Psi_vec = []
        self.combined_vector[0] = self.Phi_vec = []
        self.combined_vector[1] = self.dot_rhoCDM_vec = []
        self.combined_vector[2] = self.dot_velCDM_vec = []
        self.combined_vector[3] = self.dot_rhoB_vec = []
        self.combined_vector[4] = self.dot_velB_vec = []
        for i in range(self.Lmax + 1):
            self.combined_vector[5+i*2] = self.Theta_Dot[i] = []
            self.combined_vector[6+i*2] = self.Theta_P_Dot[i] = []


        self.neu_indx = 5 + 2*(self.Lmax+1)
        for i in range((self.Lmax+1)*self.nu_q_bins):
            self.combined_vector[i+self.neu_indx] = self.Neu_Dot[i] = []

        self.compute_funcs()

        self.testing = testing
        if self.testing:
            self.aLIST = []
            self.etaLIST = []
            self.csLIST = []
            self.hubLIST = []
            self.dtauLIST = []
            self.xeLIST = []
            self.TbarLIST=[]

        return

    def compute_funcs(self, preload=False):
        cdef cnp.ndarray[double] a0_init
        # First get Hubble(a)
        if not os.path.isfile(path + '/precomputed/background.dat'):
            a0_init = np.logspace(-9, 0.1, 1e4)
            hublist = [self.hubble(av) for av in a0_init]
            self.HubEv = interp1d(np.log10(a0_init), np.log10(hublist), kind='cubic', bounds_error=False, fill_value=-100)
            eta_list = [self.conform_T(ai) for ai in a0_init]
            np.savetxt(path + '/precomputed/ln_a_CT_working.dat', np.column_stack((a0_init, eta_list, hublist)))
        else:
            load_lna = np.loadtxt(path + '/precomputed/ln_a_CT_working.dat')
            eta_list = load_lna[:, 1]
            a0_init = load_lna[:, 0]
            hublist = load_lna[:, 2]
            self.HubEv = interp1d(np.log10(a0_init), np.log10(hublist), kind='cubic', bounds_error=True)

        self.Thermal_sln()

        self.eta_0 = self.conform_T(1.)
        self.ct_to_scaleI = interp1d(np.log10(eta_list), np.log10(a0_init), kind='cubic',
                                    bounds_error=True)

        self.scale_to_ctI = interp1d(np.log10(a0_init), np.log10(eta_list), kind='cubic',
                                    bounds_error=True)

        fileVis = path + '/precomputed/working_VisibilityFunc.dat'
        if os.path.exists(fileVis):
            visfunc = np.loadtxt(fileVis)
            self.Vfunc = interp1d(visfunc[:,0], visfunc[:,1], kind='cubic', fill_value=0., bounds_error=False)
        return

    def Hub(self, a):
        return np.power(10., self.HubEv(np.log10(a)))

    def clearfiles(self):
#        if os.path.isfile(path + '/precomputed/background.dat'):
#            os.remove(path + '/precomputed/background.dat')
#
#        if os.path.isfile(path + '/precomputed/xe_working.dat'):
#            os.remove(path + '/precomputed/xe_working.dat')
#        if os.path.isfile(path + '/precomputed/tb_working.dat'):
#            os.remove(path + '/precomputed/tb_working.dat')
#
#        if os.path.isfile(path + '/precomputed/working_expOpticalDepth.dat'):
#            os.remove(path + '/precomputed/working_expOpticalDepth.dat')
#        if os.path.isfile(path + '/precomputed/working_VisibilityFunc.dat'):
#            os.remove(path + '/precomputed/working_VisibilityFunc.dat')
        return

    def Thermal_sln(self):

        self.tb_fileNme = path + '/precomputed/tb_working.dat'
        self.Xe_fileNme = path + '/precomputed/xe_working.dat'
        if not os.path.isfile(self.tb_fileNme) or not os.path.isfile(self.Xe_fileNme):
            lgz = 4.5
            ionizing_he = True
            fst = True
            lgz_list = [lgz]
            xe_he_list = [1.16381]
            fhe = 0.245 / (4. * (1. - 0.245))
            while ionizing_he:
                lgz -= 0.01
                sln = fsolve(lambda x: self.saha(x, lgz, first=fst, helium=True), 1. + 2.*fhe)[0]
                lgz_list.append(lgz)
                xe_he_list.append(sln)
                if sln <= (fhe + 1. + 1e-3):
                    fst = False
                if sln <= (1. + 1e-3):
                    ionizing_he = False

            he_xe_tab = np.column_stack((1. / (1. + 10.**np.asarray(lgz_list)), np.asarray(xe_he_list)))
            he_xe_tab = he_xe_tab[10.**np.asarray(lgz_list) > 3e3]

            tvalsHe = np.linspace(3e3, 1e3, 1000)
            val_sln_he = odeint(lambda x,y: self.thermal_funcs(x,y,hydro=False), [fhe - 1e-4, 2.7255 * (1. + tvalsHe[0])], tvalsHe)

            he2_tab = np.asarray([[1. / (1. + tvalsHe[i]), val_sln_he[i,0]] for i in range(len(val_sln_he)) if val_sln_he[i, 0] > 1e-6])

            tvals = np.linspace(3e3, 1e-2, 10000)
            y0 = [0.99999, 2.7255 * (1. + tvals[0])]
            val_sln = odeint(self.thermal_funcs, y0, tvals)
            avals = 1. / (1. + tvals)

            fhe = 0.16381 / 2.
            tanhV = (fhe + 1.) / 2. * (1. + np.tanh( 2.*((1.+self.zreion)**(3./2.) - (1.+ tvals)**1.5 ) / (3.*0.5*np.sqrt(1.+ tvals)) ))
            zreionHE = 3.5
            tanhV += fhe / 2. * (1. + np.tanh( 2.*((1.+zreionHE)**(3./2.) - (1.+ tvals)**1.5 ) / (3.*0.5*np.sqrt(1.+ tvals)) ))

            val_sln[:,0] += tanhV
            he2_tab_inc = 10.**interp1d(np.log10(he2_tab[:,0]), np.log10(he2_tab[:,1]), fill_value=-100., bounds_error=False, kind='cubic')(np.log10(avals))
            self.Xe_dark = np.vstack((he_xe_tab, np.column_stack((avals, val_sln[:,0] + he2_tab_inc))))

            tvals2 = np.linspace(1e4, 1e-2, 10000)
            termpB = odeint(self.recast_Tb, 2.7255 * (1. + tvals2[0]), tvals2,
                           args=(self.Xe_dark, ))

            self.Tb_drk = np.column_stack((1. / (1. + tvals2), termpB))

            np.savetxt(self.tb_fileNme, self.Tb_drk)
            np.savetxt(self.Xe_fileNme, self.Xe_dark)
        else:
            try:
                self.Tb_drk = np.loadtxt(self.tb_fileNme)
                self.Xe_dark = np.loadtxt(self.Xe_fileNme)

                tvals = 1. / self.Xe_dark[:,0] - 1.
                fhe = 0.16381 / 2.
                tanhV = (fhe + 1.) / 2. * (1. + np.tanh( 2.*((1.+self.zreion)**(3./2.) - (1.+ tvals)**1.5 ) / (3.*0.5*np.sqrt(1.+ tvals)) ))
                zreionHE = 3.5
                tanhV += fhe / 2. * (1. + np.tanh( 2.*((1.+zreionHE)**(3./2.) - (1.+ tvals)**1.5 ) / (3.*0.5*np.sqrt(1.+ tvals)) ))
                self.Xe_dark[:, 1] += tanhV

            except:
                print('fail to load xe and tb dark')
                raise ValueError

        self.Tb = interp1d(np.log10(self.Tb_drk[:,0]), np.log10(self.Tb_drk[:,1]), kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        self.Xe = interp1d(np.log10(self.Xe_dark[:,0]), np.log10(self.Xe_dark[:,1]), kind='linear',
                            bounds_error=False, fill_value='extrapolate')

        return

    def saha(self, xe, lgz, first=True, helium=True):
        tg = 2.7255 * (1. + 10.**lgz) * kboltz
        fhe = 0.16381 / 2.
        hh = 4.135e-15
        if first and helium:
            ep0 = 54.4
            lhs = (xe - 1. - fhe) * xe
            rhf = np.exp(- ep0 / tg) * (1. + 2.*fhe - xe)
        elif helium and not first:
            ep0 = 24.6
            lhs = (xe - 1.)*xe
            rhf = 4. * np.exp(- ep0 / tg) * (1. + fhe - xe)
        else:
            ep0 = 13.6
            lhs = xe**2.
            rhf = np.exp(- ep0 / tg) * (1. - xe)

        rhpre = (2. * np.pi * 5.11e5 * tg)**(3./2.) / (self.n_bary * (1. + 10.**lgz)**3. * hh**3.)
        units = (1. / 2.998e10)**3.
        return lhs - units * rhpre * rhf

    def thermal_funcs(self, val, z, hydro=True):
        xe, T = val
        if hydro:
            return [self.xeDiff([xe], z, T)[0], self.dotT_normal([T], z, xe)]
        else:
            return [self.xeDiff_he([xe], z, T)[0], self.dotT_normal([T], z, xe)]

    def recast_Tb(self, val, z, xe_list):
        xe_inter = interp1d(xe_list[:,0], np.log10(xe_list[:,1]), kind='linear',
                            bounds_error=False, fill_value=np.log10(1.16381))
        return self.dotT_normal(val, z, 10.**xe_inter(z))

    def dotT_normal(self, T, z, xe):
        # d log (Tb) / d z
        cdef double thompson_xsec = 6.65e-25 # cm^2
        cdef double aval = 1. / (1. + z)
        cdef double Yp = 0.245
        cdef double mol_wei = (1. + 4. * Yp) / (1. + 2. * Yp + xe) * 0.931

        cdef double n_b = self.n_bary * (1. + z)**3. * Yp
        cdef double hub = self.Hub(aval)
        cdef double omega_Rat = self.omega_g / self.omega_b

        return (2. * T[0] * aval  - (1./hub)*(8./3.)*(mol_wei/5.11e-4) *
                omega_Rat * xe * n_b * thompson_xsec * (2.7255*(1.+z) - T[0])*Mpc_to_cm)

    def dotT(self, T, lgz, xe, a):
        # d ln (T) / d ln (a)
        cdef double thompson_xsec = 6.65e-25 # cm^2
        cdef double aval = 1. / (1. + 10.**lgz)
        cdef double Yp = 0.245
        cdef double mol_wei = (1. + 4. * Yp) / (1. + 2. * Yp + xe) * 0.931
        cdef double n_b = self.n_bary * (1.+10.**lgz)**3.
        cdef double hub = self.Hub(aval)
        cdef double omega_Rat = self.omega_g / self.omega_b

        return (-2. + (1./ (hub * aval * T))*(8./3.)*(mol_wei/5.11e-4) *
                omega_Rat * xe * n_b * thompson_xsec * (2.7255*(1.+10.**lgz) - T)*Mpc_to_cm)

    def Cs_Sqr(self, a):
        cdef double kb = 8.617e-5/1e9 # GeV/K
        cdef double thompson_xsec = 6.65e-25 # cm^2
        cdef double facxe = 10.**self.Xe(log10(a))
        cdef double Yp = 0.245
        cdef double epsil = a * 1e-2
        cdef double mol_wei
        mol_wei = (1. + 4. * Yp) / (1. + 2. * Yp + facxe) * 0.931
        cdef double Tb
        if (1./a - 1.) < 1e4:
            Tb = 10.**self.Tb(log10(a))
            Tb2 = 10.**self.Tb(log10(a - epsil))
            tbderiv = log(Tb / Tb2) / log(a / (a - epsil))
        else:
            tbderiv = -2.
            Tb = 2.7225 / a
        return kb * Tb / mol_wei * (1. - 1./3. * tbderiv)

    def xeDiff(self, val, y, tgas):
        ep0 =  10.2343 # eV
        epG =  13.6 # eV
        kb = 8.617e-5 # ev/K
        me = 5.11e5 # eV
        aval = 1. / (1.+y)
        Yp = 0.245
        n_b = self.n_bary / aval**3.
        hub = self.Hub(aval)
        alphaH = 1.14 * 1e-19 * 4.309 * (tgas/1e4)**-0.6166 / (1. + 0.6703 * (tgas/1e4)**0.53) * 1e2**3. / 2.998e10 # cm^2

        beta = alphaH*(2.*np.pi*me*kb*tgas/4.135e-15**2.)**(3./2.) / 2.998e10**3. # 1/cm

        kh = (121.5682e-9)**3./(8.*np.pi*hub) * 1e6 # cm^3 * Mpc
        lambH = 8.22458 / 2.998e10 * Mpc_to_cm # 1 / Mpc
        preFac = (1. + kh *lambH * n_b * (1. - Yp) * (1 - val[0]))  # unitless
        preFac /= (1. + kh * (lambH+beta*Mpc_to_cm*np.exp(-(epG - ep0)/(kb*tgas))) * n_b * (1. - Yp) * (1 - val[0])) # Unitless

        return [preFac*aval/hub*(-(1.-val[0])*beta*np.exp(-epG/(kb*tgas)) + val[0]**2.*n_b*(1.-Yp)*alphaH)*Mpc_to_cm]

    def xeDiff_he(self, val, y, tgas):
        # tgas = 2.7225 * (1. + y)
        ep0 =  24.6 # eV
        nu_2s = 20.6 # 2.998e8 / 60.1404e-9 * 4.135e-15
        nu_2p = 2.998e8 / 58.4334e-9 * 4.135e-15
        nu_diff2 =  (nu_2p - ep0)
        kb = 8.617e-5 # ev/K
#        Mpc_to_cm = 3.086e24
        me = 5.11e5 # eV
        aval = 1. / (1.+y)
        Yp = 0.245
        fhe = Yp / (4. * (1. - Yp))
        n_b = self.n_bary / aval**3.
        hub = self.Hub(aval)
        alphaH = 10.**-16.744 / (np.sqrt(tgas / 3.)*(1.+tgas/3.)**(1.-0.711)*(1.+tgas/10.**5.114)**(1.+0.711)) * 1e6 / 2.998e10 # cm^2
        beta = alphaH*(2.*np.pi*me*kb*tgas/4.135e-15**2.)**(3./2.) / 2.998e10**3 * np.exp(-(ep0 - nu_2s) /(kb*tgas))  # 1/cm

        kh = (58.4334e-9)**3./(8.*np.pi*hub) * 1e6 # cm^3 * Mpc
        lambH = 51.3 / 2.998e10 * Mpc_to_cm # 1 / Mpc
        preFac = (1. + kh *lambH * n_b * (1-Yp) * (fhe - val[0]) * np.exp(- nu_diff2 / (kb*tgas)))  # unitless
        preFac /= (1. + kh * (lambH + beta * Mpc_to_cm) * n_b * (1-Yp) * (fhe - val[0]) * np.exp(- nu_diff2 / (kb*tgas))) # Unitless

        return [preFac*aval/hub*(val[0]*(1. + val[0])*n_b*(1-Yp)*alphaH  -  (fhe - val[0])*beta*np.exp(-nu_2s / (kb*tgas)))*Mpc_to_cm]


    def tau_functions(self):
        self.fileN_optdep = path + '/precomputed/working_expOpticalDepth.dat'
        self.fileN_visibil = path + '/precomputed/working_VisibilityFunc.dat'
        cdef double Yp, n_b, thompson_xsec, hubbs, xevals
        cdef cnp.ndarray[double] tau, etavals, vis
        cdef int i
        if not os.path.isfile(self.fileN_visibil) or not os.path.isfile(self.fileN_optdep):
            print('File not found... calculating...')
            avals = np.logspace(-4.5, 0, 10000)
            Yp = 0.245
            thompson_xsec = 6.65e-25 # cm^2
            tau = np.zeros_like(avals)

            dtau = np.asarray([-10.**self.Xe(log10(av))* (1. - Yp) * self.n_bary * thompson_xsec * Mpc_to_cm / av**2. for av in avals])
            etavals = 10.**self.scale_to_ctI(np.log10(avals))
            vis = np.zeros_like(avals)

            for i in range(len(avals) - 1):
                tau[i + 1] = np.trapz(-dtau[i:], etavals[i:])
                vis[i + 1] = -dtau[i + 1] * exp(-tau[i + 1])
            tau[0] = tau[1]
            np.savetxt(self.fileN_optdep, np.column_stack((avals, np.exp(-tau))))
            np.savetxt(self.fileN_visibil, np.column_stack((avals, vis)))

        return

    def init_conds(self, eta_0, aval):
        cdef double OM = self.omega_M
        cdef double ONu = self.rhoNeu_true(aval) / rho_critical / hbar**3. / (2.998e10)**3./ self.little_h**2. / 1e9

        cdef double rfactor = ONu / (ONu + self.omega_R*aval**-4.)
        cdef double HUB = self.Hub(aval)

        self.inital_perturb = -1./6.
        cdef double initP = self.inital_perturb
        cdef int i

        self.Psi_vec.append(initP)
        self.Phi_vec.append(-(1.+2.*rfactor/5.)*initP)
        self.dot_rhoCDM_vec.append(-3./2.*initP)
        self.dot_velCDM_vec.append(1./2.*eta_0*self.k*initP)
        self.dot_rhoB_vec.append(-3./2.*initP)
        self.dot_velB_vec.append(1./2*eta_0*self.k*initP)

        self.Theta_Dot[0].append(-1./2.*initP)
        self.Theta_Dot[1].append(1./6.*eta_0*self.k*initP)

        for i in range(self.Lmax + 1):
            if i > 1:
                self.Theta_Dot[i].append(0.)
            self.Theta_P_Dot[i].append(0.)

        dlnf_dlnq = self.dln_f0_dln_q(aval, self.q_list)
        ep_over_q = np.sqrt((self.m_nu * aval / self.T_nu)**2. + self.q_list**2.) / self.q_list
        for i in range(self.nu_q_bins):
            self.Neu_Dot[i].append(1./2. * initP * dlnf_dlnq[i])
            self.Neu_Dot[self.nu_q_bins + i].append(- eta_0 * self.k / 6. * initP * ep_over_q[i] * dlnf_dlnq[i])
            self.Neu_Dot[self.nu_q_bins*2 + i].append(-1./30. * (self.k*eta_0)**2. * initP * dlnf_dlnq[i])

        for i in range((self.Lmax + 1 - 3)*self.nu_q_bins):
            self.Neu_Dot[i + 3*self.nu_q_bins].append(0.)
        self.step = 0

        return

    def dln_f0_dln_q(self, a, q):
        erg = np.sqrt((a * self.m_nu / self.T_nu)**2. + q**2.)
        return -q**2. * np.exp(erg) / (1. + np.exp(erg)) / erg


    def solve_system(self, compute_TH):
        self.timeT = 0.
        cdef double eta_st = np.min([1e-3/self.k, 1e-1/0.7]) # Initial conformal time in Mpc

        cdef double y_st = log(self.ct_to_scale(eta_st))
        self.init_conds(eta_st, exp(y_st))
        self.eta_vector = [eta_st]
        self.y_vector = [y_st]

        cdef int try_count = 0
        cdef int try_max = 40
        FailRUN = False
        last_step_up = False

        cdef double y_use, eta_use, y_diff, test_epsilon, a_use
        cdef int i

        while (self.eta_vector[-1] < (self.eta_0 - 1.)):

            if try_count > try_max:
                print('FAIL TRY MAX....Breaking.')
                FailRUN=True
                return
            y_use = self.y_vector[-1] + self.stepsize
            eta_use = self.scale_to_ct(exp(y_use))

            if (eta_use > self.eta_0):
                eta_use = self.eta_0
                y_use = 1.
            self.eta_vector.append(eta_use)

            y_diff = y_use - self.y_vector[-1]
            if y_diff < 1e-10:
                print('ydiff is 0... failing...')
                return

            self.y_vector.append(y_use)
            a_use = exp(y_use)

            if self.step%5000 == 0:
                print('Last a: {:.7e}, New a: {:.7e}'.format(exp(self.y_vector[-2]), a_use))
            if ((y_diff > eta_use*a_use*self.Hub(a_use)) or
                (y_diff > a_use*self.Hub(a_use)/self.k)):
                self.stepsize *= 0.5
                self.eta_vector.pop()
                self.y_vector.pop()
                continue

            self.step_solver(y_use, eta_use, a_use)

            test_epsilon = abs(self.epsilon_test(a_use))
#            print(self.eta_vector[-1], np.exp(self.y_vector[-1]), self.y_vector[-1], self.stepsize, test_epsilon)

            if test_epsilon > self.accuracy:
                self.stepsize *= 0.5
                a_target = self.y_vector[-1] - 0.05
                if self.y_vector[0] >= a_target:
                    a_target += 0.02
                jchk = -1
                while (self.y_vector[jchk]  > a_target) and (len(self.y_vector) > 1):
                    self.eta_vector.pop()
                    self.y_vector.pop()
                    self.Psi_vec.pop()
                    for i in range(self.TotalVars):
                        self.combined_vector[i].pop()
                    jchk -= 1
                try_count += 1
                continue
            self.step += 1
            if (test_epsilon < 1e-3*self.accuracy) and not last_step_up:
                self.stepsize *= 1.0
                last_step_up = True
                #print 'Increase Step Size'
            else:
                last_step_up = False
            try_count = 0

        if not FailRUN and self.testing:
            print('Saving File...')
            self.save_system()

        if not compute_TH:
            return self.combined_vector[0][-1]

        sources = np.zeros((len(self.eta_vector), 4))
        cdef double aval, phi_term_back, psi_term_back, eta_back
        sources[:, 0] = self.eta_vector

        pi_polar = [self.combined_vector[6][i] + self.combined_vector[9][i] + self.combined_vector[10][i] for i in range(len(self.eta_vector))]

        sources[:, 2] = np.asarray(self.combined_vector[4])
        der2_pi = np.zeros_like(sources[:, 0])

        for i in range(len(self.eta_vector)):
            aval = self.ct_to_scale(sources[i, 0])
            if (i > 1) and (i < len(self.eta_vector) - 1):
                h2 = sources[i + 1, 0] - sources[i, 0]
                h1 = sources[i, 0] - sources[i - 1, 0]
                if (h1 != 0.) and (h2 != 0.):
                    vis1 = self.Vfunc(self.ct_to_scale(sources[i, 0]))
                    vis2 = self.Vfunc(self.ct_to_scale(sources[i + 1, 0]))
                    vis0 = self.Vfunc(self.ct_to_scale(sources[i - 1, 0]))
                    der2_pi[i] = 2.*(h2*(pi_polar[i + 1]*vis2) - (h1+h2)*(pi_polar[i]*vis1) +
                            h1*(pi_polar[i - 1]*vis0))/(h1*h2*(h1+h2))
                    der2_pi[i] *= 3. / (4. * self.k**2.)
            if i > 0:
                psi_term_back = self.Psi_vec[i - 1]
                phi_term_back = self.combined_vector[0][i - 1]
                eta_back = self.eta_vector[i - 1]
            else:
                psi_term_back = 0.
                phi_term_back = 0.
                eta_back = 0.

            sources[i, 3] = ((self.Psi_vec[i] - psi_term_back) - (self.combined_vector[0][i] - phi_term_back)) / \
                    (self.eta_vector[i] - eta_back)
        sources[:, 1] = [self.combined_vector[5][i] + self.Psi_vec[i] for i in range(len(self.eta_vector))]
#        sources[:, 1] = [self.combined_vector[5][i] + self.Psi_vec[i] + pi_polar[i] / 4. +
#                            der2_pi[i] for i in range(len(self.eta_vector))]

        source_interp = np.zeros(( len(global_a_list), 4 ))
        s_int_1 = interp1d(sources[:, 0], sources[:, 1], kind='cubic', fill_value=0., bounds_error=False)
        s_int_2 = interp1d(sources[:, 0], sources[:, 2], kind='cubic', fill_value=0., bounds_error=False)
        s_int_3 = interp1d(sources[:, 0], sources[:, 3], kind='cubic', fill_value=0., bounds_error=False)
        source_interp[:, 0] = self.scale_to_ct(global_a_list)
        source_interp[:,1] = s_int_1(source_interp[:, 0])
        source_interp[:,2] = s_int_2(source_interp[:, 0])
        source_interp[:,3] = s_int_3(source_interp[:, 0])

        return source_interp

    def step_solver(self, double lna, double eta, double aval):
        cdef double tau_n

        if self.step > 0:
            tau_n = (lna - self.y_vector[-2]) / (self.y_vector[-2] - self.y_vector[-3])
        else:
            tau_n = (lna - self.y_vector[-2]) / self.y_vector[-2]

        cdef double delt = (lna - self.y_vector[-2])
        cdef cnp.ndarray[double, ndim=2] Jmat = self.matrix_J(eta, aval)

        ysol = solve((1.+2.*tau_n)/(1.+tau_n)*np.eye(self.TotalVars+1) - delt*Jmat, self.b_vector(tau_n),
                     overwrite_a=True, overwrite_b=True)
#        ysol = np.matmul(inv((1.+2.*tau_n)/(1.+tau_n)*np.eye(self.TotalVars+1) - delt*Jmat), self.b_vector(tau_n))
        for i in range(self.TotalVars):
            self.combined_vector[i].append(ysol[i])
        self.Psi_vec.append(-12.*(aval**2./self.k**2.*(self.rhoG(aval)*self.combined_vector[11][-1])) - self.combined_vector[0][-1] + self.neu_N2_term(aval))
        return

    def b_vector(self, tau):
        cdef cnp.ndarray[double] bvec = np.zeros(self.TotalVars+1)
        bvec[-1] = (1.+tau) - tau**2./(1.+tau)
        cdef int i
        for i in range(self.TotalVars):
            if self.step == 0:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1]
            else:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1] - tau**2./(1.+tau)*self.combined_vector[i][-2]
        return bvec

    def matrix_J(self, double eta, double a_val):
        cdef double CsndB
        cdef cnp.ndarray[double, ndim=2] Jma = np.zeros((self.TotalVars+1, self.TotalVars+1))
        cdef double RR = (4.*self.rhoG(a_val))/(3.*self.rhoB(a_val))
        cdef double HUB = self.Hub(a_val)
        cdef double dTa = -10.**self.Xe(log10(a_val))*(1. - 0.245)*self.n_bary*6.65e-29*1e4/a_val**2./3.24078e-25
        if a_val > 1e-4:
            CsndB = self.Cs_Sqr(a_val)
        else:
            CsndB = self.Cs_Sqr(1e-4) * 1e-4 / a_val

        cdef double rG = self.rhoG(a_val)
        cdef double rB = self.rhoB(a_val)
        cdef double rC = self.rhoCDM(a_val)

        eta_matter_rad = self.scale_to_ct(5. * self.omega_R / self.omega_M)
        xc = max(eta_matter_rad * self.k, 1e3)
        self.gamma_supp = lambda x: 0.5 * (1. - np.tanh((x - xc) / 50.))
        cdef gammaSup = self.gamma_supp(self.k * eta)

        if self.testing:
            self.aLIST.append(a_val)
            self.etaLIST.append(eta)
            self.hubLIST.append(HUB)
            self.csLIST.append(CsndB)
            self.dtauLIST.append(dTa)
            self.xeLIST.append(10.**self.Xe(log10(a_val)))
            self.TbarLIST.append(10.**self.Tb(log10(a_val)))

        if np.abs(HUB * a_val / dTa) < 1e-2 and np.abs(self.k / dTa) < 1e-2:
            tflip_TCA = True
        else:
            tflip_TCA = False

        tflip_TCA = False

        cdef cnp.ndarray[double] PsiTerm = np.zeros(self.TotalVars+1)
        PsiTerm[0] += -1.
        PsiTerm[9] += -12.*(a_val/self.k)**2.*rG
        PsiTerm[-1] += self.neu_N2_term(a_val)

        # Phi Time derivative
        Jma[0,:] += PsiTerm
        Jma[0, 0] += -((self.k/(HUB*a_val))**2.)/3.
        Jma[0, 1] += 1./(HUB**2.*2.)*rC
        Jma[0, 3] += 1./(HUB**2.*2.)*rB
        Jma[0, 5] += 2./(HUB**2.)*rG
        Jma[0, -1] += -4. * np.pi * GravG / (3. * HUB**2.) * self.delta_rho_neu(a_val) / hbar**2. * (Mpc_to_cm/2.998e10)**2.

        # CDM density
        Jma[1,2] += -self.k/(HUB*a_val)
        Jma[1,:] += -3.*Jma[0,:]

        # CDM velocity
        Jma[2,2] += -1.
        Jma[2,:] += self.k/(HUB*a_val)*PsiTerm

        # Baryon density
        Jma[3,4] += -self.k / (HUB*a_val)
        Jma[3,:] += -3.*Jma[0,:]

        # Theta 0
        Jma[5,7] += -self.k / (HUB*a_val) * gammaSup
        Jma[5,:] += -Jma[0,:] * gammaSup

        # Baryon velocity
        if not tflip_TCA:
            Jma[4,4] += -1. + dTa * RR / (HUB*a_val)
            Jma[4,:] += self.k/(HUB*a_val)*PsiTerm
            Jma[4,3] += self.k * CsndB / (HUB * a_val)
            Jma[4,7] += -3.*dTa * RR / (HUB * a_val)
        else:
            Jma[4,4] += -1./(1.+RR) + 2.*(RR/(1.+RR))**2. + 2.*RR*HUB*a_val/\
                        ((1.+RR)**2.*dTa)
            Jma[4,3] += CsndB*self.k/(HUB*a_val*(1.+RR))
            Jma[4,5] += RR*self.k*(1./(HUB*a_val*(1+RR)) +
                        2./((1.+RR)**2.*dTa))
            Jma[4,9] += -RR*self.k/(2.*HUB*a_val*(1+RR))
            Jma[4,7] += -6.*(RR/(1.+RR))**2.
            Jma[4,:] += (self.k/(HUB*a_val) + RR*self.k / (dTa*(1.+RR)**2.))* PsiTerm
            Jma[4,:] += -(RR*self.k/(dTa*(1.+RR)**2.)) * CsndB*Jma[3,:]
            Jma[4,:] += (RR*self.k/(dTa*(1.+RR)**2.)) * Jma[5,:]

        # ThetaP 0
        Jma[6,6] += dTa / (2.*HUB*a_val) * gammaSup
        Jma[6,9] += - dTa / (2.*HUB*a_val) * gammaSup
        Jma[6,10] += - dTa / (2.*HUB*a_val) * gammaSup
        Jma[6,8] += - self.k / (HUB*a_val) * gammaSup

        # Theta 1
        if not tflip_TCA:
            Jma[7,9] += -2.*self.k / (3.*HUB*a_val) * gammaSup
            Jma[7,7] += dTa / (HUB*a_val) * gammaSup
            Jma[7,4] += -dTa / (3.*HUB*a_val) * gammaSup
            Jma[7,5] += self.k / (3.*HUB*a_val) * gammaSup
            Jma[7,:] += self.k*PsiTerm / (3.*HUB*a_val) * gammaSup
        else:
            Jma[7,4] += -1./(3.*RR) * gammaSup
            Jma[7,3] += CsndB*self.k/(HUB*a_val*RR*3.) * gammaSup
            Jma[7,5] += self.k/(3.*HUB*a_val) * gammaSup
            Jma[7,9] += -self.k/(6.*HUB*a_val) * gammaSup
            Jma[7,:] += (1.+RR)*self.k/(3.*RR*HUB*a_val)*PsiTerm * gammaSup
            Jma[7,:] += -Jma[4,:]/(3.*RR) * gammaSup

        # ThetaP 1
        Jma[8,6] += self.k / (3.*HUB*a_val) * gammaSup
        Jma[8,10] += -2.*self.k / (3.*HUB*a_val) * gammaSup
        Jma[8,8] += dTa / (HUB*a_val) * gammaSup

        cdef int i, elV, inx

        # Theta 2
        Jma[9,7] += 2.*self.k / (5.*HUB*a_val) * gammaSup
        Jma[9,11] += -3.*self.k / (5.*HUB*a_val) * gammaSup
        Jma[9,9] += 9.*dTa / (10.*HUB*a_val) * gammaSup
        Jma[9,6] += -dTa / (10.*HUB*a_val) * gammaSup
        Jma[9,10] += -dTa /(10.*HUB*a_val) * gammaSup

        # ThetaP 2
        Jma[10,8] += 2.*self.k / (5.*HUB*a_val) * gammaSup
        Jma[10,12] += -3.*self.k / (5.*HUB*a_val) * gammaSup
        Jma[10,10] += 9.*dTa / (10.*HUB*a_val) * gammaSup
        Jma[10,8] += -dTa / (10.*HUB*a_val) * gammaSup
        Jma[10,6] += -dTa / (10.*HUB*a_val) * gammaSup

        for i in range(11, 11 + self.Lmax - 3):
            elV = i - 11 + 3
            inx = i - 11
            # Photons
            Jma[11+2*inx,11+2*inx] += dTa / (HUB*a_val) * gammaSup
            Jma[11+2*inx,11+2*inx-2] += self.k*elV/((2.*elV + 1.)*(HUB*a_val)) * gammaSup
            Jma[11+2*inx,11+2*inx+2] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val)) * gammaSup

            # Polarization
            Jma[11+2*inx+1,11+2*inx+1] += dTa / (HUB*a_val) * gammaSup
            Jma[11+2*inx+1,11+2*inx+1-2] += self.k*elV/((2.*elV + 1.)*(HUB*a_val)) * gammaSup
            Jma[11+2*inx+1,11+2*inx+1+2] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val)) * gammaSup

        # Theta Lmax
        Jma[self.neu_indx-2, self.neu_indx-4] += self.k / (HUB*a_val) * gammaSup
        Jma[self.neu_indx-2, self.neu_indx-2] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val) * gammaSup


        # ThetaP Lmax
        Jma[self.neu_indx-1, self.neu_indx-3] += self.k / (HUB*a_val) * gammaSup
        Jma[self.neu_indx-1, self.neu_indx-1] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val) * gammaSup

        ### Neutrinos
        erg_list = np.sqrt((a_val * self.m_nu / self.T_nu)**2. + self.q_list**2.)
        df_term = self.dln_f0_dln_q(a_val, self.q_list)
        for i in range(self.nu_q_bins):
            # Neu 0
            q_o_e = self.q_list[i] / erg_list[i]
            Jma[self.neu_indx+i,self.neu_indx+i+self.nu_q_bins] += -self.k / (HUB*a_val) * q_o_e
            Jma[self.neu_indx+i,:] += Jma[0,:] * df_term[i]

            # Neu 1
            Jma[self.neu_indx+i+self.nu_q_bins, :] += -self.k * PsiTerm / (3.*HUB*a_val) * df_term[i] / q_o_e
            Jma[self.neu_indx+i+self.nu_q_bins, self.neu_indx+i] += self.k / (3.*HUB*a_val) * q_o_e
            Jma[self.neu_indx+i+self.nu_q_bins, self.neu_indx+i+self.nu_q_bins*2] += -2.*self.k/ (3.*HUB*a_val) * q_o_e

            # Neu 2
            Jma[self.neu_indx+i+self.nu_q_bins*2,self.neu_indx+i+self.nu_q_bins] += 2.*self.k/ (5.*HUB*a_val) * q_o_e
            Jma[self.neu_indx+i+self.nu_q_bins*2,self.neu_indx+i+self.nu_q_bins*3] += -3.*self.k/ (5.*HUB*a_val) * q_o_e

            for j in range(self.Lmax - 3):
                # Neutrinos
                Jma[self.neu_indx+i+self.nu_q_bins*(j+3),self.neu_indx+i+self.nu_q_bins*(j+2)] += self.k*elV/((2.*elV + 1.)*(HUB*a_val)) * q_o_e
                Jma[self.neu_indx+i+self.nu_q_bins*(j+3),self.neu_indx+i+self.nu_q_bins*(j+4)] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val)) * q_o_e

            # Nu Lmax
            Jma[-2, -2 - self.nu_q_bins] += self.k / (HUB*a_val) * q_o_e
            Jma[-2, -2] += -(self.Lmax+1.)/(eta*HUB*a_val)

        return Jma

    def scale_to_ct(self, scale):
        return 10.**self.scale_to_ctI(np.log10(scale))

    def ct_to_scale(self, ct):
        return 10.**self.ct_to_scaleI(np.log10(ct))

    def scale_a(self, eta):
        return self.ct_to_scale(eta)

    def conform_T(self, a):
        return quad(lambda x: 1 / x**2. / self.hubble(x), 0., a, epsabs=1e-5, limit=30)[0]

    def hubble(self, a):
        rho_nu = self.rhoNeu_true(a) / rho_critical / hbar**3. / (2.998e10)**3./ self.little_h**2. / 1e9
        return self.H_0 * np.sqrt(self.omega_R*a**-4.+self.omega_M*a**-3.+self.omega_L  + rho_nu)

    def rhoCDM(self, a):
        return self.omega_cdm * self.H_0**2. * a**-3.

    def rhoB(self, a):
        return self.omega_b * self.H_0**2. * a**-3.

    def rhoG(self, a):
        return self.omega_g * self.H_0**2. * a**-4.

    def rhoNeu_true(self, a):
        val = np.sum(w_i_Lag * np.exp(q_i_Lag) * q_i_Lag**2. * np.sqrt((self.m_nu*a/self.T_nu)**2. + q_i_Lag**2.) / \
            (1. + np.exp(np.sqrt((self.m_nu*a/self.T_nu)**2. + q_i_Lag**2.))) ) * self.T_nu**4.
#        val = 7*np.pi**4 * self.T_nu**4. / 120.
        units =  2. / (2.*np.pi)**3.
        return 3.045 * val * 4. * np.pi * units / a**4. # units ev ^ 4

    def delta_rho_neu(self, a):
        # \delta T_00
        integrnd = [self.Neu_Dot[i][-1] * self.q_list[i]**2. * np.sqrt((self.m_nu * a / self.T_nu)**2.+self.q_list[i]**2.) / \
                    (1. + np.exp(np.sqrt((self.m_nu * a / self.T_nu)**2.+self.q_list[i]**2.) )) for i in range(self.nu_q_bins)]

        interpF = interp1d(self.q_list, integrnd, kind='cubic', bounds_error=False, fill_value = 0.)
        val = np.sum(w_i_Lag * np.exp(q_i_Lag) * interpF(q_i_Lag))
#        val = np.sum(w_i_Lag * np.exp(q_i_Lag) * integrnd)
        return -3.045 * val * 4. * np.pi * 2. / (2.*np.pi)**3.  / a**4. * self.T_nu**4.

    def neu_vel_term(self, a):
        # (\rho_nu + P_nu) * \theta , up to normalization
        integrnd = [self.Neu_Dot[self.nu_q_bins + i][-1] * self.q_list[i]**3. / \
                        (1. + np.exp(np.sqrt((self.m_nu*a/self.T_nu)**2.+self.q_list[i]**2.) )) for i in range(self.nu_q_bins)]
        interpF = interp1d(self.q_list, integrnd, kind='cubic', bounds_error=False, fill_value = 0.)
        val = np.sum(w_i_Lag * np.exp(q_i_Lag) * interpF(q_i_Lag))
#        val = np.sum(w_i_Lag * np.exp(q_i_Lag) * integrnd)
        return val * 4. * np.pi * self.k  * 2. / (2.*np.pi)**3.  / a**4. * 3.045 * self.T_nu**4.

    def neu_N2_term(self, a):
        # (\rho_nu + P_nu) * N2 , up to a normalization
        integrnd = [self.Neu_Dot[self.nu_q_bins*2 + i][-1] * self.q_list[i]**4. /
                    np.sqrt((self.m_nu*a/self.T_nu)**2. + self.q_list[i]**2.) /
                    (1. + np.exp(np.sqrt((self.m_nu*a/self.T_nu)**2.+self.q_list[i]**2.))) for i in range(self.nu_q_bins)]
        interpF = interp1d(self.q_list, integrnd, kind='cubic', bounds_error=False, fill_value = 0.)
        val = np.sum(w_i_Lag * np.exp(q_i_Lag) * interpF(q_i_Lag))
#        val = np.sum(w_i_Lag * np.exp(q_i_Lag) * integrnd)
        return (-8. * GravG * val * 3.045 / self.k**2 / np.pi * (Mpc_to_cm / hbar / 2.998e10)**2. / a**2. *self.T_nu**4.)

    def epsilon_test(self, a):
        cdef double phiTerm = 2. * self.k**2. / (3. * a) * self.combined_vector[0][-1]
        cdef double T00_1 = -a * self.H_0**2. * ( (self.omega_cdm*self.combined_vector[1][-1]+self.omega_b*self.combined_vector[3][-1])*a**-3. +\
                            4.*(self.omega_g*self.combined_vector[5][-1])*a**-4.)
        cdef double T00_nu = a * 8. * np.pi * GravG / 3. * self.delta_rho_neu(a) * (Mpc_to_cm / (2.998e10 * hbar))**2.
        cdef double HUB = self.Hub(a)
        cdef double theta0_1 = -3. * HUB * a**2. / self.k * self.H_0**2. * ((self.omega_cdm*self.combined_vector[2][-1]+
                            self.omega_b*self.combined_vector[4][-1])*a**-3. + 4.*(self.omega_g*self.combined_vector[7][-1]/a**4.))
        cdef double theta0_nu = -HUB * a**2. * 8.*np.pi*GravG / self.k**2. * self.neu_vel_term(a) * (Mpc_to_cm / (2.998e10 * hbar))**2.
        return (phiTerm + T00_1 + T00_nu + theta0_1 + theta0_nu) / HUB**2.


    def save_system(self):

        if self.testing:
            psi_term = np.zeros(len(self.eta_vector))
            for i in range(len(self.eta_vector)):
                aval = 10.**self.ct_to_scale(np.log10(self.eta_vector[i]))

            sve_tab = np.zeros((len(self.eta_vector), self.TotalVars+2))
            sve_tab[:,0] = self.eta_vector
            sve_tab[:,-1] = self.Psi_vec
            for i in range(self.TotalVars):
                sve_tab[:,i+1] = self.combined_vector[i]
            np.savetxt(path + '/OutputFiles/StandardUniverse_FieldEvolution_{:.4e}.dat'.format(self.k), sve_tab, fmt='%.8e', delimiter='    ')
            np.savetxt(path+'/OutputFiles/StandardUniverse_Background.dat',
                        np.column_stack((self.aLIST, self.etaLIST, self.xeLIST, self.hubLIST, self.csLIST, self.dtauLIST, self.TbarLIST)))
        return


def interp1(double[:] x, double[:] y, double x0):
    cdef int i = 0
    cdef double res

    if x0 > x[-1]:
        return y[-1]
    if x0 < x[0]:
        return y[0]

    while (x[i] < x0) and (i <= len(x)):
        i = i + 1

    if x[i] == x0:
        return y[i]
    if i == 0:
        return y[i] - (y[i+1] - y[i]) * (x0 - x[i]) / (x[i+1] - x[i])

    res = y[i-1] + (y[i] - y[i-1]) * (x0 - x[i-1]) / (x[i] - x[i-1])
    return res
