import os
from boltzmann_c import *
from scipy.integrate import quad
from scipy.special import spherical_jn
import itertools
import cython
cimport numpy as cnp
import numpy as np
from constants import *
#from multiprocessing import *
#from pathos.multiprocessing import ProcessingPool as Pool
#from functools import partial
import math

path = os.getcwd()

cdef extern from "math.h":
    #float INFINITY
    double exp(double)
    double sqrt(double)
    double log(double)
    double log10(double)

@cython.boundscheck(False) # turn off bounds-checking for entire function
class CMB(object):

    def __init__(self, OM_b, OM_c, OM_g, OM_L, kmin=5e-3, kmax=0.5, knum=200,
                 lmax=2500, lvals=250, Ftag='StandardUniverse', lmax_Pert=5,
                 HubbleParam=67.77, n_s_index=0.9619, T_nu=0.71599, mass_nu=0.,
                 A_s_norm=3.044, z_reion=10., Neff=3.045, killF=False):

        self.HubbleParam = HubbleParam
        self.n_s_index = n_s_index
        self.A_s = A_s_norm
        self.Neff = Neff
        self.z_reion = z_reion

        self.OM_b = OM_b
        self.OM_c = OM_c
        self.OM_M = OM_b + OM_c
        self.OM_g = OM_g
        self.OM_L = OM_L
        self.T_nu = T_nu
        self.mass_nu = mass_nu


        self.kmin = kmin
        self.kmax = kmax
        self.knum = knum
        self.k_remove = []
        self.Ftag = Ftag

        self.cparam_tag = '_Ob_{:.4e}_Oc_{:.4e}_H0_{:.4e}_Neff_{:.4e}_Ns_{:.4e}_As_{:.4e}_zreion_{:.2f}_'.format(OM_b,
                        OM_c, HubbleParam, Neff, n_s_index, A_s_norm, z_reion)
        self.cparam_tag += 'mNus_{:.3e}_Tnus_{:.3e}_'.format(mass_nu, T_nu)

        self.lmax = lmax
        self.lvals = lvals

        self.lmax_Pert = lmax_Pert
        self.lmin = 10

        self.init_pert = -1/6.

        ell_val = list(range(self.lmin, self.lmax, 20))
        indxT = len(ell_val)
        for i in list(range(indxT)):
            if (i%2 == 1) and (ell_val[indxT - i - 1] > 300):
                ell_val.pop(indxT - i - 1)

        if killF:
            self.clearfiles()
            self.loadfiles(tau=True)
        else:
            self.loadfiles(tau=False)

        self.ThetaTabTot = np.zeros((self.knum+1, len(ell_val)))
        self.ThetaTabTot[0,:] = ell_val
        return


    def runall(self, kVAL, kindx, compute_LP=False, compute_TH=False,
               compute_CMB=False, compute_MPS=False, kgrid=None, ThetaTab=None):

        LP_fail = False
        cdef cnp.ndarray[double] ret_arr, theta_arr

        if kVAL is None:
            if compute_MPS:
                self.kgrid = np.logspace(log10(self.kmin), log10(self.kmax), self.knum)
            else:
                self.kgrid = np.logspace(log10(self.kmin), log10(self.kmax), self.knum)

        if compute_LP:
            sources = self.kspace_linear_pert(kVAL, compute_CMB)
            if sources is None:
                LP_fail = True
                print('FAIL PERT: ', kVAL)
                return None
            if not compute_TH:
                return sources

        if compute_TH and not LP_fail:
            print('Computing Theta Files...\n')
            if not compute_LP:
                print('Must compute LP....')

            theta_arr = self.theta_integration(kVAL, sources)
            return theta_arr


        if compute_CMB:
            print('Computing CMB...\n')
            self.computeCMB(kgrid, ThetaTab)
        if compute_MPS:
            print('Computing Matter Power Spectrum...\n')
            self.MatterPower(ThetaTab)
        return

    def clearfiles(self):
#        if os.path.isfile(path + '/precomputed/ln_a_CT_working.dat'):
#            os.remove(path + '/precomputed/ln_a_CT_working.dat')
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

    def loadfiles(self, tau=False):

        SingleUni = Universe(1., self.OM_b, self.OM_c, self.OM_g, self.OM_L,
                             hubble_c=self.HubbleParam, zreion=self.z_reion,
                             m_nu=self.mass_nu, T_nu=self.T_nu)
        self.ct_to_scale = lambda x: SingleUni.ct_to_scale(x)
        self.scale_to_ct = lambda x: SingleUni.scale_to_ct(x)
        if tau:
            SingleUni.tau_functions()
        self.eta0 = SingleUni.eta_0
        self.H_0 = SingleUni.H_0
        self.OM_nu = SingleUni.rhoNeu_true

        opt_depthL = np.loadtxt(path + '/precomputed/working_expOpticalDepth.dat')
        self.opt_depth = interp1d(opt_depthL[:,0], opt_depthL[:,1], kind='cubic', bounds_error=False, fill_value=0.)
        visfunc = np.loadtxt(path + '/precomputed/working_VisibilityFunc.dat')
        self.Vfunc = interp1d(visfunc[:,0], visfunc[:,1], kind='cubic', fill_value=0.)
        self.eta_start = self.scale_to_ct(np.min(visfunc[:,0]))
        return

    def load_bessel(self):
        self.sphB = np.zeros(len(self.ThetaTabTot[0,:]), dtype=object)
        self.sphB_D = np.zeros(len(self.ThetaTabTot[0,:]), dtype=object)
        xlist = np.linspace(0, self.kmax * self.eta0, 4000)
        for i in range(len(self.ThetaTabTot[0,:])):
            self.sphB[i] = interp1d(xlist, spherical_jn(int(self.ThetaTabTot[0, i]), xlist), kind='cubic',
                                    fill_value=0., bounds_error=False)
            self.sphB_D[i] = interp1d(xlist, spherical_jn(int(self.ThetaTabTot[0, i]), xlist, derivative=True),
                                      kind='cubic', fill_value=0., bounds_error=False)
        return

    def kspace_linear_pert(self, kVAL, compute_TH):
        try:
            chk = len(kVAL)
            kgrid = kVal
        except TypeError:
            kgrid = [kVAL]

        for k in kgrid:
            stepsize = 1e-2

            SingleUni = Universe(k, self.OM_b, self.OM_c, self.OM_g, self.OM_L,
                                stepsize=stepsize, accuracy=1e-3, lmax=self.lmax_Pert,
                                hubble_c=self.HubbleParam, zreion=self.z_reion,
                                m_nu=self.mass_nu, T_nu=self.T_nu)
            soln = SingleUni.solve_system(compute_TH)
        return soln


    def theta_construction(self, sources, kgrid_i):
        self.load_bessel()
        cdef int i, j, z
        cdef double ell, kk, eta_v, vis, expD


        cdef cnp.ndarray[double] kgrid_new = np.logspace(log10(self.kmin), log10(self.kmax), self.knum)
        cdef cnp.ndarray[double] tau_list = sources[0, :, 0]
        if tau_list[-1] == self.eta0:
            tau_list[-1] = self.eta0 - 1.
        cdef cnp.ndarray[double] integ1 = np.zeros_like(tau_list)
        sounce_interps_SW = np.zeros(len(tau_list), dtype=object)
        sounce_interps_D = np.zeros(len(tau_list), dtype=object)
        sounce_interps_ISW = np.zeros(len(tau_list), dtype=object)

        for i in range(len(tau_list)):
            sounce_interps_SW[i] = interp1d(np.log10(kgrid_i), sources[:, i,  1],
                    kind='cubic', fill_value=0., bounds_error=False)
            sounce_interps_D[i] = interp1d(np.log10(kgrid_i), sources[:, i, 2],
                    kind='cubic', fill_value=0., bounds_error=False)
            sounce_interps_ISW[i] = interp1d(np.log10(kgrid_i), sources[:, i, 3],
                    kind='cubic', fill_value=0., bounds_error=False)

        sources_SW = np.zeros((len(tau_list), self.knum))
        sources_D = np.zeros((len(tau_list), self.knum))
        sources_ISW = np.zeros((len(tau_list), self.knum))
        for i,tau in enumerate(tau_list):
            vis =  self.visibility(tau)
            expD = self.exp_opt_depth(tau)
            sources_SW[i, :] =  vis * sounce_interps_SW[i](np.log10(kgrid_new))
            sources_D[i, :] =  vis * sounce_interps_D[i](np.log10(kgrid_new))
            sources_ISW[i, :] =  expD * sounce_interps_ISW[i](np.log10(kgrid_new))
        return tau_list, sources_SW, sources_D, sources_ISW

    def computeCMB(self, kgrid, ThetaTab):
        print('Computing CMB...')
        cdef cnp.ndarray[double] ell_tab = ThetaTab[0,:]
        cdef cnp.ndarray[double, ndim=2] CL_table = np.zeros((len(ell_tab), 2))
        cdef double GF, ell

        extraNorm = np.zeros_like(kgrid)
        for i,k in enumerate(kgrid):
            eta_st = np.min([1e-3/k, 1e-1/0.7])
            aval = self.ct_to_scale(eta_st)
            ONu = self.OM_nu(aval) / rho_critical / hbar**3. / (2.998e10)**3./ (self.H_0/1e2)**2. / 1e9
            extraNorm[i] = (1. + 2. * ONu / (0.75 * aval**2. * self.OM_M + self.OM_g / aval**-4. + ONu) / 5.)

        for i in range(len(ell_tab)):
            ell = ell_tab[i]
            CL_table[i, 0] = ell
            CL_table[i, 1] =  ell * (ell + 1) * trapz( (ThetaTab[1:, i] / self.init_pert / extraNorm)**2. *
                                                  (kgrid / 0.05)**(self.n_s_index - 1.) / kgrid, kgrid) * self.A_s

            if math.isnan(CL_table[i, 0]):
                print(i, ell)
                print(np.abs(ThetaTab[1:, i]/self.init_pert)**2.)
                print(ThetaTab[1:, i])
                print(cL_interp(np.log10(kgrid)))
                exit()

        Cl_name = path + '/OutputFiles/' + self.Ftag + '_CL_Table' + self.cparam_tag
        Cl_name += '.dat'
        np.savetxt(Cl_name, CL_table)
        return


    def exp_opt_depth(self, eta):
        return self.opt_depth(self.ct_to_scale(eta))

    def visibility(self, eta):
        return self.Vfunc(self.ct_to_scale(eta))

    def vis_max_eta(self):
        etaL = np.logspace(0, log10(self.eta0), 10000)
        visEval = self.visibility(etaL)
        return etaL[np.argmax(visEval)]

    def MatterPower(self, Tktab):
        Tktab[:, 1] /= Tktab[0, 1]
        # T(k) = \Phi(k, a=1) / \Phi(k = Large, a= 1)
        # P(k,a=1) = 2 pi^2 * \delta_H^2 * k / H_0^4 * T(k)^2
        # Tktab = self.TransferFuncs()
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        PS = np.zeros_like(Tktab[:, 0])
        for i, k in enumerate(Tktab[:, 0]):
            PS[i] = k*Tktab[i, 1]**2. * k**(self.n_s_index - 1.)

        np.savetxt(path + '/OutputFiles/' + self.Ftag + '_MatterPowerSpectrum.dat', np.column_stack((Tktab[:, 0], PS)))
        return

    def TransferFuncs(self):
        Minfields = np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(self.kmin))
        LargeScaleVal = Minfields[-1, 1]
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        Tktab = np.zeros_like(self.kgrid)
        for i,k in enumerate(self.kgrid):
            field =  np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(k))
            Tktab[i] = field[-1, 1] / LargeScaleVal
        return Tktab


   

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def trapz(double[:] y, double[:] x):
    if len(x) != len(y):
        raise ValueError('x and y must be same length')
    cdef long npts = len(x)
    cdef double tot = 0
    cdef unsigned int i

    for i in range(npts-1):
        tot += 0.5*(y[i]+y[i+1])*(x[i+1]-x[i])
    return tot




