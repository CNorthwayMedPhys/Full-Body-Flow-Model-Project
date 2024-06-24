# -*- coding: utf-8 -*-

""""
@author: Cassidy.Northway
"""

import numpy as np
import sys
import pandas as pd
import math
from scipy.interpolate import interp1d
from pytictoc import TicToc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
tt = TicToc() 
tt.tic()
np.seterr(all='ignore')

def runSim(lrr_values, mirror_dict):
    
    #%% Utility functions
    
    def periodic(t, T):
        """
        Returns equivalent time of the first period if more than one period is simulated.
        
        :param t: Time.
        :param T: Period length.
        """
        while t/T > 1.0:
            t = t - T
        return t
        
        
    def extrapolate(x0, x, y):
        """
        Returns extrapolated data point given two adjacent data points.
        
        :param x0: Data point to be extrapolated to.
        :param x: x-coordinates of known data points.
        :param y: y-coordinates of known data points.
        """
        return y[0] + (y[1]-y[0]) * (x0 - x[0])/(x[1] - x[0])
    
    
    #%% Import inlet conditions
    def inlet(qc, rc, f_inlet):
        """
        Function describing the inlet boundary condition. Returns a function.
        """
        Q = np.loadtxt(f_inlet, delimiter=',')
        t = [(elem) * qc / rc**3 for elem in Q[:,0]]
        q = [((elem) * 10** 6) / qc for elem in Q[:,1]]  #Unit conversion to cm3/s
        return interp1d(t, q, kind='linear', bounds_error=False, fill_value=q[0])
    
    
#%% Load dataframe
    
    try:
        vessel_df = pd.read_pickle ('C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project\\SysArteries.pkl')
    except:
        vessel_df = pd.read_pickle ('C:\\Users\\Cassidy.Northway\\Remote Git\\SysArteries.pkl')
    
    #2000_Olufsen
    # vessel_df.at[0,'Radius Values'] = [12.5,11.4]
    # vessel_df.at[1,'Radius Values'] = [11.4, 11.1]
    # vessel_df.at[2,'Radius Values'] = [7, 7]
    # vessel_df.at[3,'Radius Values'] = [1.11, 10.9]
    # vessel_df.at[4,'Radius Values'] = [2.9, 2.8]
    # vessel_df.at[7,'Radius Values'] = [10.9, 8.5]
    # vessel_df.at[5,'Radius Values'] = [4.4, 2.8]
    # vessel_df.at[6,'Radius Values'] = [2.9, 2.8]
    
    # vessel_df.at[0, 'lam'] =5.6
    # vessel_df.at[1, 'lam'] = 1.58
    # vessel_df.at[2, 'lam'] =5
    # vessel_df.at[3, 'lam'] =0.9
    # vessel_df.at[4, 'lam'] =65
    # vessel_df.at[7, 'lam'] =17.2
    # vessel_df.at[5, 'lam'] =97
    # vessel_df.at[6, 'lam'] =58.6
   
    #%% Artery object 
    class Artery(object):
        """
        Class representing an artery.
        
        :param pos: Artery ID
        :param Ru: Upstream radius
        :param Rd: Downstream radius
        :param lam: Length-to-radius (upstream) ratio 
        :param k: Iterable containing elasticity parameters k1, k2, k3
        :param Re: Reynold's number
        :param p0: Zero transmural pressure
        :param alpha: radii ratio for structured tree
        :param beta: radii ratio for structured tree
        :param r_min: minimum radius 
        :param Z_term: terminal impedance
        """
            
            
        def __init__(self, pos, Ru, Rd, lam, k, Re, p0, alpha, beta, r_min, Z_term, lrr,rc):
            """
            Artery constructor.
            """
            self._pos = pos
            self._Ru = Ru/rc
            self._Rd = Rd/rc
            self._L = (Ru/rc)*lam
            self._k = k
            self._Re = Re
            self._p0 = p0
            self._alpha = alpha
            self._beta = beta
            self._r_min = r_min/rc
            self._Z_term = Z_term
            self._lrr = lrr
            
        def impedance_weights(self, r_root, dt, T, tc, rc, qc, nu):
            acc = 1e-12 #numerical accuracy of impedance fcn
            r_root = r_root*rc
            dt_temp = 0.01 #Was 0.0001
            N = math.ceil(1/dt_temp)
            eta = acc**(1/(2*N))
            
            m = np.linspace(0,2*np.pi,(2*N)+1) #actual [0:2N-1] the size of 2N
            zeta = eta * np.exp(1j*m)
            Xi = 0.5*(zeta**2) - (2*zeta) + (3/2)
            
            Z_impedance = np.zeros(np.size(Xi), dtype = np.complex_)
            for ii in range(0,np.size(Xi)):
                Z_impedance[ii] = Artery.getImpedance(self, Xi[ii]/(dt_temp), r_root,rc, qc ,nu)
    
            z_n = np.zeros(np.size(Xi), dtype = np.complex_) 
            weighting = np.concatenate(([1],2*np.ones((2*N)-1),[1]))/ (4*N)
            for n in range(0,N+1): # actual range [0,N]
                z_n[n] = np.sum(weighting * Z_impedance * np.exp(-1j*n*m))
                z_n[n] = z_n[n] / (eta ** n)
            z_n = np.real(z_n)
    
            #z_n = z_n * qc / (rho * rc **4)
            #Testing has indicated this is ideal 
            
            return z_n
        
        def getImpedance(self, s, r_root,rc, qc ,nu):
            maxGens = math.ceil(math.log(self.r_min / r_root) / math.log(self.alpha)) + 1
            empty_table = np.empty((maxGens, maxGens))
            empty_table[:] = np.nan
            [Z, table] = Artery.impedance(self, s, r_root,0, 0, empty_table, rc, qc , nu)
            return Z
            
            
        def impedance(self, s, r_root, N_alpha, N_beta, table, rc, qc , nu):
            
            r_0 = r_root * (self.alpha ** (N_alpha)) *(self.beta ** (N_beta))
            
            if r_0 < self.r_min:
                ZL = 0
            else:
                if np.isnan(table[N_alpha + 1, N_beta]):
                    [ZD1, table] = Artery.impedance( self, s,r_root,N_alpha+1 , N_beta,table, rc,qc,nu)
                else:
                    ZD1 = table[N_alpha + 1, N_beta]
         
                if np.isnan(table[N_alpha, N_beta +1]):
                    [ZD2, table] = Artery.impedance( self, s,r_root,N_alpha , N_beta + 1,table, rc,qc,nu)
                else:
                    ZD2 = table[N_alpha , N_beta + 1]
           
                ZL = (ZD1 * ZD2) / (ZD1 + ZD2)
            
            Z0 = Artery.singleVesselImpedance(self, ZL, s ,r_0 , rc, qc , nu)
            table [N_alpha, N_beta] = Z0
            return [Z0, table]
                         
        def singleVesselImpedance(self,ZL, s, r_0, rc,qc, nu):
            
            gamma = 2 #velocity profile 
            nu_temp = nu * qc / rc 
            L = r_0 * self.lrr
            A0 = np.pi * (r_0 ** 2)
            Ehr = (2e7 *np.exp( -22.53*r_0) + 8.65e5) #Youngs Modulus * vessel thickness/radius
            C = (3/2) *(A0)/(Ehr) #complaince
            delta_s = (2 * nu_temp*(gamma +2))/ (rho * (r_0**2))
           
            
        
            if s == 0:
                Z0 = ZL + (2*(gamma +2)*nu_temp* self.lrr) / (np.pi * r_0**3)
                
                    
            else:
                d_s = (A0/(C*rho*s*(s+delta_s)))**(0.5)
                num = ZL + ((np.tanh(L/d_s, dtype=np.longcomplex))/(s*d_s*C))
    
                denom = s*d_s*C*ZL*np.tanh(L/d_s, dtype=np.longcomplex) + 1   
                Z0 = num/denom
                               
            return Z0
        
        def determine_tree(self, dt, T, tc,rc,qc,nu):
            """
            Intiate the tree calculcation for the artery
            
            """
            zn = Artery.impedance_weights(self, self.Rd, dt, T, tc,rc,qc,nu)
            self._zn = zn
            self._Qnk = np.zeros(np.shape(zn)[0]-1)
                               
        def initial_conditions(self, u0, dt, dataframe, mirror_dict, T, tc,rc,qc, nu, flag):
            """
            Initialises solution arrays with initial conditions.
            Checks if artery.mesh(dx) has been called first.
            
            :param u0: Initial conditions for solution
            :param ntr: Number of solution time steps to be stored
            :raises: AttributeError
            """
            if not hasattr(self, '_nx'):
                raise AttributeError('Artery not meshed. Execute mesh(self, dx) \
                                                before setting initial conditions.')
            self.U0[0,:] = self.A0.copy()
            self.U0[1,:].fill(u0)
            if dataframe.at[self.pos,'End Condition'] == 'ST':
                if  flag == 1:
                    Artery.determine_tree(self, dt, T, tc,rc,qc,nu)
              
            else:
                self._zn = 0
                self._Qnk = 0
       
            
        def mesh(self, dx, ntr):
            """
            Meshes an artery using spatial step size dx.
            
            :param dx: Spatial step size
            """
            self._dx = dx
            self._nx = int(self.L/dx)+1
            if self.nx-1 != self.L/dx:
                self.L = dx * (self.nx-1)
            X = np.linspace(0.0, self.L, self.nx)
            R = self.Ru * np.power((self.Rd/self.Ru), X/self.L)
            self._A0 = np.power(R, 2)*np.pi
            self._f = 4/3 * (self.k[0] * np.exp(self.k[1]*R) + self.k[2])
            self._df = 4/3 * self.k[0] * self.k[1] * np.exp(self.k[1]*R)
            self._xgrad = (self.Ru * np.log(self.Rd/self.Ru) * np.power((self.Rd/self.Ru), X/self.L))/self.L
            self.U = np.zeros((2, ntr, self.nx))
            self.P = np.zeros((ntr, self.nx))
            self.U0 = np.zeros((2, self.nx))
           
            
        def boundary_layer_thickness(self, nu, T):
            """
            Calculates the boundary layer thickness of the artery according to
            
            delta = sqrt(nu*T/2*pi).
            
            :param nu: Viscosity of blood
            :param T: Length of one periodic cycle.
            """
            self._delta = np.sqrt(nu*T/(2*np.pi))
            
            
        def p(self, a, **kwargs):
            """
            Calculates pressure according to the state equation.
            
            :param a: Area
            :param **kwargs: See below
            :returns: Pressure 
    
            :Keyword Arguments:
                * *j* (``int``) -- Index variable
            """
            if 'j' in kwargs:
                j = kwargs['j']
                p = self.f[j] * (1 - np.sqrt(self.A0[j]/a)) + self.p0
            else:
                p = self.f * (1 - np.sqrt(self.A0/a)) + self.p0
            return p
            
    
        def wave_speed(self, a):
            """
            Calculates the wave speed (required to check CFL condition).
            
            :param a: Area
            :returns: Wave speed
            """
            return -np.sqrt(0.5 * self.f * np.sqrt(self.A0/a))
            
            
        def F(self, U, **kwargs):
            """
            Calculates the flux vector.
            
            :param U: Previous solution
            :param **kwargs: See below
            :returns: Flux for current solution
            :raises: IndexError
            
            :Keyword Arguments:
                * *j* (``int``) -- Index variable (start)
                * *k* (``int``) -- Index variable (end)
            """
            a, q = U
            out = np.empty_like(U)
            out[0] = q
            if 'j' in kwargs:
                j = kwargs['j']
                a0 = self.A0[j]
                f = self.f[j]
            elif 'k' in kwargs:
                j = kwargs['j']
                k = kwargs['k']
                a0 = self.A0[j:k]
                f = self.f[j:k]
            else:
                raise IndexError("Required to supply at least one index in function F.")
            out[1] = np.power(q, 2)/a + f * np.sqrt(a0*a)
            return out
            
            
        def S(self, U, **kwargs):
            """
            Calculates the flux vector.
            
            :param U: Previous solution
            :param **kwargs: See below
            :returns: Flux for current solution
            :raises: IndexError
            
            :Keyword Arguments:
                * *j* (``int``) -- Index variable (start)
                * *k* (``int``) -- Index variable (end)
            """
            a, q = U
            out = np.zeros(U.shape)
            if 'j' in kwargs:
                j = kwargs['j']
                a0 = self.A0[j]
                xgrad = self.xgrad[j]
                f = self.f[j]
                df = self.df[j]
            elif 'k' in kwargs:
                j = kwargs['j']
                k = kwargs['k']
                a0 = self.A0[j:k]
                xgrad = self.xgrad[j:k]
                f = self.f[j:k]
                df = self.df[j:k]
            else:
                raise IndexError("Required to supply at least one index in function S.")
            R = np.sqrt(a0/np.pi)
            out[1] = -(2*np.pi*R/(self.Re*self.delta)) * (q/a) +\
                    (2*np.sqrt(a) * (np.sqrt(np.pi)*f +\
                    np.sqrt(a0)*df) - a*df) * xgrad
            return out
            
            
        def dBdx(self, l, xi):
            """
            Calculates dB/dx (see [1]).
            
            [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
            
            :param l: Position, either M+1/2 or -1/2.
            :param xi: Area.
            :returns: Solution to dB/dx
            """
            if l > self.L:
                x_0 = self.L-self.dx
                x_1 = self.L
                f_l = extrapolate(l, [x_0, x_1], [self.f[-2], self.f[-1]])  
                A0_l = extrapolate(l, [x_0, x_1], [self.A0[-2], self.A0[-1]])  
                df_l = extrapolate(l, [x_0, x_1], [self.df[-2], self.df[-1]])
                xgrad_l = extrapolate(l, [x_0, x_1],
                                            [self.xgrad[-2], self.xgrad[-1]])
            elif l < 0.0:
                x_0 = self.dx
                x_1 = 0.0
                f_l = extrapolate(l, [x_0, x_1], [self.f[1], self.f[0]])  
                A0_l = extrapolate(l, [x_0, x_1], [self.A0[1], self.A0[0]]) 
                df_l = extrapolate(l, [x_0, x_1], [self.df[1], self.df[0]])
                xgrad_l = extrapolate(l, [x_0, x_1],
                                            [self.xgrad[1], self.xgrad[0]])
            elif l == self.L:
                f_l = self.f[-1]
                A0_l = self.A0[-1]
                df_l = self.df[-1]
                xgrad_l = self.xgrad[-1]
            else:
                f_l = self.f[0]
                A0_l = self.A0[0]
                df_l = self.df[0]
                xgrad_l = self.xgrad[0]
            return (2*np.sqrt(xi) * (np.sqrt(np.pi)*f_l + np.sqrt(A0_l)*df_l) -\
                        xi*df_l) * xgrad_l
            
            
        def dBdxi(self, l, xi):
            """
            Calculates dB/dx_i (see [1]).
            
            [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
            
            :param l: Position, either M+1/2 or -1/2.
            :param xi: Area.
            :returns: Solution to dB/dx_i
            """
            if l > self.L:
                x_0 = self.L-self.dx
                x_1 = self.L
                f_l = extrapolate(l, [x_0, x_1], [self.f[-2], self.f[-1]])  
                A0_l = extrapolate(l, [x_0, x_1], [self.A0[-2], self.A0[-1]])  
            elif l < 0.0:
                x_0 = self.dx
                x_1 = 0.0
                f_l = extrapolate(l, [x_0, x_1], [self.f[1], self.f[0]])  
                A0_l = extrapolate(l, [x_0, x_1], [self.A0[1], self.A0[0]]) 
            elif l == self.L:
                f_l = self.f[-1]
                A0_l = self.A0[-1]
            else:
                f_l = self.f[0]
                A0_l = self.A0[0]
            return f_l/2 * np.sqrt(A0_l/xi)
            
            
        def dBdxdxi(self, l, xi):
            """
            Calculates d^2B/dxdx_i (see [1]).
            
            [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
            
            :param l: Position, either M+1/2 or -1/2.
            :param xi: Area.
            :returns: Solution to d^2B/dxdx_i
            """
            if l > self.L:
                x_0 = self.L-self.dx
                x_1 = self.L
                f_l = extrapolate(l, [x_0, x_1], [self.f[-2], self.f[-1]])   
                df_l = extrapolate(l, [x_0, x_1], [self.df[-2], self.df[-1]])   
                A0_l = extrapolate(l, [x_0, x_1], [self.A0[-2], self.A0[-1]])  
                xgrad_l = extrapolate(l, [x_0, x_1],
                                            [self.xgrad[-2], self.xgrad[-1]])
            elif l < 0.0:
                x_0 = self.dx
                x_1 = 0.0
                f_l = extrapolate(l, [x_0, x_1], [self.f[1], self.f[0]])   
                df_l = extrapolate(l, [x_0, x_1], [self.df[1], self.df[0]])   
                A0_l = extrapolate(l, [x_0, x_1], [self.A0[1], self.A0[0]])  
                xgrad_l = extrapolate(l, [x_0, x_1],
                                            [self.xgrad[1], self.xgrad[0]])
            elif l == self.L:
                f_l = self.f[-1]   
                df_l = self.df[-1]
                A0_l = self.A0[-1]
                xgrad_l = self.xgrad[-1]
            else:
                f_l = self.f[0]   
                df_l = self.df[0]
                A0_l = self.A0[0]
                xgrad_l = self.xgrad[0]
         
            return (1/(2*np.sqrt(xi)) * (f_l*np.sqrt(np.pi) +\
                                        df_l*np.sqrt(A0_l)) - df_l) * xgrad_l
            
                                        
                                        
        def dFdxi2(self, l, xi1, xi2):
            """
            Calculates dF/dx_2 (see [1]).
            
            [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
            
            :param l: Position, either M+1/2 or -1/2.
            :param xi: Area.
            :returns: Solution to dF/dx_2
            """
            if l > self.L:
                x_0 = self.L-self.dx
                x_1 = self.L
                R0_l = extrapolate(l, [x_0, x_1], 
                        [np.sqrt(self.A0[-2]/np.pi), np.sqrt(self.A0[-1]/np.pi)])
            elif l < 0.0:
                x_0 = self.dx
                x_1 = 0.0
                R0_l = extrapolate(l, [x_0, x_1], 
                        [np.sqrt(self.A0[1]/np.pi), np.sqrt(self.A0[0]/np.pi)])
            elif l == self.L:
                R0_l = np.sqrt(self.A0[-1]/np.pi)
            else:
                R0_l = np.sqrt(self.A0[0]/np.pi)
            return 2*np.pi*R0_l/(self.delta*self.Re) * xi1/(xi2*xi2)
            
            
        def dFdxi1(self, l, xi2):
            """
            Calculates dF/dx_1 (see [1]).
            
            [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
            
            :param l: Position, either M+1/2 or -1/2.
            :param xi: Area.
            :returns: Solution to dF/dx_1
            """
            if l > self.L:
                x_0 = self.L-self.dx
                x_1 = self.L
                R0_l = extrapolate(l, [x_0, x_1], 
                        [np.sqrt(self.A0[-2]/np.pi), np.sqrt(self.A0[-1]/np.pi)])
            elif l < 0.0:
                x_0 = self.dx
                x_1 = 0.0
                R0_l = extrapolate(l, [x_0, x_1], 
                        [np.sqrt(self.A0[1]/np.pi), np.sqrt(self.A0[0]/np.pi)])
            elif l == self.L:
                R0_l = np.sqrt(self.A0[-1]/np.pi)
            else:
                R0_l = np.sqrt(self.A0[0]/np.pi)
            return -2*np.pi*R0_l/(self.delta*self.Re) * 1/xi2
            
            
        def dpdx(self, l, xi):
            """
            Calculates dp/dx (see [1]).
            
            [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
            
            :param l: Position, either M+1/2 or -1/2.
            :param xi: Area.
            :returns: Solution to dp/dx
            """
            if l > self.L:
                x_0 = self.L-self.dx
                x_1 = self.L
                f_l = extrapolate(l, [x_0, x_1], [self.f[-2], self.f[-1]])   
                A0_l = extrapolate(l, [x_0, x_1], [self.A0[-2], self.A0[-1]])  
            elif l < 0.0:
                x_0 = self.dx
                x_1 = 0.0
                f_l = extrapolate(l, [x_0, x_1], [self.f[1], self.f[0]])   
                A0_l = extrapolate(l, [x_0, x_1], [self.A0[1], self.A0[0]])
            elif l == self.L:
                f_l = self.f[-1]   
                A0_l = self.A0[-1]
            else:
                f_l = self.f[0]   
                A0_l = self.A0[0]
            return f_l/2 * np.sqrt(A0_l/xi**3)
            
            
        def solve(self, lw, U_in, U_out, save, i):
            """
            Solver calling the LaxWendroff solver and storing the new solution in U0.lw
            Stores new solution in output array U if save is True.
            
            :param lw: LaxWendroff object
            :param U_in: Inlet boundary condition
            :param U_out: Outlet boundary condition
            :param save: True if current time step is to be saved
            :param i: Current time step
            """
            # solve for current timestep
            U1 = lw.solve(self.U0, U_in, U_out, self.F, self.S)
            if save:
                self.P[i,:] = self.p(self.U0[0,:])
                np.copyto(self.U[:,i,:], self.U0)
            np.copyto(self.U0, U1)
            
    
        def dump_results(self, suffix, data_dir):
            """
            Outputs solutions U, P to csv files
            
            :param suffix: Simulation identifier
            :param data_dir: Directory data files are stored in
            """
            np.savetxt("%s/%s/u%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                       self.U[1,:,:], delimiter=',')
            np.savetxt("%s/%s/a%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                       self.U[0,:,:], delimiter=',')  
            np.savetxt("%s/%s/p%d_%s.csv" % (data_dir, suffix, self.pos, suffix),
                       self.P, delimiter=',') 
                       
                       
        @property
        def L(self):
            """
            Artery length
            """
            return self._L
    
        @L.setter
        def L(self, value):
            self._L = value
            
        @property
        def nx(self):
            """
            Number of spatial steps
            """
            return self._nx
            
        @property
        def Ru(self):
            """
            Upstream radius
            """
            return self._Ru
            
        @property
        def Rd(self):
            """
            Downstream radius
            """
            return self._Rd
            
        @property
        def k(self):
            """
            Elasticity parameters for relation Eh/r = k1 * exp(k2*r) + k3
            """
            return self._k
            
        @property
        def A0(self):
            """
            Area at rest
            """
            return self._A0
            
        @property
        def dx(self):
            """
            Spatial step size
            """
            return self._dx
        
        @property
        def pos(self):
            """
            Position in ArteryNetwork
            """
            return self._pos
            
        @property
        def f(self):
            """
            f = 4/3 Eh/r
            """
            return self._f
            
        @property
        def xgrad(self):
            """
            dr/dx
            """
            return self._xgrad
            
        @property
        def df(self):
            """
            df/dr
            """        
            return self._df
    
        @property
        def Re(self):
            """
            Reynold's number
            """
            return self._Re
            
        @property
        def delta(self):
            """
            Boundary layer thickness
            """
            return self._delta
    
        @property
        def p0(self):
            """
            Zero transmural pressure
            """
            return self._p0
        
        @property
        def alpha(self):
            """
            Radii ratio for structured tree calculations
            """
            return self._alpha
        
        @property
        def beta(self):
            """
            Radii ratio for structured tree calculations
            """
            return self._beta                     
                         
        @property
        def r_min(self):
            """
            Minimum radius at which structured trees terminate
            """
            return self._r_min
                         
                         
        @property
        def Z_term(self):
            """
            Terminal impedance of structured trees
            """
            return self._Z_term 
        
        @property
        def lrr(self):
            """
            Length to radius ratio of structred treees
            """
            return self._lrr
                         
        @property
        def zn(self):
            """
            Impedance weights 
            """
            return self._zn
        
        @property
        def Qnk(self):
            """
            Stored flow values for structured tree values
            """
            return self._Qnk
    
    
    #%% Artery Network 
    
    from scipy import linalg
    from os import makedirs
    from os.path import exists
    
    class ArteryNetwork(object):
        """
        Class representing a network of arteries.
        
        :param Ru: Iterable containing upstream radii.
        :param Rd: Iterable containing downstream radii.
        :param lam: Iterable containing length-to-radius ratios.
        :param k: Iterable containing elasticity parameters.
        :param rho: Density of blood.
        :param nu: Viscosity of blood.
        :param p0: Zero transmural pressure.
        :param depth: Depth of the arterial tree, e. g. 1 for one artery, 2 for three arteries. ##I WANT TO EDIT THIS CONCEPT OUT
        :param ntr: Number of time steps in output.
        :param Re: Reynolds number.
        :param dataframe: Dataframe containing the artery information
        :param alpha: Radii ratio for structured tree
        :param beta: Radii ratio for structured tree
        :param r_min: Minimum radii
        :param Z_term: Terminal impedance
        """
        
        
        def __init__(self, rho, nu, p0, ntr, Re, k, dataframe, Z_term, r_min, lrr,rc,mirror_dict):
            """
            ArteryNetwork constructor.
            """
            #MODIFIED TO: NOT USE DEPTH TO create all the arteries but create arteries for all arteries in dataframe    
            self._t = 0.0
            self._ntr = ntr
            self._progress = 0
            self._rho = rho
            self._nu = nu
            self._p0 = p0
            self._dataframe = dataframe 
            self._arteries = [0] * len(dataframe)
            self.setup_arteries(Re, p0, k, r_min, Z_term, lrr,rc, mirror_dict)     
            
        def setup_arteries(self, Re, p0, k, r_min, Z_term, lrr,rc, mirror_dict):
            """
            Creates Artery objects.
            
            :param Ru: Iterable containing upstream radii.
            :param Rd: Iterable containing downstream radii.
            :param lam: Iterable containing length-to-radius ratios.
            :param k: Iterable containing elasticity parameters.
            :param Re: Reynolds number.
            :param Z_term: 
            """
            
            #Creates all artery objects 
            
            for i in range(0,len(self.dataframe)):
                Ru = self.dataframe.at[i,'Radius Values'][0] / 10 #From mm to cm 
                Rd = self.dataframe.at[i,'Radius Values'][1] / 10 #From mm to cm 
                lam = self.dataframe.at[i,'lam'] 
                cndt = self.dataframe.at[i,'End Condition']
                #find position in mirroring dict of i, find row number == ii
            
                if cndt == 'ST':
                    ############NEW CODE HERE##########################
                    if Rd > 0.025:
                        xi = 2.5
                        zeta = 0.4
                    elif Rd <= 0.005:
                        xi = 2.9
                        zeta = 0.9          
                    else:
                        xi = 2.76
                        zeta = 0.6
                    alpha = (1+zeta**(xi/2))**(-1/xi)
                    beta = alpha * np.sqrt(zeta)
                    [row_index, col_index] = np.where(mirror_dict == i)
                    self.arteries[i] = Artery(i, Ru, Rd, lam, k, Re, p0, alpha, beta, r_min, Z_term, lrr[row_index[0]] ,rc)
                else:  
                    self.arteries[i] = Artery(i, Ru, Rd, lam, k, Re, p0, 0, 0, r_min, Z_term, 0,rc)
    
                       
        def initial_conditions(self, intial_values, dataframe,mirror_dict, rc,qc):
            """
            Invokes initial_conditions(u0) on each artery in the network.
            
            :param u0: Initial condition for U_1.
            """
            for artery in self.arteries:
                flag = 0
                index  = artery.pos
                cndt = self.dataframe.at[index,'End Condition']
                u0 = intial_values[index]
                if cndt == 'ST':
                    [row_index, col_index] = np.where(mirror_dict == index)
                    if col_index == 0:
                        if mirror_dict[row_index,1]== 0:
                            #If no mirrored vessels then go calculate the ST values!
                            flag = 1
                        elif mirror_dict[row_index,0] < mirror_dict[row_index,1]:
                            #If we have yet to calcualte it's twins values then calc the ST
                            flag = 1
                                
                        else: #mirror_dict[row_index,1]<mirror_dict[row_index,0] 
                        #We have already calculated the st for it's twin thus we can call those values
                            twin_index = int(mirror_dict[row_index,1])
                            twin_artery =self.arteries[twin_index]
                            artery._zn = twin_artery.zn
                            artery._Qnk = twin_artery.Qnk
                    else: #col_index = 1
                        if mirror_dict[row_index,0] > mirror_dict[row_index,1]:
                            #We have not yet calculated it's twin
                            flag = 1
                        else: #mirror_dict[row_index,0] > mirror_dict[row_index,1]
                        #We have calcualted it's twin
                            twin_index = int(mirror_dict[row_index,0])
                            twin_artery =self.arteries[twin_index]
                            artery._zn = twin_artery.zn
                            artery._Qnk = twin_artery.Qnk

                artery.initial_conditions(u0, self.dt, dataframe, mirror_dict, self.T, self.tc,rc,qc, self.nu, flag)
                
                
                
        def mesh(self, dx):
            """
            Invokes mesh(nx) on each artery in the network
            
            :param dx: Spatial step size
            """
            for artery in self.arteries:
                artery.mesh(dx, self.ntr)
                
                
        def set_time(self, dt, T, tc=1):
            """
            Sets timing parameters for the artery network and invokes
            boundary_layer_thickness(T) in each artery.
            
            :param dt: Time step size.
            :param T: Length of one periodic cycle.
            :param tc: Number of cycles.
            """
            self._dt = dt
            self._tf = T*tc
            self._dtr = self.tf/self.ntr
            self._T = T
            self._tc = tc
            for artery in self.arteries:
                artery.boundary_layer_thickness(self.nu, T)
                
                
        def timestep(self):
            """
            Increases time by dt.
            """
            self._t += self.dt
                
        
        @staticmethod        
        def inlet_bc(artery, q_in, in_t, dt):
            """
            Calculates inlet boundary condition.
            
            :param artery: Inlet artery.
            :param q_in: Function containing inlet condition U_1(t).
            :param in_t: Current time.
            :param dt: Time step size.
            :returns: Array containing solution U at the inlet. 
            """
            q_0_np = q_in(in_t-dt/2) # q_0_n+1/2
            q_0_n1 = q_in(in_t) # q_0_n+1
            U_0_n = artery.U0[:,0] # U_0_n
            U_1_n = artery.U0[:,1]
            U_12_np = (U_1_n+U_0_n)/2 -\
                        dt*(artery.F(U_1_n, j=1)-artery.F(U_0_n, j=0))/(2*artery.dx) +\
                        dt*(artery.S(U_1_n, j=1)+artery.S(U_0_n, j=0))/4 # U_1/2_n+1/2
            a_0_n1 = U_0_n[0] - 2*dt*(U_12_np[1] - q_0_np)/artery.dx
            return np.array([a_0_n1, q_0_n1])
         
        
        @staticmethod
        def outlet_wk3(artery, dt, R1, R2, Ct):
            """
            Function calculating the three-element Windkessel outlet boundary
            condition.
            
            :param artery: Artery object of outlet artery
            :param dt: time step size
            :param R1: first resistance element
            :param R2: second resistance element
            :param Ct: compliance element
            :returns: Numpy array containing the outlet area and flux
            """
            theta = dt/artery.dx
            gamma = dt/2
            U0_1 = artery.U0[:,-1] # m = M
            U0_2 = artery.U0[:,-2] # m = M-1
            U0_3 = artery.U0[:,-3] # m = M-2
            a_n, q_n = U0_1
            p_new = p_n = artery.p(a_n, j=-1) # initial guess for p_out
            U_np_mp = (U0_1 + U0_2)/2 +\
                    gamma * (-(artery.F(U0_1, j=-1) - artery.F(U0_2, j=-2))/artery.dx +\
                            (artery.S(U0_1, j=-1) + artery.S(U0_2, j=-2))/2)
            U_np_mm = (U0_2 + U0_3)/2 +\
                    gamma * (-(artery.F(U0_2, j=-2) - artery.F(U0_3, j=-3))/artery.dx +\
                            (artery.S(U0_2, j=-2) + artery.S(U0_3, j=-3))/2)
            U_mm = U0_2 - theta*(artery.F(U_np_mp, j=-2) - artery.F(U_np_mm, j=-2)) +\
                    gamma*(artery.S(U_np_mp, j=-2) + artery.S(U_np_mm, j=-2))
            k = 0
            X = dt/(R1*R2*Ct)
            while k < 1000:
                p_old = p_new
                q_out = X*p_n - X*(R1+R2)*q_n + (p_old-p_n)/R1 + q_n
                a_out = a_n - theta * (q_out - U_mm[1])
                p_new = artery.p(a_out, j=-1)
                if abs(p_old - p_new) < 1e-7:
                    break
                k += 1
            return np.array([a_out, q_out])
        
        
        @staticmethod
        def outlet_p(artery, dt, P):
            """
            Function calculating cross-sectional area and flow rate for a fixed
            pressure outlet boundary condition.
            
            :param artery: Artery object of outlet artery
            :param dt: time step size
            :param P: outlet pressure
            """
            theta = dt/artery.dx
            gamma = dt/2
            U0_1 = artery.U0[:,-1]
            U0_2 = artery.U0[:,-2]
            a_n, q_n = U0_1
            p_out = P # initial guess for p_out
            a_out = (artery.A0[-1]*artery.f[-1]**2) / (artery.f[-1] - p_out)**2
            U_np_mm = (U0_1 + U0_2)/2 -\
                    theta*(artery.F(U0_1, j=-1) - artery.F(U0_2, j=-2))/2 +\
                    gamma*(artery.S(U0_1, j=-1) + artery.S(U0_2, j=-2))/2
            a_np_mp = 2*a_out - U_np_mm[0]
            q_np_mp = (a_n - a_out)/theta + U_np_mm[1]
            U_np_mp = np.array([a_np_mp, q_np_mp])
            U_out = U0_1 - theta*(artery.F(U_np_mp, j=-1) - artery.F(U_np_mm, j=-1)) +\
                    gamma*(artery.S(U_np_mp, j=-1) + artery.S(U_np_mm, j=-1))
            return U_out
        
    
        @staticmethod
        def outlet_st(artery, dt, t):
            """
            :param t: Current time step, within the period, 0<=t<=T
            """
            
            
            k_array = np.arange(0,t,dt) #actual range [0,t]
            n_value = np.size(k_array)
            
            if n_value+1 < np.size(artery.zn):
                zk_array = artery.zn[0:n_value+1]
                Qnk_array = artery.Qnk[0:n_value]
            else:
                zk_array = artery.zn
                Qnk_array = artery.Qnk
                
            #Need to have stored Q values for every time step up to this point
            #for k = 0 to n (n=current number of time steps)
            #p_out = np.sum(zk_array*Qnk_array)  #pressure at nth time step with constant time steps dt          
            #Here I take the outlet_3wk code from above and attempt to modify for my needs
            theta = dt/artery.dx
            gamma = dt/2
            U0_1 = artery.U0[:,-1] # m = M
            U0_2 = artery.U0[:,-2] # m = M-1
            U0_3 = artery.U0[:,-3] # m = M-2
            a_n, q_n = U0_1
            p_new = artery.p(a_n, j=-1) # initial guess for p_out
            U_np_mp = (U0_1 + U0_2)/2 +\
                    gamma * (-(artery.F(U0_1, j=-1) - artery.F(U0_2, j=-2))/artery.dx +\
                            (artery.S(U0_1, j=-1) + artery.S(U0_2, j=-2))/2)
            U_np_mm = (U0_2 + U0_3)/2 +\
                    gamma * (-(artery.F(U0_2, j=-2) - artery.F(U0_3, j=-3))/artery.dx +\
                            (artery.S(U0_2, j=-2) + artery.S(U0_3, j=-3))/2)
            U_mm = U0_2 - theta*(artery.F(U_np_mp, j=-2) - artery.F(U_np_mm, j=-2)) +\
                    gamma*(artery.S(U_np_mp, j=-2) + artery.S(U_np_mm, j=-2))
            k = 0
            
            while k < 1000:
                p_old = p_new
                q_out = (p_old - np.sum(zk_array[1:]*Qnk_array))/zk_array[0]
                a_out = a_n - theta * (q_out - U_mm[1])
                p_new = artery.p(a_out, j=-1)
                if abs(p_old - p_new) < 1e-7:                
                    break
                k += 1
            return np.array([a_out, q_out])
    
             #Do we want to reset k at t>T or do we want to have total t regardless of number of T
        @staticmethod
        def jacobian(x, parent, d1, d2, theta, gamma):
            """
            Calculates the Jacobian for using Newton's method to solve bifurcation inlet and outlet boundary conditions [1].
            
            [1] [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
            
            :param x: Solution of the system of equations.
            :param parent: Artery object of the parent vessel.
            :param d1: Artery object of the first daughter vessel.
            :param d2: Artery object of the second daughter vessel.
            :param theta: dt/dx
            :param gamma: dt/2
            :returns: The Jacobian for Newton's method.
            """
            ####Added in K_loss modelled after Chambers_et__al_2020 from Olufsen Github [arteries.c]
            if d1.pos == 1:
                LD_k = 0.75#/2
                RD_k = 0
               
            elif d2.pos == 1:
                RD_k =0.75/2
                LD_k = 0
               
            else:
                RD_k = 0
                LD_k = 0
                
            M12 = parent.L + parent.dx/2
            D1_12 = -d1.dx/2
            D2_12 = -d2.dx/2
            zeta7 = -parent.dpdx(parent.L, x[10])
            zeta10 = -parent.dpdx(parent.L, x[9])
            Dfr = np.zeros((18, 18)) # Jacobian
            Dfr[0,0] = Dfr[1,3] = Dfr[2,6] = Dfr[3,9] = Dfr[4,12] = Dfr[5,15] = -1
            Dfr[6,1] = Dfr[7,4] = Dfr[8,7] = Dfr[9,10] = Dfr[10,13] = Dfr[11,16] = -1
            Dfr[12,1] = Dfr[13,0] = -1
            Dfr[6,2] = Dfr[7,5] = Dfr[8,8] = Dfr[9,11] = Dfr[10,14] = Dfr[11,17] = 0.5
            Dfr[12,4] = Dfr[12,7] = Dfr[13,3] = Dfr[13,6] = 1.0
            Dfr[3,2] = -theta
            Dfr[4,5] = Dfr[5,8] = theta
            Dfr[0,2] = -2*theta*x[2]/x[11] + gamma*parent.dFdxi1(M12, x[11])
            Dfr[0,11] = theta * (x[2]**2/x[11]**2 - parent.dBdxi(M12,x[11])) +\
                        gamma * (parent.dFdxi2(M12, x[2], x[11]) +\
                                parent.dBdxdxi(M12, x[11]))
            Dfr[1,5] = 2*theta*x[5]/x[14] + gamma*d1.dFdxi1(D1_12, x[14])
            Dfr[1,14] = theta * (-x[5]**2/x[14]**2 + d1.dBdxi(D1_12,x[14])) +\
                        gamma * (d1.dFdxi2(D1_12, x[5], x[14]) +\
                                d1.dBdxdxi(D1_12, x[14]))
           
            Dfr[2,8] = 2*theta*x[8]/x[17] + gamma*d2.dFdxi1(D2_12, x[17])
            
            Dfr[2,17] = theta * ((-x[8]**2/x[17]**2 + 0) +\
                        gamma * (0) +\
                                d2.dBdxdxi(D2_12, x[17]))
           
            Dfr[14,10] = zeta7
            Dfr[14,13] = d1.dpdx(0.0, x[13])
            Dfr[15,16] = d2.dpdx(0.0, x[16])
            Dfr[16,12] = d1.dpdx(0.0, x[12])
            Dfr[17,9] = zeta10
            Dfr[17,15] = d2.dpdx(0.0, x[15])
            
            ####K_loss implementation
            if x[0] > 0:
                Dfr[16,0] = (x[0]/(x[9])**2)+(2*LD_k)
                Dfr[17,0] = (x[0]/(x[9])**2)+(2*RD_k)
                Dfr[16,9] = zeta10 +(-2*LD_k)*(x[0]**2 / x[9]**3)
                Dfr[17,9] = zeta10 +(-2*RD_k)*(x[0]**2 / x[9]**3)
            else:
                Dfr[16,0] = (x[0]/(x[9])**2)+(-2*LD_k)
                Dfr[17,0] = (x[0]/(x[9])**2)+(-2*RD_k)
                Dfr[16,9] = zeta10 +(2*LD_k)*(x[0]**2 / x[9]**3)
                Dfr[17,9] = zeta10 +(2*RD_k)*(x[0]**2 / x[9]**3)
                
            if x[1] > 0 :
                Dfr [14,1] = x[1]/(x[10]**2)*(2*LD_k)
                Dfr [15,1] = x[1]/(x[10]**2)*(2*RD_k)
                Dfr[14,10] = zeta7 +(-2*LD_k)*(x[1]**2 / x[10]**3)
                Dfr[15,10] = zeta7 +(-2*RD_k)*(x[1]**2 / x[10]**3)
            else:
                Dfr [14,1] = x[1]/(x[10]**2)*(-2*LD_k)
                Dfr [15,1] = x[1]/(x[10]**2)*(-2*RD_k)
                Dfr[14,10] = zeta7 +(2*LD_k)*(x[1]**2 / x[10]**3)
                Dfr[15,10] = zeta7 +(2*RD_k)*(x[1]**2 / x[10]**3)
            
       
                    
                
                
                
            return Dfr
            
    
        @staticmethod
        def residuals(x, parent, d1, d2, theta, gamma, U_p_np, U_d1_np, U_d2_np):
            """
            Calculates the residual equations for using Newton's method to solve bifurcation inlet and outlet boundary conditions [1].
            
            [1] M. S. Olufsen. Modeling of the Arterial System with Reference to an Anesthesia Simulator. PhD thesis, University of Roskilde, Denmark, 1998.
            [2] R. J. LeVeque. Numerical Methods for Conservation Laws. Birkhauser Verlag, Basel, Switzerland, 2nd edition, 1992.
            
            :param x: Solution of the system of equations.
            :param parent: Artery object of the parent vessel.
            :param d1: Artery object of the first daughter vessel.
            :param d2: Artery object of the second daughter vessel.
            :param theta: dt/dx
            :param gamma: dt/2
            :param U_p_np: U_(M-1/2)^(n+1/2) [2]
            :param U_p_np: U_(M-1/2)^(n+1/2) [2]
            :returns: The residual equations for Newton's method.

            """
            ####Added in K_loss modelled after Chambers_et__al_2020 from Olufsen Github [arteries.c]
            if d1.pos == 1:
                LD_k = 0.75#/2
                RD_k = 0
            elif d2.pos == 1:
                RD_k =0.75/2
                LD_k = 0
            else:
                RD_k = 0
                LD_k = 0
            
            f_p_mp = extrapolate(parent.L+parent.dx/2,
                    [parent.L-parent.dx, parent.L], [parent.f[-2], parent.f[-1]])
            f_d1_mp = extrapolate(-d1.dx/2, [d1.dx, 0.0],
                                        [d1.f[1], d1.f[0]])
            f_d2_mp = extrapolate(-d2.dx/2, [d2.dx, 0.0],
                                        [d2.f[1], d2.f[0]])
            A0_p_mp = extrapolate(parent.L+parent.dx/2,
                    [parent.L-parent.dx, parent.L], [parent.A0[-2], parent.A0[-1]])
            A0_d1_mp = extrapolate(-d1.dx/2, [d1.dx, 0.0],
                                         [d1.A0[1], d1.A0[0]])
            A0_d2_mp = extrapolate(-d2.dx/2, [d2.dx, 0.0],
                                         [d2.A0[1], d2.A0[0]])
            R0_p_mp = np.sqrt(A0_p_mp/np.pi)
            R0_d1_mp = np.sqrt(A0_d1_mp/np.pi)
            R0_d2_mp = np.sqrt(A0_d2_mp/np.pi)
            B_p_mp = f_p_mp * np.sqrt(x[11]*A0_p_mp)
            B_d1_mp = f_d1_mp * np.sqrt(x[14]*A0_d1_mp)
            B_d2_mp = f_d2_mp * np.sqrt(x[17]*A0_d2_mp)
            k1 = parent.U0[1,-1] + theta * (parent.F(U_p_np, j=-1)[1]) +\
                    gamma * (parent.S(U_p_np, j=-1)[1])
            k2 = d1.U0[1,0] - theta * (d1.F(U_d1_np, j=0)[1]) +\
                    gamma * (d1.S(U_d1_np, j=0)[1])
            k3 = d2.U0[1,0] - theta * (d2.F(U_d2_np, j=0)[1]) +\
                    gamma * (d2.S(U_d2_np, j=0)[1])
            k4 = parent.U0[0,-1] + theta*parent.F(U_p_np, j=-1)[0]
            k5 = d1.U0[0,0] - theta*d1.F(U_d1_np, j=0)[0]
            k6 = d2.U0[0,0] - theta*d2.F(U_d2_np, j=0)[0]
            k7 = U_p_np[1]/2
            k8 = U_d1_np[1]/2
            k9 = U_d2_np[1]/2
            k10 = U_p_np[0]/2
            k11 = U_d1_np[0]/2
            k12 = U_d2_np[0]/2
            k15a = -parent.f[-1] + d1.f[0]
            k15b = d1.f[0] * np.sqrt(d1.A0[0])
            k16a = -parent.f[-1] + d2.f[0]
            k16b = d2.f[0] * np.sqrt(d2.A0[0])
            k156 = parent.f[-1] * np.sqrt(parent.A0[-1])
            fr1 = k1 - x[0] - theta*(x[2]**2/x[11] + B_p_mp) +\
                    gamma*(-2*np.pi*R0_p_mp*x[2]/(parent.delta*parent.Re*x[11]) +\
                    parent.dBdx(parent.L+parent.dx/2, x[11]))
            fr2 = k2 - x[3] + theta*(x[5]**2/x[14] + B_d1_mp) +\
                    gamma*(-2*np.pi*R0_d1_mp*x[5]/(d1.delta*d1.Re*x[14]) +\
                    d1.dBdx(-d1.dx/2, x[14]))
            fr3 = k3 - x[6] + theta*(x[8]**2/x[17] + B_d2_mp) +\
                    gamma*(-2*np.pi*R0_d2_mp*x[8]/(d2.delta*d2.Re*x[17]) +\
                    d2.dBdx(-d2.dx/2, x[17]))
            fr4 = -x[9] - theta*x[2] + k4
            fr5 = -x[12] + theta*x[5] + k5
            fr6 = -x[15] + theta*x[8] + k6
            fr7 = -x[1] + x[2]/2 + k7
            fr8 = -x[4] + x[5]/2 + k8
            fr9 = -x[7] + x[8]/2 + k9
            fr10 = -x[10] + x[11]/2 + k10
            fr11 = -x[13] + x[14]/2 + k11
            fr12 = -x[16] + x[17]/2 + k12
            fr13 = -x[1] + x[4] + x[7]
            fr14 = -x[0] + x[3] + x[6]
            
            #### K_loss implementation######
            sq211 = (x[1]/x[10])**2
            
            if x[1] > 0:
                fr15 = k156/np.sqrt(x[10]) - k15b/np.sqrt(x[13]) + k15a + LD_k * sq211
                fr16 = k156/np.sqrt(x[10]) - k16b/np.sqrt(x[16]) + k16a + RD_k * sq211
            else:
                fr15 = k156/np.sqrt(x[10]) - k15b/np.sqrt(x[13]) + k15a - LD_k * sq211
                fr16 = k156/np.sqrt(x[10]) - k16b/np.sqrt(x[16]) + k16a - RD_k * sq211
                
            sq110 = (x[0]/x[9]) ** 2    
            if x[0] > 0:
                fr17 = k156/np.sqrt(x[9]) - k15b/np.sqrt(x[12]) + k15a + LD_k * sq110
                fr18 = k156/np.sqrt(x[9]) - k16b/np.sqrt(x[15]) + k16a + RD_k * sq110        
            else:
                fr17 = k156/np.sqrt(x[9]) - k15b/np.sqrt(x[12]) + k15a - LD_k * sq110
                fr18 = k156/np.sqrt(x[9]) - k16b/np.sqrt(x[15]) + k16a - RD_k * sq110
                
            return np.array([fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10,
                             fr11, fr12, fr13, fr14, fr15, fr16, fr17, fr18])
            
    
        @staticmethod
        def bifurcation(parent, d1, d2, dt):
            """
            Calculates the bifurcation boundary condition using Newton's method.
            
            :param parent: Artery object of the parent vessel.
            :param d1: Artery object of the first daughter vessel.
            :param d2: Artery object of the second daughter vessel.
            :param dt: Time step size.
            :returns: Array containing the solution at the bifurcation boundary.
            """
            
            theta = dt/parent.dx
            gamma = dt/2
            U_p_np = (parent.U0[:,-1] + parent.U0[:,-2])/2 -\
                    theta*(parent.F(parent.U0[:,-1], j=-1) - parent.F(parent.U0[:,-2], j=-2))/2 +\
                    gamma*(parent.S(parent.U0[:,-1], j=-1) + parent.S(parent.U0[:,-2], j=-2))/2
            U_d1_np = (d1.U0[:,1] + d1.U0[:,0])/2 -\
                    theta*(d1.F(d1.U0[:,1], j=1) - d1.F(d1.U0[:,0], j=0))/2 +\
                    gamma*(d1.S(d1.U0[:,1], j=1) + d1.S(d1.U0[:,0], j=0))/2
                   
            U_d2_np = (d2.U0[:,1] + d2.U0[:,0])/2 -\
                    theta*(d2.F(d2.U0[:,1], j=1) - d2.F(d2.U0[:,0], j=0))/2 +\
                    gamma*(d2.S(d2.U0[:,1], j=1) + d2.S(d2.U0[:,0], j=0))/2
            
            x0 = U_p_np[1]
            x1 = (parent.U0[1,-1] + parent.U0[1,-2])/2
            x2 = parent.U0[1,-1]
            x3 = U_d1_np[1]
            x4 = (d1.U0[1,0] + d1.U0[1,1])/2
            x5 = d1.U0[1,0]
            x6 = U_d2_np[1]
            x7 = (d2.U0[1,0] + d2.U0[1,1])/2
            x8 = d2.U0[1,0]
            x9 = U_p_np[0]
            x10 = (parent.U0[0,-1] + parent.U0[0,-2])/2
            x11 = parent.U0[0,-1]
            x12 = U_d1_np[0]
            x13 = (d1.U0[0,0] + d1.U0[0,1])/2
            x14 = d1.U0[0,0]
            x15 = U_d2_np[0]
            x16 = (d2.U0[0,0] + d2.U0[0,1])/2   
            x17 = d2.U0[0,0]
            x = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17])
            k = 0
            while k < 40:#1000:
                #############Debugging###################
                try:
                    Dfr = ArteryNetwork.jacobian(x, parent, d1, d2, theta, gamma)
                    Dfr_inv = linalg.inv(Dfr)
                    fr = ArteryNetwork.residuals(x, parent, d1, d2, theta, gamma, U_p_np, U_d1_np, U_d2_np)
                    x1 = x - np.dot(Dfr_inv, fr)
                    if (abs(x1 - x) < 1e-12).all(): #1e-12
                        break
                    k += 1
                    np.copyto(x, x1)
                except:
                    print(parent.pos)
                    print(d1.pos)
                    print(d2.pos)
                    
                    plt.figure()
                    plt.plot(parent.U0[0,:], label = str(parent.pos))
                    plt.legend()
                   
                    plt.figure()
                    plt.plot(d1.U0[0,:], label = str(d1.pos))
                    plt.legend()
                    
                    plt.figure()
                    plt.plot(d2.U0[0,:], label = str(d2.pos))
                    plt.legend()
                    
                    print(d2.U0[0,0:20])
                    
                    sys.exit()
                #################Debugging##################
            return x
                    
        
        @staticmethod
        def cfl_condition(artery, dt, t):
            """
            Tests whether the CFL condition
            
            dt/dx < u + c,
            
            where u is velocity (q/a) and c is the wave speed, is fulfilled.
            
            :param artery: Artery object for which the CFL condition is tested.
            :param dt: Time step size.
            """
            a = artery.U0[0,1]
            c = artery.wave_speed(a)
            u = artery.U0[1,1] / a
            v = [u + c, u - c]
            left = dt/artery.dx
            right = 1/np.absolute(v)
            try:
                cfl = False if (left > right).any() else True
            except ValueError:
                raise ValueError("CFL condition not fulfilled at time %e. Reduce \
    time step size." % (t))
                sys.exit(1) 
            return cfl
            
        #Needs to be modified to get daughters based on my INDEXING    
        def get_daughters(self, parent):
            
            p_index = parent.pos
            d_index = self.dataframe.at[p_index,'End Condition']
        
            return self.arteries[d_index[0]], self.arteries[d_index[1]]
            
            
        @staticmethod
        def _printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
            formatStr       = "{0:." + str(decimals) + "f}"
            percents        = formatStr.format(100 * (iteration / float(total)))
            filledLength    = int(round(barLength * iteration / float(total)))
            bar             = '█' * filledLength + '-' * (barLength - filledLength)
            sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
            if iteration == total:
                sys.stdout.write('\n')
            sys.stdout.flush()
            
            
        def print_status(self):
            """
            Prints a status bar to the terminal in 2% increments.
            """
            it = 2
            if self.t % (self.tf/(100/it)) < self.dt:
                ArteryNetwork._printProgress(self.progress, 100,
                        prefix = 'Progress:', suffix = 'Complete', barLength = 50)
                self.progress += it        
                
                
        def redimensionalise(self, rc, qc):
            """
            Converts dimensionless solutions to dimensional solutions.
            
            :param rc: Characteristic radius.
            :param qc: Characteristic flux.
            """
            for artery in self.arteries:
                artery.P = (artery.P * self.rho*qc**2 / rc**4) / 1333.22365
                artery.U[0,:,:] = artery.U[0,:,:] * rc**2  
                artery.U[1,:,:] = artery.U[1,:,:] * qc
                
        
        def solve(self, q_in, out_bc, out_args):
            """
            ArteryNetwork solver. Assigns boundary conditions to Artery object in the arterial tree and calls their solvers.
            
            :param q_in: Function for flux at the inlet.
            :param out_bc: Choice of outlet boundary conditions. '3wk' for windkessel, 'p' for constant pressure.
            :param out_args: Iterable containing outlet boundary condition parameters.
            """
            
            tr = np.linspace(self.tf-self.T, self.tf, self.ntr)
            i = 0
            it = 0
            
            self.print_status()
            self.timestep()       
            bc_in = np.zeros((len(self.arteries), 2))
            
            while self.t < self.tf:
                save = False  
                
                if i < self.ntr and (abs(tr[i]-self.t) < self.dtr or 
                                                    self.t >= self.tf-self.dt):
                    save = True
                    i += 1
                    
                for artery in self.arteries:
                    theta = self.dt/artery.dx
                    gamma = self.dt/2
                    lw = LaxWendroff(theta, gamma, artery.nx)
                    
                    
                    ############Troubleshooting##############
                    # if artery.pos in [0,1,2]:
                    #    plt.figure()
                    #    plt.plot(artery.U0[0,:], label = str(artery.pos))
                    #    plt.legend()
                    #    plt.title('Before')
                    #    print(artery.U0[0,:])
                                    
                   ############################################
                    index = artery.pos
                    end_condition = self.dataframe.at[index,'End Condition']
                       
                    #Decides if artery requires us to find it's daughters and then finds them 
                    if end_condition != 'ST':
                    
                        d1, d2 = self.get_daughters(artery)
                        x_out = ArteryNetwork.bifurcation(artery, d1, d2, self.dt)
                        U_out = np.array([x_out[9], x_out[0]])
                        bc_in[d2.pos] = np.array([x_out[15], x_out[6]])
                        bc_in[d1.pos] = np.array([x_out[12], x_out[3]])
                        
                    
                    #If artery is the first one
                    if artery.pos == 0:
                            # inlet boundary condition
                        if self.T > 0:
                            in_t = periodic(self.t, self.T)
                        else:
                            in_t = self.t
                        U_in = ArteryNetwork.inlet_bc(artery, q_in, in_t, self.dt)
                        
                    else:
                        U_in = bc_in[artery.pos]   
                    #Here based on depth determines the wk or constant pressure end condition EDIT HERE
                    
                    if end_condition == 'ST':
                        # outlet boundary condition
                        if out_bc == '3wk':
                            U_out = ArteryNetwork.outlet_wk3(artery, self.dt, *out_args)
                        if out_bc == 'p':
                            U_out = ArteryNetwork.outlet_p(artery, self.dt, *out_args)
                        elif out_bc == 'ST':
                            
                            artery.Qnk[:] = np.concatenate(([artery.U0[1,-1]],artery.Qnk[1:]))       
                            U_out = ArteryNetwork.outlet_st(artery, self.dt, self.t)
                            
                    
                    artery.solve(lw, U_in, U_out, save, i-1)
                    
                    ############Troubleshooting##############
                    
                    if artery.pos in [228]:
                        #print(U_in)
                        plt.figure()
                        plt.plot(artery.U0[0,:], label = str(artery.pos))
                        plt.legend()
                        plt.title('After')
                        #print(artery.U0[0,:])
                                    
                   ############################################
                    if ArteryNetwork.cfl_condition(artery, self.dt, self.t) == False:
                        raise ValueError(
                                "CFL condition not fulfilled at time %e. Reduce \
    time step size." % (self.t))
                        sys.exit(1)  
                print('iterations = ' + str(it))        
                self.timestep()
                self.print_status()
                it = it + 1
                #tt.toc()
                
        def dump_results(self, suffix, data_dir):
            """
            Writes solution of each artery into CSV files.
            
            :param suffix: Simulation identifier.
            :param data_dir: Directory to store CSV files in.
            """
            if not exists("%s/%s" % (data_dir, suffix)):
                makedirs("%s/%s" % (data_dir, suffix))
            for artery in self.arteries:
                artery.dump_results(suffix, data_dir)
                           
                           
        @property
        def depth(self):
            """
            Network depth
            """
            return self._depth
            
            
        @property
        def arteries(self):
            """
            List containing Artery objects
            """
            return self._arteries
            
            
        @property
        def dt(self):
            """
            Time step size
            """
            return self._dt
            
        
        @property        
        def tf(self):
            """
            Total simulation time
            """
            return self._tf
            
            
        @property
        def T(self):
            """
            Period length
            """
            return self._T
            
            
        @property
        def tc(self):
            """
            Number of periods in simulation
            """
            return self._tc
            
            
        @property
        def t(self):
            """
            Current time
            """
            return self._t
            
            
        @property
        def ntr(self):
            """
            Number of time steps in output
            """
            return self._ntr
            
            
        @property
        def dtr(self):
            """
            Time step size in output
            """
            return self._dtr
    
            
        @property
        def rho(self):
            """
            Density of blood
            """
            return self._rho
    
    
        @property
        def nu(self):
            """
            Viscosity of blood
            """
            return self._nu
            
    
        @property
        def p0(self):
            """
            Zero transmural pressure
            """
            return self._p0
        
        @property
        def dataframe(self):
            """
            Dataframe containing all vessel information
            """
            return self._dataframe
        
        @property
        def progress(self):
            """
            Simulation progress
            """
            return self._progress
            
        @progress.setter
        def progress(self, value): 
            self._progress = value
    
    #%% Lax Wendroff Object
    
    class LaxWendroff(object):
        """
        Class implementing Richtmyer's 2 step Lax-Wendroff method.
        """
        
        
        def __init__(self, theta, gamma, nx):
            """
            Constructor for LaxWendroff class.
            
            :param theta: factor for flux vector
            :param gamma: factor for source vector
            :param nx: number of spatial points
            """
            self._theta = theta
            self._gamma = gamma
            self._nx = nx
            
    
        def solve(self, U0, U_in, U_out, F, S):
            """
            Solver implementing Richtmyer's two-step Lax-Wendroff method [1,2].
            
            [1] R. D. Richtmyer. A Survey of Difference Methods for Non-Steady Fluid Dynamics. NCAR Technical Notes, 63(2), 1963.
            [2] R. J. LeVeque. Numerical Methods for Conservation Laws. Birkhauser Verlag, Basel, Switzerland, 2nd edition, 1992.
            
            :param U0: solution from previous time step
            :param U_in: inlet boundary condition
            :param U_out: outlet boundary condition
            :param F: flux function (see [2])
            :param S: source function (see [2])
            """
            
            # U0: previous timestep, U1 current timestep
            U1 = np.zeros((2,self.nx))
            # apply boundary conditions
            U1[:,0] = U_in
            U1[:,-1] = U_out
            # calculate half steps
            U_np_mp = (U0[:,2:]+U0[:,1:-1])/2 -\
                self.theta*(F(U0[:,2:], j=2, k=self.nx)-F(U0[:,1:-1], j=1, k=-1))/2 +\
                self.gamma*(S(U0[:,2:], j=2, k=self.nx)+S(U0[:,1:-1], j=1, k=-1))/2
            U_np_mm = (U0[:,1:-1]+U0[:,0:-2])/2 -\
                self.theta*(F(U0[:,1:-1], j=1, k=-1)-F(U0[:,0:-2], j=0, k=-2))/2 +\
                self.gamma*(S(U0[:,1:-1], j=1, k=-1)+S(U0[:,0:-2], j=0, k=-2))/2
            # calculate full step
            U1[:,1:-1] = U0[:,1:-1] -\
                self.theta*(F(U_np_mp, j=1, k=-1)-F(U_np_mm, j=1, k=-1)) +\
                self.gamma*(S(U_np_mp, j=1, k=-1)+S(U_np_mm, j=1, k=-1))
            return U1
            
            
        @property   
        def theta(self):
            """
            dt/dx
            """
            return self._theta
            
        @property   
        def gamma(self):
            """
            dt/2
            """
            return self._gamma
            
        @property   
        def nx(self):
            """
            Number of spatial steps
            """        
            return self._nx
    
    
    
    
    
    #%% Define parameters
    rc = 1  #cm
    qc = 10 #cm3/s
    rho = 1.055 #g/cm3
    nu = 0.049 #cm2/s

    T = 1 #s
    tc = 1 #Normally 4 #s
    dt = 1e-6 #normally 1e-5 #s
    dx = 0.01 #normally 0.1 #cm 
    
    q_in = inlet(qc, rc, 'AorticFlow_inlet.csv')
    
    Re = qc/(nu*rc) 
    T = T * qc / rc**3 # time of one cycle
    tc = tc # number of cycles to simulate
    dt = dt * qc / rc**3 # time step size
    ntr = 50 # number of time steps to be stored
    dx = dx / rc # spatial step size
    nu = nu*rc/qc # viscosity
    
    kc = rho*qc**2/rc**4
    k1 = 2.0e7 #g/s2 cm
    k2 = -22.53 # 1/cm 
    k3 = 8.65e5 #g/s2 cm
    
    k = (k1/kc, k2*rc, k3/kc) # elasticity model parameters (Eh/r) 
    out_args =[0]
    out_bc = 'ST'
    p0 =((85 * 1333.22365) * rc**4/(rho*qc**2)) # zero transmural pressure intial 85 *
      
    dataframe = vessel_df
    lrr = lrr_values
    r_min =0.003 #0.01< 0.001
    Z_term = 0 #Terminal Impedance 8
    
    #%% Run simulation
    
    #Need dataframe size
    row , col = dataframe.shape 
    intial_values = np.zeros(row)
    # intial_values[0:3] = 15
    # intial_values[3:26] = 10
    # intial_values[26:59] =2
    # intial_values[59:300] =2
    # intial_values[300:] = 0.5
    
    
    an = ArteryNetwork(rho, nu, p0, ntr, Re, k, dataframe, Z_term, r_min, lrr, rc, mirror_dict)
    
    
    an.mesh(dx)
    an.set_time(dt, T, tc)
    an.initial_conditions(intial_values/qc, dataframe,mirror_dict, rc,qc)
    
    

    
    # run solver
    an.solve(q_in, out_bc, out_args)
    
    
    # redimensionalise
    an.redimensionalise(rc, qc)
    
    file_name = 'Artery_Array'
    try:
        an.dump_results(file_name,'C:\\Users\\Cassidy.Northway\\RemoteGit')
    except:
        an.dump_results(file_name,'C:\\Users\\cbnor\\Documents\\Full Body Flow Model Project') 

