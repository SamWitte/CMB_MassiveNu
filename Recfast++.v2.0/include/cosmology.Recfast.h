//====================================================================================================================
// Author: Jens Chluba
// first implementation: June 2010
// Last modification: April 2018
// CITA, University of Toronto
// All rights reserved.
//====================================================================================================================
// 23.04.2018: added new structure for cosmology parameters
// 22.02.2018: added cosmic time for radiation-dominated era
// 08.06.2012: added option to use external hubble factor

#ifndef COSMOLOGY_RECFAST_H
#define COSMOLOGY_RECFAST_H

#include "Recfast++.h"

namespace Recfastpp_Cosmology
{
//====================================================================================================================
// Set variables of cosmology object and access them
//====================================================================================================================
void Set_Cosmology(const CosmoPars &CosmoInputs);
const CosmoPars& Get_Cosmology();

//====================================================================================================================
// Hubble-function in 1/sec
//====================================================================================================================
double H_z(double z);
double t_cos_rad(double z);   // cosmic time assuming pure radiation domination

//====================================================================================================================
// simple density parameters
//====================================================================================================================
double Omega_cdm();
double Omega_b();
double Omega_H();
double Y_p();
double f_He();

//====================================================================================================================
// allow setting Hubble function from outside of Recfast++ (added 08.06.2012)
//====================================================================================================================
void set_H_pointer(double (*Hz_p)(double));
void reset_H_pointer();

//====================================================================================================================
// hydrogen number density in m^-3
//====================================================================================================================
double NH(double z);

//====================================================================================================================
// CMB temperature at z
//====================================================================================================================
double TCMB(double z);

//====================================================================================================================
// compute total contribution from relativistic particles (photon & neutrinos)
//====================================================================================================================
double calc_Orel(double TCMB0, double Nnu, double h100);
}
#endif 

//====================================================================================================================
//====================================================================================================================