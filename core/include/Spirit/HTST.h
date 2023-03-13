#pragma once
#ifndef SPIRIT_CORE_HTST_H
#define SPIRIT_CORE_HTST_H
#include "Spirit_Defines.h"

#include "DLL_Define_Export.h"

struct State;

/*
HTST
====================================================================

```C
#include "Spirit/HTST.h"
```

Harmonic transition state theory.

Note that `HTST_Calculate` needs to be called before using any of the getter functions.
*/

/*
Calculates the HTST transition rate prefactor for the transition from a minimum over saddle point.

- `idx_image_minimum`: index of the local minimum in the chain
- `idx_image_sp`: index of the transition saddle point in the chain
- `n_eigenmodes_keep`: the number of energy eigenmodes to keep in memory (0 = none, negative value = all)
- `sparse`: when set to `true` the sparse version is used, which greatly improves speed and memory footprint,
            but does not evaluate the eigenvectors and all single eigenvalues.
            The sparse version should only be used in the abscence of DDI

Note: The Get_Eigenvalues/vectors functions only work after HTST_Calculate with sparse=false has been called.
Note: In the sparse version zero mode checking has not been implemented yet.
Note: that the method assumes you gave it correct images, where the
gradient is zero and which correspond to a minimum and a saddle point
respectively.
*/
PREFIX scalar HTST_Calculate(
    State * state, int idx_image_minimum, int idx_image_sp, int n_eigenmodes_keep = 0, bool sparse = false,
    int idx_chain = -1 );

/*
Retrieves a set of information from HTST:
- temperature_exponent: the exponent of the temperature-dependent prefactor
- me: sqrt(2pi k_B)^(N_0^M - N_0^SP)
- Omega_0: sqrt( prod(lambda^M) / prod(lambda^SP) )
- s: sqrt( prod(a^2 / lambda^SP) )
- volume_min: zero mode volume at the minimum
- volume_sp: zero mode volume at the saddle point
- prefactor_dynamical: the dynamical part of the rate prefactor
- prefactor: the total rate prefactor for the transition
*/
PREFIX void HTST_Get_Info(
    State * state, scalar * temperature_exponent, scalar * me, scalar * Omega_0, scalar * s, scalar * volume_min,
    scalar * volume_sp, scalar * prefactor_dynamical, scalar * prefactor, int * n_eigenmodes_keep,
    int idx_chain = -1 ) SUFFIX;

/*
Fetches HTST information eigenvalues at the min (array of length 2*NOS). Note: Only works after HTST_Calculate with
sparse=false has been called.
*/
PREFIX void HTST_Get_Eigenvalues_Min( State * state, scalar * eigenvalues_min, int idx_chain = -1 ) SUFFIX;

/*
Fetches HTST eigenvectors at the minimum (array of length 2*NOS*htst_info.n_eigenmodes_keep). Note: Only works after
HTST_Calculate with sparse=false has been called.
*/
PREFIX void HTST_Get_Eigenvectors_Min( State * state, scalar * eigenvectors_min, int idx_chain = -1 ) SUFFIX;

/*
Fetches HTST eigenvalues at the saddle point (array of length 2*NOS). Note: Only works after HTST_Calculate with
sparse=false has been called.
*/
PREFIX void HTST_Get_Eigenvalues_SP( State * state, scalar * eigenvalues_sp, int idx_chain = -1 ) SUFFIX;

/*
Fetches HTST eigenvectors at the saddle point (array of length 2*NOS*htst_info.n_eigenmodes_keep). Note: Only works
after HTST_Calculate with sparse=false has been called.
*/
PREFIX void HTST_Get_Eigenvectors_SP( State * state, scalar * eigenvectors_sp, int idx_chain = -1 ) SUFFIX;

/*
Fetches HTST information:
- velocities along the unstable mode (array of length 2*NOS). Note: Only works after HTST_Calculate with sparse=false
has been called.
*/
PREFIX void HTST_Get_Velocities( State * state, scalar * velocities, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif