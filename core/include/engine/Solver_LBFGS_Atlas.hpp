#pragma once
#ifndef SPIRIT_CORE_ENGINE_SOLVER_LBFGS_ATLAS_HPP
#define SPIRIT_CORE_ENGINE_SOLVER_LBFGS_ATLAS_HPP

#include <utility/Constants.hpp>
// #include <utility/Exception.hpp>
#include <engine/Backend_par.hpp>

#include <algorithm>

using namespace Utility;

[[nodiscard]] inline auto atlas_calc_gradients_async(
    const_vectorfield_view s, // spins
    const_vectorfield_view f, // forces
    const_scalarfield_view a3, vector2field_view residuals )
{
    return stdexec::bulk(
        s.size(),
        [=](std::size_t i)
        {
            auto J00 = s[i][1] * s[i][1] + s[i][2] * ( s[i][2] + a3[i] );
            auto J10 = -s[i][0] * s[i][1];
            auto J01 = -s[i][0] * s[i][1];
            auto J11 = s[i][0] * s[i][0] + s[i][2] * ( s[i][2] + a3[i] );
            auto J02 = -s[i][0] * ( s[i][2] + a3[i] );
            auto J12 = -s[i][1] * ( s[i][2] + a3[i] );

            residuals[i][0] = -( J00 * f[i][0] + J01 * f[i][1] + J02 * f[i][2] );
            residuals[i][1] = -( J10 * f[i][0] + J11 * f[i][1] + J12 * f[i][2] );
        } );
}

template<>
inline void Method_Solver<Solver::LBFGS_Atlas>::Initialize()
{
    this->n_lbfgs_memory = 3; // how many updates the solver tracks to estimate the hessian
    this->atlas_updates  = std::vector<field<vector2field>>(
        this->noi, field<vector2field>( this->n_lbfgs_memory, vector2field( this->nos, { 0, 0 } ) ) );
    this->grad_atlas_updates = std::vector<field<vector2field>>(
        this->noi, field<vector2field>( this->n_lbfgs_memory, vector2field( this->nos, { 0, 0 } ) ) );
    this->rho                  = scalarfield( this->n_lbfgs_memory, 0 );
    this->alpha                = scalarfield( this->n_lbfgs_memory, 0 );
    this->forces               = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual       = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->atlas_coords3        = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 1 ) );
    this->atlas_directions     = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_residuals      = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_residuals_last = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->atlas_q_vec          = std::vector<vector2field>( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->maxmove              = 0.05;
    this->local_iter           = 0;

    for( int img = 0; img < this->noi; img++ )
    {
        // Choose atlas3 coordinates
        for( int i = 0; i < this->nos; i++ )
        {
            this->atlas_coords3[img][i] = ( *this->configurations[img] )[i][2] > 0 ? 1.0 : -1.0;
            // Solver_Kernels::ncg_spins_to_atlas( *this->configurations[i], this->atlas_coords[i], this->atlas_coords3[i] );
        }
    }
}

/*
    Stereographic coordinate system implemented according to an idea of F. Rybakov
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121)
*/
// clang-format off
template<>
inline void Method_Solver<Solver::LBFGS_Atlas>::Iteration()
{
    using namespace Execution;

    int noi = configurations.size();
    int nos = ( *configurations[0] ).size();

    // Current force
    this->Calculate_Force( this->configurations, this->forces );

    for( int img = 0; img < this->noi; img++ )
    {
        auto image    = const_view_of( *this->configurations[img] );
        auto a3       = const_view_of( this->atlas_coords3[img] );
        auto f        = const_view_of( this->forces[img] );
        auto s        = const_view_of( image );
        auto fv       = view_of( this->forces_virtual[img] );
        auto grad_ref = view_of( this->atlas_residuals[img] );

        auto task = schedule( exec_context )
        |   stdexec::when_all(
                stdexec::bulk( this->nos, [=]( std::size_t i )
                { 
                    fv[i] = s[i].cross( f[i] ); 
                } )
                | stdexec::then([]{})
                ,
                atlas_calc_gradients_async( image, f, a3, grad_ref )
                | stdexec::then([]{})
            );

        stdexec::sync_wait( std::move( task ) ).value();
    }

    // Calculate search direction
    Solver_Kernels::lbfgs_get_searchdir(
        this->local_iter, this->rho, this->alpha, this->atlas_q_vec, this->atlas_directions, this->atlas_updates,
        this->grad_atlas_updates, this->atlas_residuals, this->atlas_residuals_last, this->n_lbfgs_memory, maxmove );

    scalar a_norm_rms = 0;
    // Scale by averaging
    for( int img = 0; img < noi; img++ )
    {
        a_norm_rms = std::max(
            a_norm_rms,
            scalar( sqrt(
                Backend::par::reduce(
                    this->atlas_directions[img], [] SPIRIT_LAMBDA( const Vector2 & v ) { return v.squaredNorm(); } )
                / nos ) ) );
    }
    scalar scaling = ( a_norm_rms > maxmove ) ? maxmove / a_norm_rms : 1.0;

    for( int img = 0; img < noi; img++ )
    {
        auto d = atlas_directions[img].data();
        Backend::par::apply( nos, [scaling, d] SPIRIT_LAMBDA( int idx ) { d[idx] *= scaling; } );
    }

    // Rotate spins
    Solver_Kernels::atlas_rotate( this->configurations, this->atlas_coords3, this->atlas_directions );

    if( Solver_Kernels::ncg_atlas_check_coordinates( this->configurations, this->atlas_coords3, -0.6 ) )
    {
        Solver_Kernels::lbfgs_atlas_transform_direction(
            this->configurations, this->atlas_coords3, this->atlas_updates, this->grad_atlas_updates,
            this->atlas_directions, this->atlas_residuals_last, this->rho );
    }
}
// clang-format on

template<>
inline std::string Method_Solver<Solver::LBFGS_Atlas>::SolverName()
{
    return "LBFGS_Atlas";
}

template<>
inline std::string Method_Solver<Solver::LBFGS_Atlas>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno using stereographic atlas";
}

#endif
