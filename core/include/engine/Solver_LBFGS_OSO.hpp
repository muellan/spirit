#pragma once
#ifndef SPIRIT_CORE_ENGINE_SOLVER_LBFGS_OSO_HPP
#define SPIRIT_CORE_ENGINE_SOLVER_LBFGS_OSO_HPP

#include <utility/Constants.hpp>
// #include <utility/Exception.hpp>

#include <algorithm>

using namespace Utility;

template<>
inline void Method_Solver<Solver::LBFGS_OSO>::Initialize()
{
    this->n_lbfgs_memory = 3; // how many previous iterations are stored in the memory
    this->delta_a        = std::vector<field<vectorfield>>(
        this->noi, field<vectorfield>( this->n_lbfgs_memory, vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->delta_grad = std::vector<field<vectorfield>>(
        this->noi, field<vectorfield>( this->n_lbfgs_memory, vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->rho            = scalarfield( this->n_lbfgs_memory, 0 );
    this->alpha          = scalarfield( this->n_lbfgs_memory, 0 );
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->searchdir      = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad_pr        = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->q_vec          = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->local_iter     = 0;
    this->maxmove        = Constants::Pi / 200.0;
};

/*
    Implemented according to Aleksei Ivanov's paper: https://arxiv.org/abs/1904.02669
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121).
*/

template<>
inline void Method_Solver<Solver::LBFGS_OSO>::Iteration()
{
    using namespace Execution;

    // update forces which are -dE/ds
    this->Calculate_Force( this->configurations, this->forces );
    // calculate gradients for OSO
    for( int img = 0; img < this->noi; img++ )
    {
        auto s  = const_view_of( *this->configurations[img] );
        auto fv = view_of( this->forces_virtual[img] );
        auto f  = view_of( this->forces[img] );

        auto task = schedule( exec_context )
                    | stdexec::bulk( s.size(), [=]( std::size_t i ) { fv[i] = s[i].cross( f[i] ); } )
                    | Solver_Kernels::oso_calc_gradients_async( this->grad[img], s, f );

        stdexec::sync_wait( std::move( task ) ).value();
    }

    // calculate search direction
    Solver_Kernels::lbfgs_get_searchdir(
        this->local_iter, this->rho, this->alpha, this->q_vec, this->searchdir, this->delta_a, this->delta_grad,
        this->grad, this->grad_pr, this->n_lbfgs_memory, maxmove );

    // TODO integrate into one stdexec workflow?
    // Scale direction
    scalar scaling = 1;
    for( int img = 0; img < noi; img++ )
    {
        scaling = std::min( Solver_Kernels::maximum_rotation( searchdir[img], maxmove ), scaling );
    }

    for( int img = 0; img < noi; img++ )
    {
        Vectormath::scale( searchdir[img], scaling );
    }

    // rotate spins
    Solver_Kernels::oso_rotate( exec_context, this->configurations, this->searchdir );
}

template<>
inline std::string Method_Solver<Solver::LBFGS_OSO>::SolverName()
{
    return "LBFGS_OSO";
}

template<>
inline std::string Method_Solver<Solver::LBFGS_OSO>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno using exponential transforms";
}

#endif
