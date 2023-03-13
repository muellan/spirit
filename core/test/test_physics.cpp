#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_LLG.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <data/State.hpp>

#include "catch.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <sstream>

using Catch::Matchers::WithinAbs;

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
constexpr scalar epsilon_2 = 1e-10;
constexpr scalar epsilon_3 = 1e-12;
constexpr scalar epsilon_4 = 1e-12;
constexpr scalar epsilon_5 = 1e-6;
constexpr scalar epsilon_6 = 1e-7;
#else
constexpr scalar epsilon_2 = 1e-2;
constexpr scalar epsilon_3 = 1e-3;
constexpr scalar epsilon_4 = 1e-4;
constexpr scalar epsilon_5 = 1e-5;
constexpr scalar epsilon_6 = 1e-6;
#endif

TEST_CASE( "Dynamics solvers should follow Larmor precession", "[physics]" )
{
    constexpr auto input_file = "core/test/input/physics_larmor.cfg";
    std::vector<int> solvers{
        Solver_Heun,
        Solver_Depondt,
        Solver_SIB,
        Solver_RungeKutta4,
    };

    // Create State
    auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );

    // Set up one the initial direction of the spin
    scalar init_direction[3] = { 1., 0., 0. };            // vec parallel to x-axis
    Configuration_Domain( state.get(), init_direction ); // set spin parallel to x-axis

    // Assure that the initial direction is set
    // (note: this pointer will stay valid throughout this test)
    const auto * direction = System_Get_Spin_Directions( state.get() );
    REQUIRE( direction[0] == 1 );
    REQUIRE( direction[1] == 0 );
    REQUIRE( direction[2] == 0 );

    // Make sure that mu_s is the same as the one define in input file
    scalar mu_s{};
    Geometry_Get_mu_s( state.get(), &mu_s );
    REQUIRE( mu_s == 2 );

    // Get the magnitude of the magnetic field (it has only z-axis component)
    scalar B_mag{};
    scalar normal[3]{ 0, 0, 1 };
    Hamiltonian_Get_Field( state.get(), &B_mag, normal );

    // Get time step of method
    scalar damping = 0.3;
    scalar tstep    = Parameters_LLG_Get_Time_Step( state.get() );
    Parameters_LLG_Set_Damping( state.get(), damping );

    scalar dtg = tstep * Constants_gamma() / ( 1.0 + damping * damping );

    for( auto solver : solvers )
    {
        // Set spin parallel to x-axis
        Configuration_Domain( state.get(), init_direction );
        Simulation_LLG_Start( state.get(), solver, -1, -1, true );

        for( int i = 0; i < 100; i++ )
        {
            INFO( "Solver " << solver << " failed spin trajectory test at iteration " << i );

            // A single iteration
            Simulation_SingleShot( state.get() );

            // Expected spin orientation
            scalar phi_expected = dtg * ( i + 1 ) * B_mag;
            scalar sz_expected  = std::tanh( damping * dtg * ( i + 1 ) * B_mag );
            scalar rxy_expected = std::sqrt( 1 - sz_expected * sz_expected );
            scalar sx_expected  = std::cos( phi_expected ) * rxy_expected;

            // TODO: why is precision so low for Heun and SIB solvers? Other solvers manage ~1e-10
            REQUIRE_THAT( direction[0], WithinAbs( sx_expected, epsilon_5 ) );
            REQUIRE_THAT( direction[2], WithinAbs( sz_expected, 1e-6 ) );
        }

        Simulation_Stop( state.get() );
    }
}

TEST_CASE( "Finite difference and regular Hamiltonian should match", "[physics]" )
{
    // Hamiltonians to be tested
    std::vector<const char *> hamiltonian_input_files{
        "core/test/input/fd_pairs.cfg",
        // "core/test/input/fd_neighbours",
        // "core/test/input/fd_gaussian.cfg",
    };

    for( const auto * input_file : hamiltonian_input_files )
    {
        INFO( " Testing " << input_file );

        auto state = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
        Configuration_Random( state.get() );
        const auto & spins = *state->active_image->spins;

        // Compare gradients
        auto grad    = vectorfield( state->nos );
        auto grad_fd = vectorfield( state->nos );
        state->active_image->hamiltonian->Gradient_FD( spins, grad_fd );
        state->active_image->hamiltonian->Gradient( spins, grad );
        for( int i = 0; i < state->nos; i++ )
        {
            INFO( "i = " << i << ", epsilon = " << epsilon_2 << "\n" );
            INFO( "Gradient (FD) = " << grad_fd[i].transpose() << "\n" );
            INFO( "Gradient      = " << grad[i].transpose() << "\n" );
            REQUIRE( grad_fd[i].isApprox( grad[i], epsilon_2 ) );
        }

        // Compare Hessians
        auto hessian    = MatrixX( 3 * state->nos, 3 * state->nos );
        auto hessian_fd = MatrixX( 3 * state->nos, 3 * state->nos );
        state->active_image->hamiltonian->Hessian_FD( spins, hessian_fd );
        state->active_image->hamiltonian->Hessian( spins, hessian );
        INFO( "epsilon = " << epsilon_3 << "\n" );
        INFO( "Hessian (FD) =\n" << hessian_fd << "\n" );
        INFO( "Hessian      =\n" << hessian << "\n" );
        REQUIRE( hessian_fd.isApprox( hessian, epsilon_3 ) );
    }
}

TEST_CASE( "Dipole-Dipole Interaction", "[physics]" )
{
    // Config file where only DDI is enabled
    constexpr auto input_file = "core/test/input/physics_ddi.cfg";
    auto state                = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );

    Configuration_Random( state.get() );
    auto & spins = *state->active_image->spins;

    // FFT gradient and energy
    auto grad_fft = vectorfield( state->nos );
    state->active_image->hamiltonian->Gradient( spins, grad_fft );
    auto energy_fft = state->active_image->hamiltonian->Energy( spins );

    // Direct (cutoff) gradient and energy
    auto n_periodic_images = std::vector<int>{ 4, 4, 4 };
    Hamiltonian_Set_DDI( state.get(), SPIRIT_DDI_METHOD_CUTOFF, n_periodic_images.data(), -1 );
    auto grad_direct = vectorfield( state->nos );
    state->active_image->hamiltonian->Gradient( spins, grad_direct );
    auto energy_direct = state->active_image->hamiltonian->Energy( spins );

    // Compare gradients
    for( int i = 0; i < state->nos; i++ )
    {
        INFO( "Failed DDI-Gradient comparison at i = " << i << ", epsilon = " << epsilon_4 << "\n" );
        INFO( "Gradient (FFT)    = " << grad_fft[i].transpose() << "\n" );
        INFO( "Gradient (Direct) = " << grad_direct[i].transpose() << "\n" );
        REQUIRE( grad_fft[i].isApprox(
            grad_direct[i], epsilon_4 ) ); // Seems this is a relative test, not an absolute error margin
    }

    // Compare energies
    INFO( "Failed energy comparison test! epsilon = " << epsilon_6 );
    INFO( "Energy (Direct) = " << energy_direct << "\n" );
    INFO( "Energy (FFT)    = " << energy_fft << "\n" );
    REQUIRE_THAT( energy_fft, WithinAbs( energy_direct, epsilon_6 ) );
}
