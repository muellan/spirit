#include <engine/Backend_par.hpp>

template<>
inline void Method_Solver<Solver::VP>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->velocities = std::vector<vectorfield>( this->noi, vectorfield( this->nos, Vector3::Zero() ) ); // [noi][nos]
    this->velocities_previous = velocities;                                                              // [noi][nos]
    this->forces_previous     = velocities;                                                              // [noi][nos]
    this->projection          = std::vector<scalar>( this->noi, 0 );                                     // [noi]
    this->force_norm2         = std::vector<scalar>( this->noi, 0 );                                     // [noi]
}

/*
    Template instantiation of the Simulation class for use with the VP Solver.
        The velocity projection method is often efficient for direct minimization,
        but deals poorly with quickly varying fields or stochastic noise.
    Paper: P. F. Bessarab et al., Method for finding mechanism and activation energy
           of magnetic transitions, applied to skyrmion and antivortex annihilation,
           Comp. Phys. Comm. 196, 335 (2015).
*/
template<>
inline void Method_Solver<Solver::VP>::Iteration()
{
    using namespace Execution;

    scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for( int img = 0; img < noi; ++img )
    {
        auto f    = const_view_of( forces[img] );
        auto v    = const_view_of( velocities[img] );
        auto f_pr = view_of( forces_previous[img] );
        auto v_pr = view_of( velocities_previous[img] );

        for_each_index(
            exec_context, f.size(),
            [=]( std::size_t i )
            {
                f_pr[i] = f[i];
                v_pr[i] = v[i];
            } );
    }

    // Get the forces on the configurations
    this->Calculate_Force( configurations, forces );
    this->Calculate_Force_Virtual( configurations, forces, forces_virtual );

    for( int img = 0; img < noi; ++img )
    {
        auto v      = view_of( velocities[img] );
        auto f      = const_view_of( forces[img] );
        auto f_pr   = const_view_of( forces_previous[img] );
        auto m_temp = this->m;

        // TODO coalesce steps in one stdexec task
        // Calculate the new velocity
        for_each_index( exec_context, f.size(), [=]( std::size_t i ) { v[i] += 0.5 / m_temp * ( f_pr[i] + f[i] ); } );

        // Get the projection of the velocity on the force
        // TODO integrate into one stdexec workflow
        projection[img]  = Vectormath::dot( velocities[img], forces[img] );
        force_norm2[img] = Vectormath::dot( forces[img], forces[img] );
    }

    for( int img = 0; img < noi; ++img )
    {
        projection_full += projection[img];
        force_norm2_full += force_norm2[img];
    }

    for( int img = 0; img < noi; ++img )
    {
        auto f         = const_view_of( forces[img] );
        auto v         = view_of( velocities[img] );
        auto conf      = view_of( *( configurations[img] ) );
        auto conf_temp = view_of( *( configurations_temp[img] ) );

        scalar dt    = this->systems[img]->llg_parameters->dt;
        scalar ratio = projection_full / force_norm2_full;
        auto m_temp  = this->m;

        // Calculate the projected velocity

        if( projection_full <= 0 )
        {
            generate_indexed( exec_context, v, [=]( std::size_t i ) { return Vector3{ 0, 0, 0 }; } );
        }
        else
        {
            generate_indexed( exec_context, v, [=]( std::size_t i ) { return f[i] * ratio; } );
        }
        for_each_index(
            exec_context, conf.size(),
            [=]( std::size_t i )
            {
                conf_temp[i] = conf[i] + dt * v[i] + 0.5 / m_temp * dt * f[i];
                conf[i]      = conf_temp[i].normalized();
            } );

        // TODO: use this insted, but as of now, doesn't work on the GPU
        // auto task = schedule(exec_context)
        // |   if_then_else(
        //         [=]{ return projection_full <= 0; },
        //         stdexec::bulk(v.size(), [=](std::size_t i){
        //             v[i] = Vector3 { 0, 0, 0 };
        //         }),
        //         stdexec::bulk(v.size(), [=](std::size_t i){
        //             v[i] = f[i] * ratio;
        //         })
        //     )
        // |   stdexec::bulk(conf.size(), [=](std::size_t i){
        //         conf_temp[i] = conf[i] + dt * v[i] + 0.5 / m_temp * dt * f[i];
        //         conf[i]      = conf_temp[i].normalized();
        //     });
        //
        // stdexec::sync_wait(std::move(task)).value();
    }
}

template<>
inline std::string Method_Solver<Solver::VP>::SolverName()
{
    return "VP";
}

template<>
inline std::string Method_Solver<Solver::VP>::SolverFullName()
{
    return "Velocity Projection";
}
