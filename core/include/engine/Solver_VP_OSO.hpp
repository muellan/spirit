template<>
inline void Method_Solver<Solver::VP_OSO>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->velocities = std::vector<vectorfield>( this->noi, vectorfield( this->nos, Vector3::Zero() ) ); // [noi][nos]
    this->velocities_previous = velocities;                                                              // [noi][nos]
    this->forces_previous     = velocities;                                                              // [noi][nos]
    this->grad                = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad_pr             = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->projection          = std::vector<scalar>( this->noi, 0 ); // [noi]
    this->force_norm2         = std::vector<scalar>( this->noi, 0 ); // [noi]
    this->searchdir           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
}

/*
    Template instantiation of the Simulation class for use with the VP Solver.
        The velocity projection method is often efficient for direct minimization,
        but deals poorly with quickly varying fields or stochastic noise.
    Paper: P. F. Bessarab et al., Method for finding mechanism and activation energy
           of magnetic transitions, applied to skyrmion and antivortex annihilation,
           Comp. Phys. Comm. 196, 335 (2015).

    Instead of the cartesian update scheme with re-normalization, this implementation uses the orthogonal spin
   optimization scheme, described by A. Ivanov in https://arxiv.org/abs/1904.02669.
*/

template<>
inline void Method_Solver<Solver::VP_OSO>::Iteration()
{
    using namespace Execution;

    scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for( int img = 0; img < noi; ++img )
    {
        auto g    = const_view_of( grad[img] );
        auto v    = const_view_of( velocities[img] );
        auto g_pr = view_of( grad_pr[img] );
        auto v_pr = view_of( velocities_previous[img] );

        for_each_index(
            exec_context, nos,
            [=]( std::size_t i )
            {
                g_pr[i] = g[i];
                v_pr[i] = v[i];
            } );
    }

    // Get the forces on the configurations
    this->Calculate_Force( configurations, forces );
    this->Calculate_Force_Virtual( configurations, forces, forces_virtual );

    for( int img = 0; img < this->noi; img++ )
    {
        auto image  = const_view_of( *this->configurations[img] );
        auto forces = const_view_of( this->forces[img] );
        auto grad   = view_of( this->grad[img] );

        // clang-format off
        auto task = schedule(exec_context)
        |   Solver_Kernels::oso_calc_gradients_async( grad, image, forces )
        |   stdexec::bulk(grad.size(), [=](std::size_t i){
                grad[i] *= -1.0;
            });
        // clang-format on

        stdexec::sync_wait( std::move( task ) ).value();
    }

    for( int img = 0; img < noi; ++img )
    {
        auto g      = const_view_of( this->grad[img] );
        auto g_pr   = const_view_of( this->grad_pr[img] );
        auto v      = view_of( velocities[img] );
        auto m_temp = this->m;

        // Calculate the new velocity
        for_each_index( exec_context, nos, [=]( std::size_t i ) { v[i] += 0.5 / m_temp * ( g_pr[i] + g[i] ); } );

        // Get the projection of the velocity on the force
        // TODO integrate into one stdexec workflow
        projection[img]  = Vectormath::dot( velocities[img], this->grad[img] );
        force_norm2[img] = Vectormath::dot( this->grad[img], this->grad[img] );
    }

    for( int img = 0; img < noi; ++img )
    {
        projection_full += projection[img];
        force_norm2_full += force_norm2[img];
    }

    for( int img = 0; img < noi; ++img )
    {
        auto v      = view_of( this->velocities[img] );
        auto sd     = view_of( this->searchdir[img] );
        auto g      = view_of( this->grad[img] );
        auto m_temp = this->m;

        scalar dt    = this->systems[img]->llg_parameters->dt;
        scalar ratio = projection_full / force_norm2_full;

        // Calculate the projected velocity

        if( projection_full <= 0 )
        {
            generate_indexed( exec_context, v, [=]( std::size_t i ) { return Vector3{ 0, 0, 0 }; } );
        }
        else
        {
            generate_indexed( exec_context, v, [=]( std::size_t i ) { return g[i] * ratio; } );
        }
        generate_indexed( exec_context, sd, [=]( std::size_t i ) { return dt * v[i] + 0.5 / m_temp * dt * g[i]; } );

        // TODO: use this insted, but as of now, doesn't work on the GPU
        // auto task = schedule(exec_context)
        // |   if_then_else(
        //         [=]{ return projection_full <= 0; },
        //         stdexec::bulk(v.size(), [=](std::size_t i){
        //             v[i] = Vector3 { 0, 0, 0 };
        //         }),
        //         stdexec::bulk(v.size(), [=](std::size_t i){
        //             v[i] = g[i] * ratio;
        //         })
        //     )
        // |   stdexec::bulk(conf.size(), [=](std::size_t i){
        //         sd[i] = dt * v[i] + 0.5 / m_temp * dt * g[i];;
        //     });
        //
        // stdexec::sync_wait(std::move(task)).value();
    }

    Solver_Kernels::oso_rotate( exec_context, this->configurations, this->searchdir );
}

template<>
inline std::string Method_Solver<Solver::VP_OSO>::SolverName()
{
    return "VP_OSO";
}

template<>
inline std::string Method_Solver<Solver::VP_OSO>::SolverFullName()
{
    return "Velocity Projection using exponential transforms";
}
