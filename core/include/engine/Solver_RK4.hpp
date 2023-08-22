template<>
inline void Method_Solver<Solver::RungeKutta4>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->configurations_k1 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k1[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->configurations_k2 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k2[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->configurations_k3 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k3[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->temp1 = vectorfield( this->nos, { 0, 0, 0 } );
}

/*
    Template instantiation of the Simulation class for use with the 4th order Runge Kutta Solver.
*/
template<>
inline void Method_Solver<Solver::RungeKutta4>::Iteration()
{
    using namespace Execution;

    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    // Predictor for each image
    for( int img = 0; img < this->noi; ++img )
    {
        auto conf           = const_view_of( *this->configurations[img] );
        auto force          = const_view_of( this->forces_virtual[img] );
        auto k1             = view_of( *this->configurations_k1[img] );
        auto conf_predictor = view_of( *this->configurations_predictor[img] );

        for_each_index(
            exec_context, conf.size(),
            [=]( std::size_t i )
            {
                k1[i] = -1 * conf[i].cross( force[i] );
                // Predictor for k2
                conf_predictor[i] = conf[i] + 0.5 * k1[i];
                conf_predictor[i].normalize();
            } );
    }

    // Calculate_Force for the predictor
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Predictor for each image
    for( int img = 0; img < this->noi; ++img )
    {
        auto conf           = const_view_of( *this->configurations[img] );
        auto force          = const_view_of( this->forces_virtual_predictor[img] );
        auto k2             = view_of( *this->configurations_k2[img] );
        auto conf_predictor = view_of( *this->configurations_predictor[img] );

        for_each_index(
            exec_context, conf.size(),
            [=]( std::size_t i )
            {
                k2[i] = -1 * conf_predictor[i].cross( force[i] );
                // Predictor for k3
                conf_predictor[i] = conf[i] + 0.5 * k2[i];
                conf_predictor[i].normalize();
            } );
    }

    // Calculate_Force for the predictor (k3)
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Predictor for each image
    for( int img = 0; img < this->noi; ++img )
    {
        auto conf           = const_view_of( *this->configurations[img] );
        auto force          = const_view_of( this->forces_virtual_predictor[img] );
        auto k3             = view_of( *this->configurations_k3[img] );
        auto conf_predictor = view_of( *this->configurations_predictor[img] );

        for_each_index(
            exec_context, conf.size(),
            [=]( std::size_t i )
            {
                k3[i] = -1 * conf_predictor[i].cross( force[i] );
                // Predictor for k4
                conf_predictor[i] = conf[i] + k3[i];
                conf_predictor[i].normalize();
            } );
    }

    // Calculate_Force for the predictor (k4)
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Corrector step for each image
    for( int img = 0; img < this->noi; img++ )
    {
        auto force          = const_view_of( this->forces_virtual_predictor[img] );
        auto k1             = const_view_of( *this->configurations_k1[img] );
        auto k2             = const_view_of( *this->configurations_k2[img] );
        auto k3             = const_view_of( *this->configurations_k3[img] );
        auto conf_predictor = const_view_of( *this->configurations_predictor[img] );
        auto conf           = view_of( *this->configurations[img] );

        for_each_index(
            exec_context, conf.size(),
            [=]( std::size_t i )
            {
                auto k4 = -1 * conf_predictor[i].cross( force[i] );
                // 4th order Runge Kutta step
                conf[i] = conf[i] + ( 1.0 / 6.0 ) * k1[i] + ( 1.0 / 3.0 ) * k2[i] + ( 1.0 / 3.0 ) * k3[i]
                          + ( 1.0 / 6.0 ) * k4;
                conf[i].normalize();
            } );
    }
}

template<>
inline std::string Method_Solver<Solver::RungeKutta4>::SolverName()
{
    return "RK4";
}

template<>
inline std::string Method_Solver<Solver::RungeKutta4>::SolverFullName()
{
    return "Runge Kutta (4th order)";
}
