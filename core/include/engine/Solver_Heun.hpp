template<>
inline void Method_Solver<Solver::Heun>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->temp1 = vectorfield( this->nos, { 0, 0, 0 } );
}

/*
    Template instantiation of the Simulation class for use with the Heun Solver.
        The Heun method is a basic solver for the PDE at hand here. It is sufficiently
        efficient and stable.
    This method is described for spin systems including thermal noise in
        U. Nowak, Thermally Activated Reversal in Magnetic Nanostructures,
        Annual Reviews of Computational Physics IX Chapter III (p 105) (2001)
*/
template<>
inline void Method_Solver<Solver::Heun>::Iteration()
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
        auto vforces        = const_view_of( forces_virtual[img] );
        auto conf           = const_view_of( *this->configurations[img] );
        auto conf_temp      = view_of( *this->configurations_temp[img] );
        auto conf_predictor = view_of( *this->configurations_predictor[img] );

        for_each_index(
            exec_context, conf.size(),
            [=]( std::size_t i )
            {
                conf_temp[i]      = -1 * conf[i].cross( vforces[i] );
                conf_predictor[i] = conf[i] + conf_temp[i];
                conf_predictor[i].normalize();
            } );
    }

    // Calculate_Force for the Corrector
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Corrector step for each image
    for( int img = 0; img < this->noi; img++ )
    {
        auto conf_predictor    = const_view_of( *this->configurations_predictor[img] );
        auto vforces_predictor = const_view_of( forces_virtual_predictor[img] );
        auto conf_temp         = const_view_of( *this->configurations_temp[img] );
        auto conf              = view_of( *this->configurations[img] );

        // Second step - Corrector
        for_each_index(
            exec_context, conf.size(),
            [=]( std::size_t i )
            {
                conf[i] = conf[i] + 0.5 * conf_temp[i] - 0.5 * conf_predictor[i].cross( vforces_predictor[i] );
                conf[i].normalize();
            } );
    }
}

template<>
inline std::string Method_Solver<Solver::Heun>::SolverName()
{
    return "Heun";
}

template<>
inline std::string Method_Solver<Solver::Heun>::SolverFullName()
{
    return "Heun";
}
