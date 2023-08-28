[[nodiscard]] inline auto
sib_transform_async( const_vectorfield_view spins, const_vectorfield_view force, vectorfield_view out )
{
    return stdexec::bulk(
        spins.size(),
        [=]( std::size_t i )
        {
            Vector3 e1, a2, A;
            scalar detAi;
            e1 = spins[i];
            A  = 0.5 * force[i];

            // 1/determinant(A)
            detAi = 1.0 / ( 1 + pow( A.norm(), 2.0 ) );

            // calculate equation witho the predictor?
            a2 = e1 - e1.cross( A );

            out[i][0]
                = ( a2[0] * ( A[0] * A[0] + 1 ) + a2[1] * ( A[0] * A[1] - A[2] ) + a2[2] * ( A[0] * A[2] + A[1] ) )
                  * detAi;
            out[i][1]
                = ( a2[0] * ( A[1] * A[0] + A[2] ) + a2[1] * ( A[1] * A[1] + 1 ) + a2[2] * ( A[1] * A[2] - A[0] ) )
                  * detAi;
            out[i][2]
                = ( a2[0] * ( A[2] * A[0] - A[1] ) + a2[1] * ( A[2] * A[1] + A[0] ) + a2[2] * ( A[2] * A[2] + 1 ) )
                  * detAi;
        } );
}

template<>
inline void Method_Solver<Solver::SIB>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );
}

/*
    Template instantiation of the Simulation class for use with the SIB Solver.
        The semi-implicit method B is an efficient midpoint solver.
    Paper: J. H. Mentink et al., Stable and fast semi-implicit integration of the stochastic
           Landau-Lifshitz equation, J. Phys. Condens. Matter 22, 176001 (2010).
*/
// clang-format off
template<>
inline void Method_Solver<Solver::SIB>::Iteration()
{
    using namespace Execution;

    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // First part of the step
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    for( int img = 0; img < this->noi; ++img )
    {
        auto image     = const_view_of(*this->systems[img]->spins);
        auto vforces   = const_view_of(forces_virtual[img]);
        auto predictor = view_of(*this->configurations_predictor[img]);

        auto task = schedule(exec_context)
        |   sib_transform_async(image, vforces, predictor)
        |   stdexec::bulk(predictor.size(), [=](std::size_t i)
            {
                predictor[i] += image[i];
                predictor[i] *= 0.5;
            });

        stdexec::sync_wait(std::move(task)).value();
    }

    // Second part of the step
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    for( int img = 0; img < this->noi; ++img )
    {
        auto image        = view_of(*this->systems[img]->spins);
        auto fv_predictor = const_view_of(forces_virtual_predictor[img]);

        auto task = schedule(exec_context)
        |   sib_transform_async(image, fv_predictor, image);

        stdexec::sync_wait(std::move(task)).value();
    }
}
// clang-format on

template<>
inline std::string Method_Solver<Solver::SIB>::SolverName()
{
    return "SIB";
}

template<>
inline std::string Method_Solver<Solver::SIB>::SolverFullName()
{
    return "Semi-implicit B";
}
