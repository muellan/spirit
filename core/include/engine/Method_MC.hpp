#pragma once
#ifndef SPIRIT_CORE_ENGINE_METHOD_MC_HPP
#define SPIRIT_CORE_ENGINE_METHOD_MC_HPP

#include "Spirit_Defines.h"
#include <engine/Method.hpp>
// #include <engine/Method_Solver.hpp>
#include <data/Spin_System.hpp>
// #include <data/Parameters_Method_MC.hpp>

#include <vector>

namespace Engine
{

/*
    The Monte Carlo method
*/
class Method_MC : public Method
{
public:
    // Constructor
    Method_MC( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain );

    // Method name as string
    std::string Name() override;

protected:
    virtual void Displace_Spin(int ispin, vectorfield & spins_new, std::uniform_real_distribution<scalar> & distribution, std::vector<int> & changed_indices, vectorfield & old_spins);
    virtual void Spin_Trial(int spin, vectorfield & spins_new, const vectorfield & spins_old, std::uniform_real_distribution<scalar> & distribution );
    virtual scalar Compute_Energy_Diff(const std::vector<int> & changed_indices, vectorfield & spins_new, const vectorfield & spins_old);
    virtual void Reject(const std::vector<int> & rejected_indices, const vectorfield & spins_old);

    std::shared_ptr<Data::Parameters_Method_MC> parameters_mc;
    // Vector to save the previous spin directions
    vectorfield spins_new;

    // Solver_Iteration represents one iteration of a certain Solver
    virtual void Iteration() override;

    // Metropolis iteration with adaptive cone radius
    void Metropolis( const vectorfield & spins_old, vectorfield & spins_new );

    // Save the current Step's Data: spins and energy
    void Save_Current( std::string starttime, int iteration, bool initial = false, bool final = false ) override;
    // A hook into the Method before an Iteration of the Solver
    void Hook_Pre_Iteration() override;
    // A hook into the Method after an Iteration of the Solver
    void Hook_Post_Iteration() override;

    // Sets iteration_allowed to false for the corresponding method
    void Initialize() override;
    // Sets iteration_allowed to false for the corresponding method
    void Finalize() override;

    // Log message blocks
    void Message_Start() override;
    void Message_Step() override;
    void Message_End() override;


    // Cosine of current cone angle
    scalar cone_angle;
    int n_rejected;
    scalar acceptance_ratio_current;
    int nos_nonvacant;

    // Random vector array
    vectorfield xi;
};

} // namespace Engine

#endif