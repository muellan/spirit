#include <Eigen/Dense>

#include <engine/Backend_par.hpp>
#include <engine/Solver_Kernels.hpp>
#include <utility/Constants.hpp>
#include <utility/View.hpp>

using namespace Utility;
using Utility::Constants::Pi;

namespace Engine
{
namespace Solver_Kernels
{

void oso_rotate(
    Execution::Context exec_context, std::vector<std::shared_ptr<vectorfield>> & configurations,
    std::vector<vectorfield> & searchdir )
{
    const int noi = configurations.size();
    const int nos = configurations[0]->size();

    for( int img = 0; img < noi; ++img )
    {
        auto sd = const_view_of( searchdir[img] );
        auto s  = view_of( *configurations[img] );

        Execution::for_each_index(
            exec_context, nos,
            [=]( std::size_t i )
            {
                scalar theta = ( sd[i] ).norm();
                scalar q = cos( theta ), w = 1 - q, x = -sd[i][0] / theta, y = -sd[i][1] / theta, z = -sd[i][2] / theta,
                       s1 = -y * z * w, s2 = x * z * w, s3 = -x * y * w, p1 = x * sin( theta ), p2 = y * sin( theta ),
                       p3 = z * sin( theta );

                scalar t1, t2, t3;
                if( theta > 1.0e-20 ) // if theta is too small we do nothing
                {
                    t1      = ( q + z * z * w ) * s[i][0] + ( s1 + p1 ) * s[i][1] + ( s2 + p2 ) * s[i][2];
                    t2      = ( s1 - p1 ) * s[i][0] + ( q + y * y * w ) * s[i][1] + ( s3 + p3 ) * s[i][2];
                    t3      = ( s2 - p2 ) * s[i][0] + ( s3 - p3 ) * s[i][1] + ( q + x * x * w ) * s[i][2];
                    s[i][0] = t1;
                    s[i][1] = t2;
                    s[i][2] = t3;
                };
            } );
    }
}

scalar maximum_rotation( const vectorfield & searchdir, scalar maxmove )
{
    int nos          = searchdir.size();
    scalar theta_rms = 0;
    theta_rms        = sqrt(
        Backend::par::reduce( searchdir, [] SPIRIT_LAMBDA( const Vector3 & v ) { return v.squaredNorm(); } ) / nos );
    scalar scaling = ( theta_rms > maxmove ) ? maxmove / theta_rms : 1.0;
    return scaling;
}

void atlas_rotate(
    std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<scalarfield> & a3_coords,
    const std::vector<vector2field> & searchdir )
{
    int noi = configurations.size();
    int nos = configurations[0]->size();
    for( int img = 0; img < noi; img++ )
    {
        auto spins = configurations[img]->data();
        auto d     = searchdir[img].data();
        auto a3    = a3_coords[img].data();
        Backend::par::apply(
            nos,
            [nos, spins, d, a3] SPIRIT_LAMBDA( int idx )
            {
                const scalar gamma = ( 1 + spins[idx][2] * a3[idx] );
                const scalar denom = ( spins[idx].head<2>().squaredNorm() ) / gamma
                                     + 2 * d[idx].dot( spins[idx].head<2>() ) + gamma * d[idx].squaredNorm();
                spins[idx].head<2>() = 2 * ( spins[idx].head<2>() + d[idx] * gamma );
                spins[idx][2]        = a3[idx] * ( gamma - denom );
                spins[idx] *= 1 / ( gamma + denom );
            } );
    }
}

bool ncg_atlas_check_coordinates(
    const std::vector<std::shared_ptr<vectorfield>> & spins, std::vector<scalarfield> & a3_coords, scalar tol )
{
    int noi = spins.size();
    int nos = ( *spins[0] ).size();

    // We use `int` instead of `bool`, because somehow cuda does not like pointers to bool
    // TODO: fix in future
    field<int> result = field<int>( 1, int( false ) );

    for( int img = 0; img < noi; img++ )
    {
        auto s    = spins[0]->data();
        auto a3   = a3_coords[img].data();
        int * res = &result[0];

        Backend::par::apply(
            nos,
            [s, a3, tol, res] SPIRIT_LAMBDA( int idx )
            {
                if( s[idx][2] * a3[idx] < tol && res[0] == int( false ) )
                    res[0] = int( true );
            } );
    }

    return bool( result[0] );
}

void lbfgs_atlas_transform_direction(
    std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<scalarfield> & a3_coords,
    std::vector<field<vector2field>> & atlas_updates, std::vector<field<vector2field>> & grad_updates,
    std::vector<vector2field> & searchdir, std::vector<vector2field> & grad_pr, scalarfield & rho )
{
    int noi = configurations.size();
    int nos = configurations[0]->size();

    for( int n = 0; n < atlas_updates[0].size(); n++ )
    {
        rho[n] = 1 / rho[n];
    }

    for( int img = 0; img < noi; img++ )
    {
        auto s    = ( *configurations[img] ).data();
        auto a3   = a3_coords[img].data();
        auto sd   = searchdir[img].data();
        auto g_pr = grad_pr[img].data();
        auto rh   = rho.data();

        auto n_mem = atlas_updates[img].size();

        field<Vector2 *> t1( n_mem ), t2( n_mem );
        for( int n = 0; n < n_mem; n++ )
        {
            t1[n] = ( atlas_updates[img][n].data() );
            t2[n] = ( grad_updates[img][n].data() );
        }

        auto a_up = t1.data();
        auto g_up = t2.data();

        Backend::par::apply(
            nos,
            [s, a3, sd, g_pr, rh, a_up, g_up, n_mem] SPIRIT_LAMBDA( int idx )
            {
                scalar factor = 1;
                if( s[idx][2] * a3[idx] < 0 )
                {
                    // Transform coordinates to optimal map
                    a3[idx] = ( s[idx][2] > 0 ) ? 1 : -1;
                    factor  = ( 1 - a3[idx] * s[idx][2] ) / ( 1 + a3[idx] * s[idx][2] );
                    sd[idx] *= factor;
                    g_pr[idx] *= factor;

                    for( int n = 0; n < n_mem; n++ )
                    {
                        rh[n] = rh[n] + ( factor * factor - 1 ) * a_up[n][idx].dot( g_up[n][idx] );
                        a_up[n][idx] *= factor;
                        g_up[n][idx] *= factor;
                    }
                }
            } );
    }

    for( int n = 0; n < atlas_updates[0].size(); n++ )
    {
        rho[n] = 1 / rho[n];
    }
}

} // namespace Solver_Kernels
} // namespace Engine
