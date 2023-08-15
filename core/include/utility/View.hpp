#pragma once
#ifndef SPIRIT_CORE_UTILITY_VIEW_HPP
#define SPIRIT_CORE_UTILITY_VIEW_HPP

#include <utility/Execution_Defs.hpp>

#ifdef SPIRIT_USE_CUDA
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#endif

#include <ranges>
#include <span>

namespace Utility
{

struct range_size_t
{
    template<std::ranges::sized_range R>
    [[nodiscard]] HOSTDEVICEQUALIFIER auto constexpr operator()( R const & r ) const noexcept
    {
        return std::ranges::size( r );
    }

    template<std::ranges::sized_range R1, std::ranges::sized_range R2, std::ranges::sized_range... Rs>
    [[nodiscard]] HOSTDEVICEQUALIFIER auto constexpr
    operator()( R1 const & r1, R2 const & r2, Rs const &... rs ) const noexcept
    {
        return std::min( std::ranges::size( r1 ), operator()( r2, rs... ) );
    }
};

template<std::ranges::contiguous_range Range>
[[nodiscard]] HOSTDEVICEQUALIFIER constexpr decltype( auto ) make_view( Range && r ) noexcept
{
    return std::span{ std::ranges::data( (Range &&)r ), std::ranges::size( (Range &&)r ) };
}

template<class T>
[[nodiscard]] HOSTDEVICEQUALIFIER constexpr auto make_view( std::span<T> s ) noexcept
{
    return s;
}

#ifdef SPIRIT_USE_CUDA

template<class T>
[[nodiscard]] constexpr auto HOSTDEVICEQUALIFIER make_view( thrust::universal_vector<T> & v ) noexcept
{
    return std::span<T>{ thrust::raw_pointer_cast( v.data() ), v.size() };
}

template<class T>
[[nodiscard]] constexpr auto HOSTDEVICEQUALIFIER make_view( thrust::universal_vector<T> const & v ) noexcept
{
    return std::span<T const>{ thrust::raw_pointer_cast( v.data() ), v.size() };
}

#endif

struct view_of_t
{
    template<std::ranges::range Range>
    [[nodiscard]] constexpr decltype( auto ) HOSTDEVICEQUALIFIER operator()( Range && r ) const noexcept
    {
        return make_view( r );
    }
};

struct const_view_of_t
{
    template<std::ranges::range Range>
    [[nodiscard]] constexpr decltype( auto ) HOSTDEVICEQUALIFIER operator()( Range const & r ) const noexcept
    {
        return make_view( r );
    }
};

} // namespace Utility

inline constexpr Utility::range_size_t range_size{};
inline constexpr Utility::view_of_t view_of{};
inline constexpr Utility::const_view_of_t const_view_of{};

#endif
