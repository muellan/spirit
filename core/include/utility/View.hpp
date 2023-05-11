#pragma once
#ifndef SPIRIT_CORE_UTILITY_VIEW_OF_HPP
#define SPIRIT_CORE_UTILITY_VIEW_OF_HPP


#ifdef SPIRIT_USE_CUDA
#include <thrust/device_vector.h>                                                
#endif                                                                           
                                                                                 
#include <span>                                                                  
#include <ranges>                                                                  
                                                                                 
                                                                                 
//-----------------------------------------------------------------------------  
#ifdef SPIRIT_USE_CUDA                                                                   
                                                                                 
template <typename T>                                                            
[[nodiscard]] constexpr auto                                                     
view_of (thrust::device_vector<T>& v) noexcept                                   
{                                                                                
    return std::span<T>{thrust::raw_pointer_cast(v.data()), v.size()};           
}                                                                                
                                                                                 
template <typename T>                                                            
[[nodiscard]] constexpr auto                                                     
view_of (thrust::device_vector<T> const& v) noexcept {                           
    return std::span<T const>{thrust::raw_pointer_cast(v.data()), v.size()};     
}                                                                                
                                                                                 
#endif                                                                           
                                                                                 
                                                                                 
template <std::ranges::contiguous_range Range>                                   
[[nodiscard]] constexpr auto                                                     
view_of (Range&& r) noexcept                                                     
{                                                                                
    return std::span{std::ranges::data(std::forward<Range>(r)),                  
                     std::ranges::size(std::forward<Range>(r))};                 
}                                                                                
                                                                                 
                                                                                 
template <typename T>                                                            
[[nodiscard]] constexpr auto                                                     
view_of (std::span<T> s) noexcept { return s; }                                  



#endif


