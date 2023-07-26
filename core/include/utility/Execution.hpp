#pragma once
#ifndef SPIRIT_CORE_UTILITY_EXECUTION_HPP
#define SPIRIT_CORE_UTILITY_EXECUTION_HPP


#include <utility/Exception.hpp>

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

#ifdef SPIRIT_USE_CUDA
    #include <nvexec/stream_context.cuh>
    #include <nvexec/multi_gpu_context.cuh>
#endif

#include <cstdint>                   
#include <thread>                   
#include <cassert>                   


namespace Execution {


struct Void_Schedule {};

struct Void_Context {
    static Void_Schedule get_scheduler() noexcept { return Void_Schedule{}; }
};



#ifdef SPIRIT_NO_STDEXEC

struct Resource_Shape
{
    inline static constexpr int devices = 1;
    inline static constexpr int threads = 1;
};

class Compute_Resource {
public:
    friend class Context;

    explicit
    Compute_Resource (int = 0) {}

private:
    Resource_Shape shape_;
    Void_Context resource_;
};

#else   // SPIRIT_NO_STDEXEC


#ifdef SPIRIT_USE_CUDA

struct Resource_Shape
{
    int devices = 1;
    inline static constexpr std::int64_t threads = 2147483647L*65536L*65536L;
};  

class Compute_Resource {
public:
    friend class Context;

    explicit
    Compute_Resource ():
        shape_{ .devices = 1 }
    {
#ifdef SPIRIT_USE_MULTI_GPU
        cudaGetDeviceCount(&shape_.devices);
#endif
    }

private:
    Resource_Shape shape_;
#ifdef SPIRIT_USE_MULTI_GPU
    nvexec::multi_gpu_stream_context resource_;
#else
    nvexec::stream_context resource_; 
#endif
};
       
#else  // SPIRIT_USE_CUDA

struct Resource_Shape
{
    inline static constexpr int devices = 1;
    int threads = 1;
};

class Compute_Resource {
public:
    friend class Context;

    explicit
    Compute_Resource (int num_threads = std::thread::hardware_concurrency()):
        shape_{ .threads = num_threads },
        resource_(num_threads)
    {}

private:
    Resource_Shape shape_;
    exec::static_thread_pool resource_;
};

#endif  // SPIRIT_USE_CUDA
#endif  // SPIRIT_NO_STDEXEC




/* 
 * Reference to compute resources that can be stored / passed by value
 */
class Context
{
public:
    // TODO default ctor actually needed?
    Context () = default;

    Context (Compute_Resource& res) noexcept: res_{&res} {}

    [[nodiscard]]
    auto get_scheduler () { 
        assert (res_ != nullptr);
        return res_->resource_.get_scheduler();
    }

    [[nodiscard]]
    Resource_Shape resource_shape () const noexcept { 
        return static_cast<bool>(res_) ? res_->shape_ : Resource_Shape{};
    }

private:
    Compute_Resource* res_ = nullptr;
};


}


#endif

