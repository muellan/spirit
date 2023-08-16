#pragma once
#ifndef SPIRIT_CORE_UTILITY_STDEXEC_ALGORITHMS_HPP
#define SPIRIT_CORE_UTILITY_STDEXEC_ALGORITHMS_HPP

#include <utility/Execution.hpp>
#include <utility/Indices.hpp>
#include <utility/View.hpp>

// #include <fmt/format.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// #include <execution>
#include <algorithm>
#include <concepts>
#include <numeric>
#include <ranges>
#include <span>
#include <utility>

// clang-format off
namespace Execution {


//-----------------------------------------------------------------------------
// template <class... Ts>
// using any_sender_of =
//     typename exec::any_receiver_ref<stdexec::completion_signatures<Ts...>>::template any_sender<>;



//-----------------------------------------------------------------------------
template <class Fn, class Range>
concept IndexToValueMapping = 
    std::copy_constructible<Fn> &&
    std::invocable<Fn,std::size_t> &&
    std::convertible_to<std::invoke_result_t<Fn,std::size_t>,
                        std::ranges::range_value_t<Range>>;




//-----------------------------------------------------------------------------
[[nodiscard]]
inline auto schedule (Context ctx)
{
    return stdexec::schedule(ctx.get_scheduler());
}




//-----------------------------------------------------------------------------
template <class T, class... Ts>
auto transfer_just (Context ctx, T&& t, Ts&&... ts)
{
    return stdexec::transfer_just(ctx.get_scheduler(), (T&&)t, (Ts&&)ts...);
}




//-----------------------------------------------------------------------------
template <class InRange, class OutRange>
requires
    std::ranges::random_access_range<InRange> &&
    std::ranges::random_access_range<OutRange>
[[nodiscard]]
auto copy_async (InRange const& input, OutRange& output)
{
    auto in = view_of(input);
    auto out = view_of(output);

    return stdexec::bulk(range_size(input, output),
            [=](std::size_t i){ out[i] = in[i]; });
}


//-----------------------------------------------------------------------------
template <class InValue, class OutValue>
[[nodiscard]]
auto copy_async (std::span<const InValue> in, std::span<OutValue> out)
{
    return stdexec::bulk(range_size(in, out),
            [=](std::size_t i){ out[i] = in[i]; });
}




//-----------------------------------------------------------------------------
/// @brief fully synchronous 'bulk' variant
template <class Body>
requires
    std::copy_constructible<Body> &&
    std::invocable<Body,std::size_t>
void for_each_index (Context ctx, std::size_t size, Body&& body)
{
    auto task = 
        stdexec::schedule(ctx.get_scheduler())
    |   stdexec::bulk(size, (Body&&)body);

    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <class InRange, class Body>
requires 
    std::ranges::random_access_range<InRange> &&
    std::ranges::sized_range<InRange> &&
    std::copy_constructible<Body> &&
    std::invocable<Body,std::ranges::range_value_t<InRange>>
[[nodiscard]] stdexec::sender
auto for_each_async (Context ctx, InRange&& input, Body body)
{
    return
        stdexec::transfer_just(ctx.get_scheduler(), view_of((InRange&&)input)) 
    |   stdexec::bulk(std::ranges::size(input),
            [=](std::size_t idx, auto in) {
                body(in[idx]);
            })
    |   stdexec::then([](auto in){ return in; });
}


//-----------------------------------------------------------------------------
template <class InRange, class Body>
requires 
    std::ranges::random_access_range<InRange> &&
    std::ranges::sized_range<InRange> &&
    std::copy_constructible<Body> &&
    std::invocable<Body,std::ranges::range_value_t<InRange>>
void for_each (Context ctx, InRange&& input, Body&& body)
{
    auto task = for_each_async(ctx, (InRange&&)input, (Body&&)body); 
    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <class GridExtents, class Body>
requires 
    std::copy_constructible<Body>
void for_each_grid_index (Context ctx, GridExtents ext, Body body)
{
    // size of collapsed index range
    std::size_t size = 1;
    for (auto x : ext) { size *= static_cast<std::size_t>(x); }
    
    auto const N = static_cast<int>(ext.size());
    auto const tileCount = std::min(size, static_cast<std::size_t>(ctx.resource_shape().threads));
    auto const tileSize  = static_cast<std::size_t>((size + tileCount - 1) / tileCount);

    auto sched = ctx.get_scheduler();

    auto task = stdexec::schedule(sched) 
    |   stdexec::bulk(tileCount, [=](std::size_t tileIdx) mutable
        {
            // start/end of collapsed index range
            auto const start = tileIdx * tileSize;
            auto const end   = std::min(size, (tileIdx + 1) * tileSize);
            if (start >= end) return;
            // compute start index
            GridExtents idx;
            for (int i = 0; i < N; ++i) { idx[i] = 0; }

            if (start > 0) {
                std::size_t mul[N];
                mul[0] = 1;
                for (int i = 0; i < N-1; ++i) {
                    mul[i+1] = mul[i] * ext[i]; 
                }
                auto offset = start;
                for (int i = N; i > 0; --i) {
                    if (offset >= mul[i-1]) {
                        idx[i-1] += offset / mul[i-1];
                        offset = offset % mul[i-1];
                        if (offset == 0) break;
                    }
                }
            }
            // execute body on local index subrange
            for (auto ci = start; ci < end; ++ci) {
                body(idx);
                // increment index
                for (int i = 0; i < N; ++i) {
                    ++idx[i];
                    if (idx[i] < ext[i]) break;
                    idx[i] = 0;
                }
            }
        });

    stdexec::sync_wait(task).value();
}




//-----------------------------------------------------------------------------
template <class OutRange, class Generator>
requires std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         IndexToValueMapping<Generator,OutRange>
[[nodiscard]] stdexec::sender
auto generate_indexed_async (
    Context ctx, OutRange & output, Generator gen)
{
    return
        stdexec::transfer_just(ctx.get_scheduler(), view_of(output))
    |   stdexec::bulk(std::ranges::size(output),
            [=](auto idx, auto out) {
                out[idx] = gen(idx);
            })
    |   stdexec::then([](auto out){ return out; });
}


//-----------------------------------------------------------------------------
template <class OutRange, class Generator>
requires std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         IndexToValueMapping<Generator,OutRange>
void generate_indexed (Context ctx, OutRange & output, Generator&& gen)
{
    auto task = generate_indexed_async(ctx, output, (Generator&&)(gen));
    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <class InRange, class OutRange, class Transf>
requires std::ranges::random_access_range<InRange> &&
         std::ranges::sized_range<InRange> &&
         std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         std::copy_constructible<Transf>
[[nodiscard]] stdexec::sender
auto transform_async (
    Context ctx, InRange const& input, OutRange & output, Transf fn)
{
    return
        stdexec::transfer_just(ctx.get_scheduler(), 
                               view_of(input), view_of(output) )
    |   stdexec::bulk(std::ranges::size(input),
            [=](std::size_t idx, auto in, auto out) {
                out[idx] = fn(in[idx]);
            })
    |   stdexec::then([](auto&&, auto out){ return out; });
}


//-----------------------------------------------------------------------------
template <class InRange, class OutRange, class Transf>
requires std::ranges::random_access_range<InRange> &&
         std::ranges::sized_range<InRange> &&
         std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         std::copy_constructible<Transf>
void transform (
    Context ctx, InRange const& input, OutRange & output, Transf&& fn)
{
    auto task = transform_async(ctx, input, output, (Transf&&)(fn)); 
    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
template <class InRange, class OutRange, class Transf>
requires std::ranges::random_access_range<InRange> &&
         std::ranges::sized_range<InRange> &&
         std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         std::copy_constructible<Transf>
[[nodiscard]] stdexec::sender
auto transform_indexed_async (
    Context ctx, InRange const& input, OutRange & output, Transf fn)
{
    return
        stdexec::transfer_just(ctx.get_scheduler(), 
                               view_of(input), view_of(output) )
    |   stdexec::bulk(std::ranges::size(input),
            [=](auto idx, auto in, auto out) {
                out[idx] = fn(idx, in[idx]);
            })
    |   stdexec::then([](auto out){ return out; });

}


//-----------------------------------------------------------------------------
template <class InRange, class OutRange, class Transf>
requires std::ranges::random_access_range<InRange> &&
         std::ranges::sized_range<InRange> &&
         std::ranges::random_access_range<OutRange> &&
         std::ranges::sized_range<OutRange> &&
         std::copy_constructible<Transf>
void transform_indexed (
    Context ctx, InRange const& input, OutRange & output, Transf&& fn)
{
    auto task = transform_indexed_async(ctx, input, output, (Transf&&)(fn));
    stdexec::sync_wait(std::move(task)).value();
}




//-----------------------------------------------------------------------------
#ifndef SPIRIT_USE_CUDA


template <class Result>
[[nodiscard]] stdexec::sender
auto reduce_async (Result init)
{
    return exec::reduce(init);
}


template <class Result, class ReductionOp>
[[nodiscard]] stdexec::sender
auto reduce_async (Result init, ReductionOp&& redOp)
{
    return exec::reduce(init, (ReductionOp&&)redOp);
}

#else


template <class Result>
[[nodiscard]] stdexec::sender
auto reduce_async (Result init)
{
    return nvexec::reduce(init);
}


template <class Result, class ReductionOp>
[[nodiscard]] stdexec::sender
auto reduce_async (Result init, ReductionOp&& redOp)
{
    return nvexec::reduce(init, (ReductionOp&&)redOp);
}

#endif


template <class InRange, class Result>
requires
    std::ranges::random_access_range<InRange> &&
    std::ranges::sized_range<InRange>
[[nodiscard]] Result 
reduce (
    Context ctx,
    InRange const& input, Result&& init)
{
    auto task = 
        stdexec::transfer_just(ctx.get_scheduler(), view_of(input))
    |   reduce_async(ctx, input, (Result&&)(init));

    return std::get<0>(stdexec::sync_wait(std::move(task)).value());
}


template <class InRange, class Result, class ReductionOp>
requires
    std::ranges::random_access_range<InRange> &&
    std::ranges::sized_range<InRange> &&
    std::copy_constructible<ReductionOp>
[[nodiscard]] Result 
reduce (
    Context ctx,
    InRange const& input, Result&& init, ReductionOp&& redOp)
{
    auto task = 
        stdexec::transfer_just(ctx.get_scheduler(), view_of(input))
    |   reduce_async(ctx, input, (Result&&)(init), (ReductionOp&&)(redOp));

    return std::get<0>(stdexec::sync_wait(std::move(task)).value());
}




// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2,
//     class OutRange,
//     class Transf,
//     class Value1 = std::ranges::range_value_t<InRange1>,
//     class Value2 = std::ranges::range_value_t<InRange2>,
//     class OutValue = std::ranges::range_value_t<OutRange>
// >
// requires 
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::random_access_range<OutRange> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::ranges::sized_range<OutRange> &&
//     std::copy_constructible<Transf> &&
//     std::invocable<Transf,Value1,Value2> &&
//     std::convertible_to<OutValue,std::invoke_result_t<Transf,Value1,Value2>>
// [[nodiscard]] stdexec::sender
// auto zip_transform_async (
//     Context ctx,
//     InRange1 const& input1,
//     InRange2 const& input2,
//     OutRange & output,
//     Transf fn)
// {
//     auto const inSize = std::min(std::ranges::size(input1),
//                                  std::ranges::size(input2));
//
//     return
//         stdexec::transfer_just(ctx.get_scheduler(),
//                                view_of(input1), view_of(input2),
//                                view_of(output) )
//     |   stdexec::bulk(inSize,
//         [=](auto idx, auto in1, auto in2, auto out) {
//             out[idx] = fn(in1[idx], in2[idx]);
//         })
//     |   stdexec::then([](auto&&, auto&&, auto out){ return out; });
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2,
//     class OutRange,
//     class Transf,
//     class Value1 = std::ranges::range_value_t<InRange1>,
//     class Value2 = std::ranges::range_value_t<InRange2>,
//     class OutValue = std::ranges::range_value_t<OutRange>
// >
// requires 
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::random_access_range<OutRange> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::ranges::sized_range<OutRange> &&
//     std::copy_constructible<Transf> &&
//     std::invocable<Transf,Value1,Value2,OutValue&>
// [[nodiscard]] stdexec::sender
// auto zip_transform_async (
//     Context ctx,
//     InRange1 const& input1,
//     InRange2 const& input2,
//     OutRange & output,
//     Transf fn)
// {
//     auto const inSize = std::min(std::ranges::size(input1),
//                                  std::ranges::size(input2));
//
//     return
//         stdexec::transfer_just(ctx.get_scheduler(),
//                                view_of(input1), view_of(input2), view_of(output))
//     |   stdexec::bulk(inSize,
//         [=](auto idx, auto in1, auto in2, auto out) {
//             fn(in1[idx], in2[idx], out[idx]);
//         })
//     |   stdexec::then([](auto&&, auto&&, auto out){ return out; });
// }
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2,
//     class OutRange,
//     class Transf
// >
// requires 
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::random_access_range<OutRange> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::ranges::sized_range<OutRange> &&
//     std::copy_constructible<Transf>
// void zip_transform (
//     Context ctx,
//     InRange1 const& input1,
//     InRange2 const& input2,
//     OutRange & output,
//     Transf&& fn)
// {
//     auto task = zip_transform_async(ctx, input1, input2, output,
//                                     (Transf&&)(fn));
//     
//     stdexec::sync_wait(std::move(task)).value();
// }
//
//
// #ifndef SPIRIT_USE_CUDA
//
//
// //-----------------------------------------------------------------------------
// template <class InRange, class Result, class ReductionOp>
// requires
//     std::ranges::random_access_range<InRange> &&
//     std::ranges::sized_range<InRange> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] stdexec::sender 
// auto reduce_async (
//     Context ctx,
//     InRange const& input, Result init, ReductionOp redOp = std::plus<>{})
// {
//     using size_t_ = std::ranges::range_size_t<InRange>;
//
//     auto const inSize      = std::ranges::size(input);
//     auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
//     auto const tileSize    = (inSize + threadCount - 1) / threadCount;
//     auto const tileCount   = (inSize + tileSize - 1) / tileSize;
//
//     std::vector<Result> partials (tileCount, Result(0));
//
//     return
//         stdexec::transfer_just(ctx.get_scheduler(),
//                                view_of(input), std::move(partials) )
//     |   stdexec::bulk(tileCount,
//         [=](std::size_t tileIdx, auto in, auto&& parts)
//         {
//             auto const first = std::ranges::begin(in) + (tileIdx * tileSize);
//             auto const last  = std::ranges::begin(in)
//                              + std::min(inSize, (tileIdx + 1) * tileSize);
//
//             parts[tileIdx] = std::reduce(std::next(first), last, *first, redOp);
//         })
//     |   stdexec::then([=](auto, auto&& parts)
//         {
//             return std::reduce(begin(parts), end(parts), init);
//         });
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange,
//     class Result,
//     class TransformOp,
//     class ReductionOp
// >
// requires
//     std::ranges::random_access_range<InRange> &&
//     std::ranges::sized_range<InRange> &&
//     std::copy_constructible<TransformOp> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] stdexec::sender 
// auto transform_reduce_async (
//     Context ctx,
//     InRange const& input, Result init,
//     TransformOp transf,
//     ReductionOp redOp = std::plus<>{})
// {
//     using size_t_ = std::ranges::range_size_t<InRange>;
//
//     auto const inSize      = std::ranges::size(input);
//     auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
//     auto const tileSize    = (inSize + threadCount - 1) / threadCount;
//     auto const tileCount   = (inSize + tileSize - 1) / tileSize;
//
//     std::vector<Result> partials (tileCount, Result(0));
//
//     return
//         stdexec::transfer_just(ctx.get_scheduler(),
//                                view_of(input), std::move(partials) )
//     |   stdexec::bulk(tileCount,
//         [=](std::size_t tileIdx, auto in, auto&& parts)
//         {
//             auto const first = std::ranges::begin(in) + (tileIdx * tileSize);
//             auto const last  = std::ranges::begin(in)
//                              + std::min(inSize, (tileIdx + 1) * tileSize);
//
//             parts[tileIdx] = std::transform_reduce(
//                              std::next(first), last, *first, transf, redOp);
//         })
//     |   stdexec::then([=](auto, auto&& parts)
//         {
//             return std::reduce(begin(parts), end(parts), init);
//         });
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2, 
//     class Result,
//     class ReductionOp
// >
// requires
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] stdexec::sender
// auto zip_reduce_async (
//     Context ctx,
//     InRange1 const& input1,
//     InRange2 const& input2,
//     Result init,
//     ReductionOp&& redOp = std::plus<>{})
// {
//     using size_t_ = std::common_type_t<std::ranges::range_size_t<InRange1>,
//                                        std::ranges::range_size_t<InRange2>>;
//
//     auto const inSize      = std::min(std::ranges::size(input1), std::ranges::size(input2));
//     auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
//     auto const tileSize    = (inSize + threadCount - 1) / threadCount;
//     auto const tileCount   = (inSize + tileSize - 1) / tileSize;
//
//     std::vector<Result> partials (tileCount, Result(0));
//
//     return stdexec::transfer_just(ctx.get_scheduler(),
//                                   view_of(input1), view_of(input2),
//                                   std::move(partials))
//     |   stdexec::bulk(tileCount,
//         [=](std::size_t tileIdx, auto in1, auto in2, auto&& parts)
//         {
//             auto const start = tileIdx * tileSize;
//             auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);
//
//             for (auto i = start; i < end; ++i) {
//                 parts[tileIdx] = redOp(parts[tileIdx], in1[i], in2[i]);
//             }
//         })
//     |   stdexec::then([=](auto, auto, auto&& parts)
//         {
//             return std::reduce(begin(parts), end(parts), init);
//         });
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2, 
//     class Result,
//     class ReductionOp
// >
// requires
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] stdexec::sender 
// auto zip_reduce_sum_async (
//     Context ctx,
//     InRange1 const& input1,
//     InRange2 const& input2,
//     Result init,
//     ReductionOp redOp = std::plus<>{})
// {
//     using size_t_ = std::common_type_t<std::ranges::range_size_t<InRange1>,
//                                        std::ranges::range_size_t<InRange2>>;
//
//     auto const inSize      = std::min(std::ranges::size(input1), std::ranges::size(input2));
//     auto const threadCount = std::min(inSize, static_cast<size_t_>(ctx.resource_shape().threads));
//     auto const tileSize    = (inSize + threadCount - 1) / threadCount;
//     auto const tileCount   = (inSize + tileSize - 1) / tileSize;
//
//     std::vector<Result> partials (tileCount, Result(0));
//
//     return stdexec::transfer_just(ctx.get_scheduler(),
//                                   view_of(input1), view_of(input2),
//                                   std::move(partials))
//     |   stdexec::bulk(tileCount,
//         [=](std::size_t tileIdx, auto in1, auto in2, auto&& parts)
//         {
//             auto const start = tileIdx * tileSize;
//             auto const end   = std::min(inSize, (tileIdx + 1) * tileSize);
//
//             for (auto i = start; i < end; ++i) {
//                 parts[tileIdx] += redOp(in1[i], in2[i]);
//             }
//         })
//     |   stdexec::then([=](auto, auto, auto&& parts)
//         {
//             return std::reduce(begin(parts), end(parts), init);
//         });
// }
//
//
// #else
//
//
// //-----------------------------------------------------------------------------
// template <class InRange, class ResultValue>
// requires
//     std::ranges::random_access_range<InRange> &&
//     std::ranges::sized_range<InRange>
// [[nodiscard]] stdexec::sender
// auto reduce_async (
//     Context ctx,
//     InRange const& input, ResultValue init)
// {
//     return stdexec::transfer_just(ctx.get_scheduler(), view_of(input))
//          | nvexec::reduce(init)
//          | stdexec::then([](auto in){ return in; });
// }
//
//
// //-----------------------------------------------------------------------------
// template <class InRange, class ResultValue, class ReductionOp>
// requires
//     std::ranges::random_access_range<InRange> &&
//     std::ranges::sized_range<InRange> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] stdexec::sender
// auto reduce_async (
//     Context ctx,
//     InRange const& input, ResultValue init, ReductionOp&& redOp)
// {
//     return stdexec::transfer_just(ctx.get_scheduler(), view_of(input))
//          | nvexec::reduce(init, (ReductionOp&&)redOp)
//          | stdexec::then([](auto in){ return in; });
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2, 
//     class Result,
//     class ReductionOp
// >
// requires
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] stdexec::sender 
// auto zip_reduce_async (
//     Context,
//     InRange1 const&,
//     InRange2 const&,
//     Result init,
//     ReductionOp)
// {
//     // TODO implement
//     return stdexec::just() | stdexec::then([=]{ return init; });
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2, 
//     class Result,
//     class ReductionOp
// >
// requires
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] stdexec::sender 
// auto zip_reduce_sum_async (
//     Context,
//     InRange1 const&,
//     InRange2 const&,
//     Result init,
//     ReductionOp)
// {
//     // TODO implement
//     return stdexec::just() | stdexec::then([=]{ return init; });
// }
//
//
// #endif  // SPIRIT_USE_CUDA
//
//
//
//
// //-----------------------------------------------------------------------------
// template <class InRange, class Result, class ReductionOp>
// requires
//     std::ranges::random_access_range<InRange> &&
//     std::ranges::sized_range<InRange> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] Result 
// reduce (
//     Context ctx,
//     InRange const& input, Result&& init, ReductionOp&& redOp)
// {
//     auto task = reduce_async(ctx, input,
//                              (Result&&)(init),
//                              (ReductionOp&&)(redOp));
//
//     return std::get<0>(stdexec::sync_wait(std::move(task)).value());
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange,
//     class Result,
//     class TransformOp,
//     class ReductionOp
// >
// requires
//     std::ranges::random_access_range<InRange> &&
//     std::ranges::sized_range<InRange> &&
//     std::copy_constructible<TransformOp> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] Result 
// transform_reduce (
//     Context ctx,
//     InRange const& input, Result&& init, 
//     TransformOp&& trOp, ReductionOp&& redOp)
// {
//     auto task = transform_reduce_async(ctx, input,
//                                        (Result&&)(init),
//                                        (TransformOp&&)(trOp),
//                                        (ReductionOp&&)(redOp));
//
//     return std::get<0>(stdexec::sync_wait(std::move(task)).value());
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2, 
//     class Result,
//     class ReductionOp
// >
// requires
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] Result 
// zip_reduce_sum (
//     Context ctx,
//     InRange1 const& input1,
//     InRange2 const& input2,
//     Result&& init,
//     ReductionOp&& redOp)
// {
//     auto task = zip_reduce_sum_async(ctx, input1, input2,
//                                      (Result&&)(init),
//                                      (ReductionOp&&)(redOp));
//   
//     return std::get<0>(stdexec::sync_wait(std::move(task)).value());
// }
//
//
//
//
// //-----------------------------------------------------------------------------
// template <
//     class InRange1,
//     class InRange2, 
//     class Result,
//     class ReductionOp
// >
// requires
//     std::ranges::random_access_range<InRange1> &&
//     std::ranges::random_access_range<InRange2> &&
//     std::ranges::sized_range<InRange1> &&
//     std::ranges::sized_range<InRange2> &&
//     std::copy_constructible<ReductionOp>
// [[nodiscard]] Result 
// zip_reduce (
//     Context ctx,
//     InRange1 const& input1,
//     InRange2 const& input2,
//     Result&& init,
//     ReductionOp&& redOp)
// {
//     auto task = zip_reduce_async(ctx, input1, input2,
//                                  (Result&&)(init),
//                                  (ReductionOp&&)(redOp));
//   
//     return std::get<0>(stdexec::sync_wait(std::move(task)).value());
// }


}  // namespace Execution
// clang-format on

#endif
