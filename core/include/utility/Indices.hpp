#pragma once
#ifndef SPIRIT_CORE_UTILITY_INDICES_HPP
#define SPIRIT_CORE_UTILITY_INDICES_HPP

#include <utility/Execution_Defs.hpp>

#include <cstdint>
#include <iterator>

namespace Utility
{

//-----------------------------------------------------------------------------
/**
 * @brief  non-modifiable range that (conceptually) contains consecutive indices
 */
class index_range
{
public:
    using value_type = std::size_t;
    using size_type  = value_type;

private:
    value_type beg_ = 0;
    value_type end_ = 0;

public:
    class iterator
    {
    public:
        using iterator_category = std::contiguous_iterator_tag;
        using value_type        = index_range::value_type;
        using difference_type   = std::int64_t;

    private:
        value_type i_ = 0;

    public:
        HOSTDEVICEQUALIFIER
        constexpr iterator() = default;

        HOSTDEVICEQUALIFIER
        constexpr explicit iterator( value_type i ) noexcept : i_{ i } {}

        [[nodiscard]] HOSTDEVICEQUALIFIER constexpr value_type operator*() const noexcept
        {
            return i_;
        }

        [[nodiscard]] HOSTDEVICEQUALIFIER constexpr value_type operator[]( difference_type idx ) const noexcept
        {
            return i_ + idx;
        }

        HOSTDEVICEQUALIFIER
        constexpr auto operator<=>( iterator const & ) const noexcept = default;

        HOSTDEVICEQUALIFIER
        constexpr iterator & operator++() noexcept
        {
            ++i_;
            return *this;
        }

        HOSTDEVICEQUALIFIER
        constexpr iterator & operator--() noexcept
        {
            ++i_;
            return *this;
        }

        HOSTDEVICEQUALIFIER
        constexpr iterator operator++( int ) noexcept
        {
            auto old{ *this };
            ++i_;
            return old;
        }

        HOSTDEVICEQUALIFIER
        constexpr iterator operator--( int ) noexcept
        {
            auto old{ *this };
            --i_;
            return old;
        }

        HOSTDEVICEQUALIFIER
        constexpr iterator & operator+=( difference_type offset ) noexcept
        {
            i_ += offset;
            return *this;
        }

        HOSTDEVICEQUALIFIER
        constexpr iterator & operator-=( difference_type offset ) noexcept
        {
            i_ -= offset;
            return *this;
        }

        [[nodiscard]] HOSTDEVICEQUALIFIER constexpr friend iterator
        operator+( iterator const & it, difference_type idx ) noexcept
        {
            return iterator{ it.i_ + idx };
        }

        [[nodiscard]] HOSTDEVICEQUALIFIER constexpr friend iterator
        operator+( difference_type idx, iterator const & it ) noexcept
        {
            return iterator{ it.i_ + idx };
        }

        [[nodiscard]] HOSTDEVICEQUALIFIER constexpr friend iterator
        operator-( iterator const & it, difference_type idx ) noexcept
        {
            return iterator{ it.i_ - idx };
        }

        [[nodiscard]] HOSTDEVICEQUALIFIER constexpr friend iterator
        operator-( difference_type idx, iterator const & it ) noexcept
        {
            return iterator{ it.i_ - idx };
        }

        [[nodiscard]] HOSTDEVICEQUALIFIER friend constexpr difference_type
        operator-( iterator const & a, iterator const & b ) noexcept
        {
            return difference_type( b.i_ ) - difference_type( a.i_ );
        }
    };

    using const_iterator = iterator;

    HOSTDEVICEQUALIFIER
    constexpr index_range() = default;

    HOSTDEVICEQUALIFIER
    constexpr explicit index_range( value_type end ) noexcept : beg_{ 0 }, end_{ end } {}

    HOSTDEVICEQUALIFIER
    constexpr explicit index_range( value_type beg, value_type end ) noexcept : beg_{ beg }, end_{ end } {}

    [[nodiscard]] HOSTDEVICEQUALIFIER constexpr value_type operator[]( size_type idx ) const noexcept
    {
        return beg_ + idx;
    }

    [[nodiscard]] HOSTDEVICEQUALIFIER size_type size() const noexcept
    {
        return end_ - beg_;
    }

    [[nodiscard]] HOSTDEVICEQUALIFIER bool empty() const noexcept
    {
        return end_ <= beg_;
    }

    [[nodiscard]] HOSTDEVICEQUALIFIER constexpr iterator begin() const noexcept
    {
        return iterator{ beg_ };
    }

    [[nodiscard]] HOSTDEVICEQUALIFIER constexpr iterator end() const noexcept
    {
        return iterator{ end_ };
    }

    [[nodiscard]] HOSTDEVICEQUALIFIER friend constexpr iterator begin( index_range const & r ) noexcept
    {
        return r.begin();
    }

    [[nodiscard]] HOSTDEVICEQUALIFIER friend constexpr iterator end( index_range const & r ) noexcept
    {
        return r.end();
    }

    [[nodiscard]] HOSTDEVICEQUALIFIER friend constexpr index_range const & make_view( index_range const & r ) noexcept
    {
        return r;
    }
};

} // namespace Utility

#endif
