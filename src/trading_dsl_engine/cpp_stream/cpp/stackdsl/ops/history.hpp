#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template<std::size_t N,class In,class Out,std::int64_t Limit=-1,class Execution=DirectExecution<N>>
struct FFillNode {
    std::array<double,Execution::state_size> last{};
    std::array<std::int64_t,Execution::state_size> streak{};
    std::array<std::uint8_t,Execution::state_size> seen{};
    void setup() noexcept { last.fill(kNaN); streak.fill(0); seen.fill(0); }
    template<class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* out=ctx.template write_ptr<Out>();
        const std::size_t begin=execution_lane_begin<N,Execution>(ctx);
        const std::size_t end=execution_lane_end<N,Execution>(ctx);
        for(std::size_t lane=begin;lane<end;++lane){
            const std::size_t index=Execution::state_index(ctx,lane);
            const double value=ctx.template read<In>(lane);
            if(finite(value)){ last[index]=value; streak[index]=0; seen[index]=1; out[lane]=value; }
            else if(seen[index] && (Limit<0 || streak[index]<Limit)){ ++streak[index]; out[lane]=last[index]; }
            else out[lane]=kNaN;
        }
    }
};

template<std::size_t N,class In,class Out,std::size_t Lag,std::size_t MaxLag,class Execution=DirectExecution<N>>
struct ShiftNode {
    static_assert(Lag<=MaxLag);
    static constexpr std::size_t Capacity=MaxLag+1;
    std::array<std::array<double,Execution::state_size>,Capacity> buffer{};
    std::size_t position=0;
    std::size_t count=0;
    void setup() noexcept { for(auto& row:buffer) row.fill(kNaN); position=0; count=0; }
    template<class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* out=ctx.template write_ptr<Out>();
        const std::size_t read_position=(position+Capacity-Lag)%Capacity;
        const std::size_t begin=execution_lane_begin<N,Execution>(ctx);
        const std::size_t end=execution_lane_end<N,Execution>(ctx);
        for(std::size_t lane=begin;lane<end;++lane){
            const std::size_t index=Execution::state_index(ctx,lane);
            const double value=ctx.template read<In>(lane);
            out[lane]=Lag==0?value:(count>=Lag?buffer[read_position][index]:kNaN);
            buffer[position][index]=value;
        }
        position=(position+1)%Capacity;
        if(count<Capacity)++count;
    }
};

}  // namespace stackdsl
