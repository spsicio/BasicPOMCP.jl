ent_from_part_e(ent_partial, tot) = log(tot) - ent_partial / tot
ent_from_part_g(ent_partial, tot) = 1.0 - ent_partial / (tot * tot)
ent_part_e(w) = isapprox(w, 0.0) ? 0.0 : w * log(w)
ent_part_g(w) = w * w

abstract type AbstractPOMCPBelief{S} end

struct POMCPBelief{S} <: AbstractPOMCPBelief{S} end

mutable struct POMCPeBelief{S} <: AbstractPOMCPBelief{S}
    n::Int # number of particles
    ws::Dict{S, Int} # weights of the states in the distribution
    ent::Float64 # entropy of the distribution
    ent_partial::Float64 # intermediate variables for entropy computation
    
    function POMCPeBelief{S}(sp::S, ent_part) where {S}
        ws = Dict{S, Int}(sp => 1)
        new(1, ws, 0.0, ent_part(1))
    end
end

abstract type AbstractPOMCPNodeFilter end
abstract type AbstractPOMCPeNodeFilter <: AbstractPOMCPNodeFilter end
struct POMCPNodeFilter <: AbstractPOMCPNodeFilter end
struct POMCPeNodeFilter <: AbstractPOMCPeNodeFilter end
struct POMCPgNodeFilter <: AbstractPOMCPeNodeFilter end

belief_type(::Type{POMCPNodeFilter}, ::Type{P}) where {P<:POMDP} = POMCPBelief{statetype(P)}
belief_type(::Type{<:AbstractPOMCPeNodeFilter}, ::Type{P}) where {P<:POMDP} = POMCPeBelief{statetype(P)}

init_node_belief(::POMCPNodeFilter, ::S) where {S} = POMCPBelief{S}()
init_node_belief(::POMCPeNodeFilter, sp::S) where {S} = POMCPeBelief{S}(sp, ent_part_e)
init_node_belief(::POMCPgNodeFilter, sp::S) where {S} = POMCPeBelief{S}(sp, ent_part_g)

function push_pariticle!(::POMCPBelief, ::POMCPNodeFilter, sp) end

function push_pariticle!(b::POMCPeBelief, ::POMCPeNodeFilter, sp) 
    b.n += 1
    cur = get(b.ws, sp, 0); nxt = cur + 1
    b.ent_partial += ent_part_e(nxt) - ent_part_e(cur)
    b.ent = ent_from_part_e(b.ent_partial, nxt)
    b.ws[sp] = nxt
end

function push_pariticle!(b::POMCPeBelief, ::POMCPgNodeFilter, sp) 
    b.n += 1
    cur = get(b.ws, sp, 0); nxt = cur + 1
    b.ent_partial += ent_part_g(nxt) - ent_part_g(cur)
    b.ent = ent_from_part_g(b.ent_partial, nxt)
    b.ws[sp] = nxt
end

weighted_entropy(b::POMCPBelief) = 0.0
weighted_entropy(b::POMCPeBelief) = b.ent * b.n
