struct MaxUCBe
    c::Float64
    e::Float64
    threshold::Int
    discount::Float64
    chain::Bool
    MaxUCBe(c, e; threshold=0, discount=1.0, chain=false) = new(c, e, threshold, discount, chain)
end

function select_best(crit::MaxUCBe, h_node::POMCPObsNode, _best_node_mem::Vector{Int}, rng)
    t = h_node.tree
    h = h_node.node

    ltn = log(t.total_n[h])
    best_nodes = empty!(_best_node_mem)
    best_criterion_val = -Inf
    for node in t.children[h]
        n = t.n[node]
        if isinf(t.v[node])
            criterion_value = t.v[node]
        elseif n == 0
            criterion_value = Inf
        else
            delta_ent = (h == 1 ? 0 : t.node_belief[h].ent) - t.a_sum_ent[node] / n
            sub_max_delta_ent = crit.discount * t.a_max_delta_ent[node]
            if n > crit.threshold
                tmp = crit.chain ? delta_ent + sub_max_delta_ent : max(delta_ent, sub_max_delta_ent)
                if t.o_max_delta_ent[h] < tmp # BackPropagate
                    t.o_max_delta_ent[h] = tmp
                end
            end
            criterion_value = n == 1 ? Inf :
                    t.v[node] + crit.c*sqrt(ltn/n) +
                    crit.e * (delta_ent + sub_max_delta_ent) / sqrt(log(n))
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    ha = rand(rng, best_nodes)
    return ha
end

struct MaxUCB
    c::Float64
end

function select_best(crit::MaxUCB, h_node::POMCPObsNode, _best_node_mem::Vector{Int}, rng)
    t = h_node.tree
    h = h_node.node

    ltn = log(t.total_n[h])
    best_nodes = empty!(_best_node_mem)
    best_criterion_val = -Inf
    for node in t.children[h]
        n = t.n[node]
        if n == 0 && ltn <= 0.0
            criterion_value = t.v[node]
        elseif n == 0 && t.v[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = t.v[node] + crit.c*sqrt(ltn/n)
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    ha = rand(rng, best_nodes)
    return ha
end