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