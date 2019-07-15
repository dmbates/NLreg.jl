function all_names!(v::Set{Symbol}, expr::Expr)
    if expr.head == :call
        for a in expr.args[2:end]
            isa(a, Symbol) ? push!(v, a) : all_names!(v, a)
        end
    end
    v
end
all_names!(v::Set{Symbol}, expr::Real) = v 
all_names(expr::Expr) = all_names!(Set{Symbol}(), expr)

"""
    nlower(n)

Return `(n*(n+1))/2`, the number of elements in the lower triangle of a
square matrix of size `n`
"""
nlower(n) = (n * (n + 1)) >> 1
