using Convex, SCS
using LinearAlgebra
using GLMakie


# Define the optimization variable
function p1(X, y)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)
    problem = minimize(L, [theta >= 0])
    solve!(problem, SCS.Optimizer; silent_solver=true)
    return theta.value
end



# Define another optimization variable but thiss time with constraints
# la contraintes est que theta>=0 et sum(theta)<=S
function p2(X, y, S)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)
    problem = minimize(L, [theta >= 0, sum(theta) <= S])
    solve!(problem, SCS.Optimizer; silent_solver=true)
    return theta.value
end


   

# faire varier S

# calculer lerreur valeur absolue entre theta_estime et theta_vraie
# et trace cette erreur en fct de S 


# tracer cette erreur en fct de S


# Trouver le S optimal qui minimise l'erreur
function S_optimal(X, y, theta_true)
    S_vals = 4.0:0.01:8.0
    errors = Float64[]
    for S in S_vals
        theta_est = p2(X, y, S)
        push!(errors, norm(theta_est - theta_true, 1))
    end
    S_opt = S_vals[argmin(errors)]
    min_error = minimum(errors)

    # tracer cette erreur en fct de S
    fig = Figure(resolution=(800,500))
    ax = Axis(fig[1,1], xlabel="S", ylabel="Erreur", title="Erreur en fonction de S")
    lines!(ax, S_vals, errors, label="Erreur ||theta_est - theta_true||")
    vlines!(ax, [S_opt], color=:red, linestyle=:dash, label="S optimal = $(round(S_opt,digits=2))")
    Legend(fig[1,2], ax)

    return S_opt, min_error, fig
end


function solve_p2_duale(X, y, S)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)

    problem = minimize(L, [theta >= 0, sum(theta) <= S])
    solve!(problem, SCS.Optimizer; silent_solver=true)

    # valeurs duales
    dual_value_inf = problem.constraints[1].dual
    dual_value_sum = problem.constraints[2].dual

    return dual_value_inf, dual_value_sum
end


# évolution des valeurs duales en fonction de S.
# tracer les valeurs duales en fonction de S.
# analyser les résultats obtenus.

function dual_values_vs_S(X, y, S_range)
    dual_values_inf = Float64[]
    dual_values_sum = Float64[]

     for S in S_range
        dual_inf_vec, dual_sum = solve_p2_duale(X, y, S)
        push!(dual_values_inf, sum(dual_inf_vec))
        push!(dual_values_sum, dual_sum)
    end

    # tracer les valeurs duales en fonction de S
    fig = Figure(resolution=(800,500))
    ax = Axis(fig[1,1], xlabel="S", ylabel="Valeurs duales", title="Valeurs duales en fonction de S")
    lines!(ax, S_range, dual_values_inf, label="Valeur duale inférieure")
    lines!(ax, S_range, dual_values_sum, label="Valeur duale somme")
    Legend(fig[1,2], ax)

    return fig
end