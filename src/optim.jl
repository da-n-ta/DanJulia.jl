using Convex, SCS
using LinearAlgebra
using GLMakie


# Define the optimization variable
function p1(X, y)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)
    problem = minimize(L, [theta >= 0])
    solve!(problem, SCS.Optimizer; silent=true)

    return evaluate(theta)

end


# la contraintes est que theta>=0 et sum(theta)<=S
function p2(X, y, S)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)
    problem = minimize(L, [theta >= 0, sum(theta) <= S])
    solve!(problem, SCS.Optimizer; silent=true)

    return evaluate(theta)

end

# affiche le graphique et l'erreur pour différentes valeurs de S
function p2_withS(X, y, S_values, theta_true)

    n = length(theta_true)
    indices = 1:n

    errors_dict = Dict{Float64, Vector{Float64}}()

    fig_errors = Figure()
    ax = Axis(fig_errors[1, 1],xlabel = "Index de θ", ylabel = "Erreur |θ_true - θ_est|", title = "Erreur entre θ_estimé et θ_vrai pour différentes valeurs de S")

    for S in S_values
        theta_estime = DanJulia.p2(X, y, S)
        errors = abs.(theta_estime .- theta_true)

        errors_dict[float(S)] = errors
        lines!(ax, indices, errors, label = "S = $S")
    end

    axislegend(ax)
    display(fig_errors)

    return fig_errors, errors_dict
end


# Trouver le S optimal qui minimise l'erreur
function S_optimal(X, y, theta_true)
    S_vals = 4.0:0.01:8.0
    errors = zeros(length(S_vals))
    for (i, S) in enumerate(S_vals)
        theta_est = p2(X, y, S)
        errors[i] = norm(theta_est - theta_true, 1)
    end
    S_opt = S_vals[argmin(errors)]
    min_error = minimum(errors)

    # tracer cette erreur en fct de S
    fig = Figure(resolution=(800,500))
    ax = Axis(fig[1,1], xlabel="S", ylabel="Erreur", title="Erreur en fonction de S")
    lines!(ax, S_vals, errors, label="Erreur ||theta_est - theta_true||")
    vlines!(ax, [S_opt], color=:red, linestyle=:dash, label="S optimal = $(round(S_opt,digits=2))")
    axislegend(ax)


    return S_opt, min_error, fig
end


function solve_p2_duale(X, y, S)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)

    problem = minimize(L, [theta >= 0, sum(theta) <= S])
    solve!(problem, SCS.Optimizer; silent=true)

    # valeurs duales
    dual_value_inf = problem.constraints[1].dual
    dual_value_sum = problem.constraints[2].dual

    @assert length(dual_value_sum) == 1
    dual_value_sum = dual_value_sum[1]

    return dual_value_inf, dual_value_sum
end


# évolution des valeurs duales en fonction de S.
# tracer les valeurs duales en fonction de S.
# analyser les résultats obtenus.

function dual_values_vs_S(X, y, S_range)
    # On suppose que solve_p2_duale(X, y, S) renvoie
    # dual_inf_vec (vector) et dual_sum (scalar)
    n = length(solve_p2_duale(X, y, S_range[1])[1])  # nombre de composantes
    dual_values_matrix = zeros(n, length(S_range))    # chaque ligne = une composante

    dual_values_sum = zeros(length(S_range))

    for (i, S) in enumerate(S_range)
        dual_inf_vec, dual_sum = solve_p2_duale(X, y, S)
        dual_values_matrix[:, i] .= dual_inf_vec       # chaque ligne = composante
        dual_values_sum[i] = dual_sum
    end

    # tracer les valeurs duales par composante
    fig = Figure(resolution=(800, 500))
    ax = Axis(fig[1,1], xlabel="S", ylabel="Valeurs duales", title="Valeurs duales en fonction de S")

    for j in 1:n
        lines!(ax, S_range, dual_values_matrix[j, :], label="Position duale $j")
    end

    # si tu veux aussi tracer la somme
    lines!(ax, S_range, dual_values_sum, color=:black, linestyle=:dash, label="Somme des duales")

    axislegend(ax)
    display(fig)

    return fig, dual_values_matrix, dual_values_sum
end


function plot_theta_comparison(theta_est, theta_true)
    fig = Figure(resolution=(800,400))
    ax = Axis(fig[1,1], xlabel="Index", ylabel="Value",
              title="θ_est vs θ_true")

    lines!(ax, theta_true, label="θ_true")
    lines!(ax, theta_est, label="θ_est")
    axislegend(ax)

    return fig
end


function residual_vs_S(X, y, S_vals)
    res = zeros(length(S_vals))

    for (i, S) in enumerate(S_vals)
        θ = p2(X, y, S)
        res[i] = norm(X*θ - y)
    end

    fig = Figure(resolution=(800,400))
    ax = Axis(fig[1,1], xlabel="S", ylabel="‖Xθ − y‖₂",
              title="Residual Norm vs S")
    
    lines!(ax, S_vals, res, label="‖Xθ − y‖₂") 
    axislegend(ax)

    return fig, res
end


function theta_path(X, y, S_vals)
    n = size(X, 2)
    path = zeros(n, length(S_vals))

    for (j, S) in enumerate(S_vals)
        path[:, j] = p2(X, y, S)
    end

    fig = Figure(resolution=(900,500))
    ax = Axis(fig[1,1], xlabel="S", ylabel="θᵢ",
              title="Regularization Path θᵢ(S)")

    for i in 1:n
        lines!(ax, S_vals, path[i, :], label="θ_$i")
    end

    axislegend(ax)
    return fig, path
end

function check_KKT(X, y, theta, dual_inf, dual_sum, S)
    r = X * theta - y

    # gradient of 1/2 ||Xθ - y||²: grad = Xᵀ r
    grad = X' * r

    # KKT stationarity condition:
    # grad + dual_inf + dual_sum * 1 = 0
    kkt_stationarity = grad .+ dual_inf .+ dual_sum

    return (
        grad = grad,
        kkt_stationarity = kkt_stationarity,
        comp_slackness_inf = dual_inf .* theta,
        comp_slackness_sum = dual_sum * (S - sum(theta)),
        primal_feasible = (minimum(theta) >= -1e-6) && (sum(theta) <= S + 1e-6),
        dual_feasible = (minimum(dual_inf) >= -1e-6) && (dual_sum >= -1e-6)
    )
end

