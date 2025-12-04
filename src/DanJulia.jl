module DanJulia
using Convex,SCS

# Generate random problem data
using Random,LinearAlgebra
Random.seed!(2)
n = 10
m = 20
sigma = 0.1
X = randn(m,n)
theta_true = rand(n)
xi = randn(m)
y = X * theta_true + sigma * xi

# Define the optimization variable
function p1(X, y)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)
    problem = minimize(L, [theta >= 0])
    solve!(problem, SCS.Optimizer)
    println("theta_estime = ", theta.value)
    return theta.value
end

p1(X, y)
println("theta_vraie = ", theta_true)

# Define another optimization variable but thiss time with constraints
# la contraintes est que theta>=0 et sum(theta)<=S
function p2(X, y, S)
    n = size(X, 2)
    theta = Variable(n)
    L = sumsquares(X * theta - y)
    problem = minimize(L, [theta >= 0, sum(theta) <= S])
    solve!(problem, SCS.Optimizer)
    println("theta_estime = ", theta.value)
    return theta.value
end
# pour S=3
p2(X, y, 3.0)
println("theta_vraie = ", theta_true)
   

# faire varier S
for S in 1.0:1.0:10.0
    
    println("Pour S = ", S)
    p2(X, y, S)
    println("theta_vraie = ", theta_true)
    
end
# calculer lerreur valeur absolue entre theta_estime et theta_vraie
# et trace cette erreur en fct de S 
using Plots
errors = []
S_values = 1.0:1.0:10.0
for S in S_values

    theta_estime = p2(X, y, S)
    error = norm(theta_estime - theta_true, 1)
    push!(errors, error)
end 
# tracer cette erreur en fct de S
plot(S_values, errors, xlabel="S", ylabel="Erreur (norme 1)", title="Erreur entre theta_estime et theta_vraie en fonction de S", legend=false)

# Trouver le S optimal qui minimise l'erreur
function S_optimal(X,y,theta_true)
    errors = []
    S_vals = 4.0:0.01:8.0
    for S in S_vals
        theta_est = p2(X,y,S)
        error = norm(theta_est - theta_true, 1)
        push!(errors, error)
    end
    S_opt = S_vals[argmin(errors)]
    print(S_opt," est la valeur de S qui minimise l'erreur.")
    print(", avec une erreur de ", minimum(errors))
end

S_optimal(X,y,theta_true)


end

