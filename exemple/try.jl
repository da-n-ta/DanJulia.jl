using GLMakie
using Random,LinearAlgebra
using DanJulia

Random.seed!(2)
n = 10
m = 20
sigma = 0.1
X = randn(m,n)
theta_true = rand(n)
xi = randn(m)
y = X * theta_true + sigma * xi

DanJulia.p1(X, y)
println("theta_vraie = ", theta_true)

# pour S=3
DanJulia.p2(X, y, 3.0)
println("theta_vraie = ", theta_true)

errors = []
S_values = 1.0:1.0:10.0
for S in S_values
    theta_estime = DanJulia.p2(X, y, S)
    error = norm(theta_estime - theta_true, 1)
    push!(errors, error)
end 

fig = Figure()
ax = Axis(fig[1, 1], xlabel="S", ylabel="Erreur (norme 1)", title="Erreur entre theta_estime et theta_vraie en fonction de S")
lines!(ax, S_values, errors)
display(fig)

#trouver le S optimal
DanJulia.S_optimal(X,y,theta_true)