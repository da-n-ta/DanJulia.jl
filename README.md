# DanJulia

Petites fonctions d'optimisation (Convex + SCS) et visualisations (Makie) pour des exercices et démonstrations sur la régression quadratique sous contraintes (θ ≥ 0, somme ≤ S).

## Objectif
Fournir des outils simples pour :
- résoudre des problèmes de moindres carrés sous contraintes convexes,
- étudier l'effet d'une contrainte de somme sur la solution (path de régularisation),
- examiner les valeurs duales et vérifier les conditions KKT,
- visualiser les erreurs, résidus et trajectoires de paramètres.

## Prérequis
- Julia 1.12 (ou proche)
- Paquets : Convex, SCS, GLMakie (optionnel : CairoMakie pour export de figures)

Installation (depuis la racine du dépôt)
1. Cloner le dépôt :
   git clone https://github.com/da-n-ta/DanJulia.jl.git
   cd DanJulia.jl

2. Installer les dépendances et précompiler :
   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'


## Usage rapide
Lancer l'exemple fourni :
   julia --project=. exemple/try.jl

Dans REPL ou script :
```julia
using DanJulia
θ1 = p1(X, y)             # solution non négative (min L2)
θ2 = p2(X, y, 3.0)        # θ >= 0 et sum(θ) <= 3.0
fig, errs = p2_withS(X,y, 1.0:1.0:7.0, θ_true)
Sopt, errmin, figS = S_optimal(X,y, θ_true)
```
## Fonctions principales
p1(X, y) -> Vector{Float64}
Résout min_θ ||Xθ − y||₂² sous θ ≥ 0.

p2(X, y, S) -> Vector{Float64}
Même que p1 avec contrainte additionnelle sum(θ) ≤ S.

p2_withS(X, y, S_values, θ_true) -> (Figure, Dict)
Calcule θ pour plusieurs S, trace les erreurs |θ_true − θ_est| et retourne un dictionnaire d'erreurs.

S_optimal(X, y, θ_true) -> (S_opt, min_error, Figure)
Balaye S et renvoie la S qui minimise l'erreur (norme 1 par défaut), l'erreur minimale et la figure.

solve_p2_duale(X, y, S) -> (dual_inf_vector, dual_sum_scalar)
Résout p2 et renvoie les multiplicateurs (duaux) associés aux contraintes (θ ≥ 0 et sum ≤ S).

dual_values_vs_S(X, y, S_range) -> (Figure, matrix, vector)
Calcule et trace l'évolution des valeurs duales des composantes et du multiplicateur de somme en fonction de S.

plot_theta_comparison(θ_est, θ_true) -> Figure
Trace θ_est vs θ_true.

residual_vs_S(X, y, S_vals) -> (Figure, residuals)
Trace la norme du résidu ‖Xθ − y‖₂ en fonction de S.

theta_path(X, y, S_vals) -> (Figure, path_matrix)
Trace la trajectoire de chaque composante θᵢ(S) sur la grille S_vals.

check_KKT(X, y, θ, dual_inf, dual_sum, S) -> Dict-like résultat
Calcule gradient, vérifie stationnarité, conditions de complémentarité et fais quelques tests de faisabilité primal/dual.