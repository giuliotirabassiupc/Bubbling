using ChaosTools, CairoMakie, LinearAlgebra, DelimitedFiles

a = 0.42
b = 2.0
c = 4.0
mins = 0
maxs = 0.3
Ns = 100

function rossler(x, p, n)
    @inbounds x1, x2, x3 = x[1], x[2], x[3]
    @inbounds a, b, c = p[1], p[2], p[3]
    SVector(-x2 - x3, x1 + a * x2, b + x3 * (x1 - c))
end

function rossler_jacob(x, p, n)
    @inbounds x1, x2, x3 = x[1], x[2], x[3]
    @inbounds a, b, c, s = p[1], p[2], p[3], p[4]
    row1 = SVector(-s, -1, -1)
    row2 = SVector(1, a - s, 0)
    row3 = SVector(x3, 0, x1 - c - s)
    return vcat(row1', row2', row3')
end

orbits = readdlm("Rossler_orbits_(0.2, 0.2, 7.0).csv", '\t', Float64)

sigmas = LinRange(mins, maxs, Ns)
results = zeros(Ns, size(orbits, 1) + 1)
results[:, 1] .= sigmas
for (n, (oi, x0, y0, z0, T)) in enumerate(eachrow(orbits))
    lambdas = zeros(Ns)
    for i in eachindex(sigmas)
        s = sigmas[i]
        ds = CoupledODEs(rossler, [x0, y0, z0], [a, b, c, s])
        tands = TangentDynamicalSystem(ds; J=rossler_jacob)
        tr = trajectory(tands, T)
        monodromy = get_deviations(tands)
        floquets = eigvals(Array(monodromy))
        lyap = [log(abs(floquets[j])) / T for j in eachindex(floquets)]
        lambdas[i] = maximum(lyap)
        print(s, " ", lambdas[i], "\n")
    end
    results[:, n+1] .= lambdas
end

writedlm("Floquets_rossler_(0.2, 0.2, 7.0).csv", results, '\t')
