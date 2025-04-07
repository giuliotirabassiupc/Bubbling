using ChaosTools, CairoMakie, DelimitedFiles


N = 100000
Δt = 0.01
Ttr = 1000

a = 0.2
b = 0.2
c = 7.0

mins = 0
maxs = 0.2
Ns = 100
u0 = fill(0.0, 3)

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

lambdas = zeros(Ns, 3)
sigmas = LinRange(mins, maxs, Ns)
for i in eachindex(sigmas)
    s = sigmas[i]
    ds = CoupledODEs(rossler, u0, [a, b, c, s])
    tands = TangentDynamicalSystem(ds; J=rossler_jacob)
    lambdas[i, :] .= lyapunovspectrum(tands, N; Δt=Δt, u0=u0, Ttr=Ttr)
    print(s, " ", lambdas[i, :], "\n")
end

writedlm("MSF_rossler_(0.2, 0.2, 7.0).csv", hcat(sigmas, lambdas), '\t')

fig = Figure()
ax = Axis(fig[1, 1]; xlabel=L"sigma", ylabel=L"\lambda")
plot!(ax, sigmas, lambdas[:, 1])
display(fig)


