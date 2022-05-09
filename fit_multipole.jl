using Optim, CSV, DataFrames, Plots, StatsPlots
import YAML
using ForwardDiff
plotlyjs()

n_fit = 4

function multipole(params::Vector{ComplexF64}, z::ComplexF64)
	v = params[1]
	npoles = (length(params) - 1) ÷ 2
	for i in 1:n_fit
		v += params[2i] / (z - params[2i+1])
	end
	return v
end
multipole(params::Vector{ComplexF64}, z::Vector{ComplexF64}) = map(zz -> multipole(params, zz), z)

function multipole_g!(storage, params::Vector{ComplexF64}, z::Vector{ComplexF64}, s::Vector{ComplexF64})
	storage[1] = sum(multipole(params, z) - s)
	for i in 1:n_fit
		gfa = (vz, vs) -> (multipole(params, vz) - vs) / conj(vz - params[2i+1])
		ga = sum(map(gfa, z, s))

		gfb = (vz, vs) -> (multipole(params, vz) - vs) * conj(params[2i] / (vz - params[2i+1])^2)
		gb = sum(map(gfb, z, s))

		storage[2i] = ga
		storage[2i+1] = gb
	end
end

function dyson(ϵ, Σc, precision=0.0001)
	E = ϵ
	for i in 1:10
		E = ϵ + Σc(E)
		if abs(E - ϵ) < precision * abs(ϵ)
			println("early stopping")
			break
		end
		if abs(E - ϵ) > abs(ϵ)*0.1
			println("Dyson diveging")
		end
	end
	return E
end

function initialize_params(n_fit)
	init_params = Array{ComplexF64}(undef, 1 + 2 * n_fit)
	init_params[1] = 0
	for i in 1:n_fit
		a = complex(i, 0)
		b = complex(i*0.5 * (-1)^i, -0.01)
		init_params[2i] = a
		init_params[2i+1] = b
	end
	return init_params
end

function read_qe_Σ(filename_re, filename_im)
	in_re = CSV.read(filename_re, DataFrame, header=["ω", "useless", "val", "useless2"]);
	in_im = CSV.read(filename_im, DataFrame, header=["ω", "useless", "val", "useless2"]);
	Σc = DataFrame();
	Σc.ω = complex.(0, in_re.ω);
	Σc.val = complex.(in_re.val, in_im.val);
	return Σc
end

firstpos = findfirst(ω -> imag(ω)>0, Σc.ω)
fitΣ = Σc[firstpos:firstpos+50, :];

params = initialize_params(n_fit)

loss = mpparams -> sum(abs2.(multipole(mpparams, fitΣ.ω) - fitΣ.val))
#loss_g! = (storage,mpparams) -> multipole_g!(storage, mpparams, fitΣ.ω, fitΣ.val)

g = y -> gradient(loss, y)[1]
function g!(storage, mpparams)
	res = g(mpparams)
	for i in 1:length(res)
		storage[i] = res[i]
	end
end

#od = OnceDifferentiable(loss, init_params; autodiff=:forward);
#res = optimize(loss, params, ConjugateGradient())
res = optimize(loss, g!, params, ConjugateGradient())
params = Optim.minimizer(res)

fitΣ.fitted = multipole(params, fitΣ.ω)
fitΣ.fitted_real = multipole(params, fitΣ.ω*-im)
@df fitΣ plot(imag.(:ω), real.(fitΣ.val))
@df fitΣ plot!(imag(:ω), imag.(fitΣ.val))
@df fitΣ plot!(imag(:ω), real.(fitΣ.fitted))
@df fitΣ plot!(imag(:ω), imag.(fitΣ.fitted))

@df fitΣ plot(imag.(:ω), real.(fitΣ.fitted_real), label="fit_real_real")
@df fitΣ plot!(imag(:ω), imag.(fitΣ.fitted_real), label="fit_real_im")


function main()
	energy_filename = "energy.yaml"
	energies = YAML.load_file(energy_filename)
	HOMO, LUMO, target = energies["HOMO"], energies["LUMO"], energies["target"]
	#-13.58874, 0.21149
	# -14.42306, 0.90053
	offset = (HOMO + LUMO) / 2
	target = complex(target - offset, 0)
	
	params = estimate_params()
	E = dyson(target, ω -> multipole(params, ω))
	E += offset
end