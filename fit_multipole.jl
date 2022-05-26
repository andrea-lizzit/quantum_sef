using Optim, CSV, DataFrames, Plots, StatsPlots
import YAML
using Zygote
plotlyjs()

function multipole(params::Vector{ComplexF64}, z::ComplexF64, npoles)
	v = params[1]
	#npoles = (length(params) - 1) ÷ 2
	for i in 1:npoles
		v += params[2i] / (z - params[2i+1])
	end
	return v
end
multipole(params::Vector{ComplexF64}, z::Vector{ComplexF64}, npoles) = map(zz -> multipole(params, zz, npoles), z)

function multipole_g!(storage, params::Vector{ComplexF64}, z::Vector{ComplexF64}, s::Vector{ComplexF64})
	storage[1] = sum(multipole(params, z) - s)
	npoles = (length(params) - 1) ÷ 2
	for i in 1:npoles
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

function initialize_params(n_poles)
	init_params = Array{ComplexF64}(undef, 1 + 2 * n_poles)
	init_params[1] = 0
	for i in 1:n_poles
		a = complex(i, 0)
		b = complex(i*0.5 * (-1)^i, -0.01)
		init_params[2i] = a
		init_params[2i+1] = b
	end
	return init_params
end

function read_qe_Σ(prefix, suffix)
	filename_re = prefix * "-re_on_im" * suffix
	filename_im = prefix * "-im_on_im" * suffix
	in_re = CSV.read(filename_re, DataFrame, header=["ω", "qefit", "val", "onreal"], delim=' ', ignorerepeated=true);
	in_im = CSV.read(filename_im, DataFrame, header=["ω", "qefit", "val", "onreal"], delim=' ', ignorerepeated=true);
	Σc = DataFrame();
	Σc.ω = complex.(0, in_re.ω);
	Σc.val = complex.(in_re.val, in_im.val);
	Σc.qefit = complex.(in_re.qefit, in_im.qefit);
	return Σc
end


function estimate_params(Σc, n_poles, params = nothing)
	if isnothing(params)
		params = initialize_params(n_poles)
	end

	loss = mpparams -> sum(abs2.(multipole(mpparams, Σc.ω, n_poles) - Σc.val))
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
	return params
end
function get_estimator(params, n_poles)
	estimator = ω -> multipole(params, ω, n_poles)
end

function plot_Σ(Σc, estimator; ϵ = nothing)
	Σc.fitted = estimator(Σc.ω)
	Σc.fitted_real = estimator(Σc.ω*-im)
	plot_im = @df Σc plot(imag.(:ω), [real.(Σc.val), imag.(Σc.val), real.(Σc.fitted), imag.(Σc.fitted), real.(Σc.qefit), imag.(Σc.qefit)], label=["reference real" "reference imaginary" "fit real" "fit imaginary" "qefit real" "qefit imaginary"])

	#@df Σc plot!(imag(:ω), imag.(Σc.val), label="reference imaginary")
	#@df Σc plot!(imag(:ω), real.(Σc.fitted), label="fit real")
	#@df Σc plot!(imag(:ω), imag.(Σc.fitted), label="fit imaginary")
	
	plot_re = @df Σc plot(imag.(:ω), [real.(Σc.fitted_real), imag.(Σc.fitted_real)], label=["fit_real_real" "fit_real_im"])
	if !isnothing(ϵ)
		line = map(x -> -ϵ + x, imag.(Σc.ω))
		plot!(plot_re, imag.(Σc.ω), line)
	end
	p = plot(plot_im, plot_re, layout=(2, 1))
	return p
	#@df Σc plot!(imag(:ω), imag.(Σc.fitted_real), label="fit_real_im")
end

println("starting main")
n_poles = 2;
n_fit = 120;
Σc = read_qe_Σ("methane/ch4", "00005");
fitΣ = begin
	firstpos = findfirst(ω -> imag(ω)>0, Σc.ω);
	Σc[firstpos:firstpos+n_fit, :];
end;
energies = YAML.load_file("methane/energies.yaml");
HOMO, LUMO, target = energies["HOMO"], energies["LUMO"], energies["target"];
#-13.58874, 0.21149
# -14.42306, 0.90053
offset = (HOMO + LUMO) / 2;
target = complex(target - offset, 0)

params = estimate_params(fitΣ, n_poles);
nparams = estimate_params(fitΣ, n_poles, params)
estimator = get_estimator(params, n_poles)
p = plot_Σ(Σc, estimator)
plot!(size=(960,1080))
E = dyson(target, estimator)
E += offset
println(E)
