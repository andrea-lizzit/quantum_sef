modelparams = Model.init_params()

gen = Tweak(modelparams)

res = []
for params in gen.generate(10):
	model = Model(params)
	se = model(z)
	fmodel = Model.fit(se)
	params - fmodel.params
	res.append(params - fmodel.params)

print(res)