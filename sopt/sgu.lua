function sopt.sgu(func, x, config, state)
	local config   = config or {}
	local state    = state or config
	local lr       = config.learning_rate or sopt.constant(1e-3)
	local mom      = config.momentum or sopt.sutskever_blend(0.999)
	local mom_type = config.momentum_type or sopt.none
	state.iter = state.iter or 0

	local k = state.iter
	local cur_lr = lr(k)
	state.iter = state.iter + 1
	assert(cur_lr > 0)

	if mom_type == sopt.none then
		local fx, grad_fx = func(x)
		x:add(-cur_lr, grad_fx)
		return x, {fx}
	elseif mom_type == sopt.cm then
		local fx, grad_fx = func(x)
		if not state.vel then
			state.vel = torch.Tensor():typeAs(x):resizeAs(x):
				copy(grad_fx):mul(-cur_lr)
		else
			local cur_mom = mom(k)
			assert(cur_mom > 0 and cur_mom < 1)
			state.vel:mul(cur_mom):add(-cur_lr, grad_fx)
		end
		x:add(state.vel)
		return x, {fx}
	elseif mom_type == sopt.nag then
		if not state.temp then
			local fx, grad_fx = func(x)
			state.temp = torch.Tensor():typeAs(x):resizeAs(x):
				copy(grad_fx):mul(-cur_lr)
			x:add(state.temp)
			return x, {fx}
		end

		-- Evaluate the function at the trial point.
		local cur_mom = mom(k)
		assert(cur_mom > 0 and cur_mom < 1)
		state.temp:mul(cur_mom):add(x)
		local fx, grad_fx = func(state.temp)

		-- Update the parameters.
		state.temp:add(-1, x):add(-cur_lr, grad_fx)
		x:add(state.temp)
		return x, {fx}
	else
		error("Invalid momentype type '" .. mom_type .. "'.")
	end
end
