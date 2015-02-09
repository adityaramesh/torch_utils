function sopt.rmsprop(func, x, config, state)
	local config   = config or {}
	local state    = state or config
	local eps      = config.epsilon or 1e-10
	local lr       = config.learning_rate or sopt.constant(1)
	local decay    = config.decay or sopt.constant(0.95)
	local mom      = config.momentum or sopt.sutskever_blend(0.999)
	local mom_type = config.momentum_type or sopt.none
	state.iter = state.iter or 0

	local k = state.iter
	local cur_lr = lr(k)
	local cur_decay = decay(k)
	state.iter = state.iter + 1

	if not state.temp then
		state.temp = torch.Tensor():typeAs(x):resizeAs(x):zero()
		state.exp_grad_fx = torch.Tensor():typeAs(x):resizeAs(x):zero()
	end

	if mom_type == sopt.none then
		local fx, grad_fx = func(x)

		state.temp:pow(grad_fx, 2):mul(1 - cur_decay)
		state.exp_grad_fx:mul(cur_decay):add(state.temp)
		state.temp:add(state.exp_grad_fx, eps):sqrt()
		x:addcdiv(-cur_lr, grad_fx, state.temp)
		return x, {fx}
	elseif mom_type == sopt.nag then
		local cur_mom = mom(k)
		cur_mom = 0

		-- Evaluate the function at the test point. At this time,
		-- `state.temp` contains the update from the previous iteration.
		state.temp:add(x, cur_mom, state.temp)
		local fx, grad_fx = func(state.temp)

		-- Update the parameters.
		state.temp:pow(grad_fx, 2):mul(1 - cur_decay)
		state.exp_grad_fx:mul(cur_decay):add(state.temp)
		state.temp:add(state.exp_grad_fx, eps):sqrt()
		state.temp:cdiv(grad_fx, state.temp):mul(-cur_lr)
		x:add(state.temp)
		return x, {fx}
	else
		error("Invalid momentum type '" .. mom_type .. "'.")
	end
end
