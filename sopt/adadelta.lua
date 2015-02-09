function sopt.adadelta(func, x, config, state)
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
		state.temp = torch.Tensor():typeAs(x):resizeAs(x)
		-- We add `eps` to the running expectations of the update and
		-- gradient preemptively, and subtract `eps` right after we
		-- compute the step for each iteration.  It is important that we
		-- subtract `eps` from the expectations at iteration `k` before
		-- folding them into the expectations at iteration `k + 1`.
		-- Otherwise, convergence is detrimented. Since the expected
		-- update lags behind by one iteration, we initialize it with a
		-- value of `eps` here.
		state.exp_update = torch.Tensor():typeAs(x):resizeAs(x):fill(eps)
		state.exp_grad_fx = torch.Tensor():typeAs(x):resizeAs(x):fill(0)
	end

	if mom_type == sopt.none then
		local fx, grad_fx = func(x)

		-- Update the parameters.
		state.temp:pow(grad_fx, 2):mul(1 - cur_decay)
		state.exp_grad_fx:mul(cur_decay):add(state.temp):add(eps)
		state.temp:cdiv(state.exp_update, state.exp_grad_fx):sqrt():
			cmul(grad_fx)
		state.exp_update:add(-eps)
		state.exp_grad_fx:add(-eps)
		x:add(-cur_lr, state.temp)

		-- Update the decaying RMS of the updates.
		state.temp:pow(2):mul(1 - cur_decay)
		state.exp_update:mul(cur_decay):add(state.temp):add(eps)
		return x, {fx}
	elseif mom_type == sopt.nag then
		local cur_mom = mom(k)

		-- Evaluate the function at the test point.
		state.temp:add(x, cur_mom, state.exp_update)
		local fx, grad_fx = func(state.temp)

		-- Update the parameters.
		state.temp:pow(grad_fx, 2):mul(1 - cur_decay)
		state.exp_grad_fx:mul(cur_decay):add(state.temp):add(eps)
		state.temp:cdiv(state.exp_update, state.exp_grad_fx):sqrt():
			cmul(grad_fx)
		state.exp_update:add(-eps)
		state.exp_grad_fx:add(-eps)
		x:add(-cur_lr, state.temp)

		-- Update the decaying RMS of the updates.
		state.temp:pow(2):mul(1 - cur_decay)
		state.exp_update:mul(cur_decay):add(state.temp):add(eps)
		return x, {fx}
	else
		error("Invalid momentum type '" .. mom_type .. "'.")
	end
end
