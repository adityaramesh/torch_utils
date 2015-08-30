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
		state.exp_update = torch.Tensor():typeAs(x):resizeAs(x):zero()
		state.exp_grad = torch.Tensor():typeAs(x):resizeAs(x):zero()
	end

	local isnan = function(x) return x ~= x end

	if mom_type == sopt.none then
		local loss, grad = func(x)

		state.temp:pow(grad, 2):mul(1 - cur_decay)
		state.exp_grad:mul(cur_decay):add(state.temp)

		-- Compute the update.
		state.exp_update:add(eps)
		state.exp_grad:add(eps)
		state.temp:cdiv(state.exp_update, state.exp_grad):sqrt():
			cmul(grad):mul(cur_lr)
		state.exp_update:add(-eps)
		state.exp_grad:add(-eps)
		x:add(-1, state.temp)

		local descent = -state.temp:dot(grad)
		local theta = math.acos(descent / (state.temp:norm() * grad:norm()))
		if isnan(theta) then
			theta = -1
		end

		local new_loss, new_grad = func(x, false)
		local eta_a = (new_loss - loss) / descent
		local eta_w = math.abs(state.temp:dot(new_grad) / descent)

		-- Update the decaying RMS of the updates.
		state.temp:pow(2):mul(1 - cur_decay)
		state.exp_update:mul(cur_decay):add(state.temp)
		return x, {loss}, theta, eta_a, eta_w
	elseif mom_type == sopt.nag then
		local cur_mom = mom(k)

		-- Evaluate the function at the test point.
		state.temp:add(x, cur_mom, state.exp_update)
		local loss, grad = func(state.temp)

		state.temp:pow(grad, 2):mul(1 - cur_decay)
		state.exp_grad:mul(cur_decay):add(state.temp)

		-- Compute the update.
		state.exp_update:add(eps)
		state.exp_grad:add(eps)
		state.temp:cdiv(state.exp_update, state.exp_grad):sqrt():
			cmul(grad):mul(cur_lr)
		state.exp_update:add(-eps)
		state.exp_grad:add(-eps)
		x:add(-1, state.temp)

		local descent = -state.temp:dot(grad)
		local theta = math.acos(descent / (state.temp:norm() * grad:norm()))
		if isnan(theta) then
			theta = -1
		end

		local new_loss, new_grad = func(x, false)
		local eta_a = (new_loss - loss) / descent
		local eta_w = math.abs(state.temp:dot(new_grad) / descent)

		-- Update the decaying RMS of the updates.
		state.temp:pow(2):mul(1 - cur_decay)
		state.exp_update:mul(cur_decay):add(state.temp):add(eps)
		return x, {loss}, theta, eta_a, eta_w
	else
		error("Invalid momentum type '" .. mom_type .. "'.")
	end
end
