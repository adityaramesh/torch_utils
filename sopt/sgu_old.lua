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

	local isnan = function(x) return x ~= x end

	if mom_type == sopt.none then
		local loss, grad = func(x)
		x:add(-cur_lr, grad)

		local descent = -math.pow(grad:norm(), 2)
		local theta = 0

		local new_loss, new_grad = func(x, false)
		local eta_a = (new_loss - loss) / (cur_lr * descent)
		local eta_w = math.abs(grad:dot(new_grad) / descent)

		return x, {loss}, theta, eta_a, eta_w
	elseif mom_type == sopt.cm then
		assert(false)
		--local loss, grad = func(x)
		--if not state.vel then
		--	state.vel = torch.Tensor():typeAs(x):resizeAs(x):
		--		copy(grad):mul(-cur_lr)
		--else
		--	local cur_mom = mom(k)
		--	assert(cur_mom > 0 and cur_mom < 1)
		--	state.vel:mul(cur_mom):add(-cur_lr, grad)
		--end
		--x:add(state.vel)
		--return x, {loss}
	elseif mom_type == sopt.nag then
		if not state.temp then
			local loss, grad = func(x)
			state.temp = torch.Tensor():typeAs(x):resizeAs(x):
				copy(grad):mul(-cur_lr)
			x:add(state.temp)

			local descent = -math.pow(grad:norm(), 2)
			local theta = math.pi

			local new_loss, new_grad = func(x, false)
			local eta_a = (new_loss - loss) / (cur_lr * descent)
			local eta_w = math.abs(grad:dot(new_grad) / descent)

			return x, {loss}, theta, eta_a, eta_w, 0, 0, 0, 0, 0
		end

		-- Evaluate the function at the trial point.
		local cur_mom = mom(k)
		assert(cur_mom > 0 and cur_mom < 1)
		state.temp:mul(cur_mom):add(x)
		local loss, grad = func(state.temp)

		-- Update the parameters.
		state.temp:add(-1, x):add(-cur_lr, grad)
		x:add(state.temp)

		local descent = state.temp:dot(grad)
		local theta = math.acos(descent / (state.temp:norm() * grad:norm()))
		if isnan(theta) then
			theta = -1
		end

		local new_loss, new_grad = func(x, false)
		-- We treat the effective learning rate as one in this case.
		local eta_a = (new_loss - loss) / descent
		local eta_w = math.abs(state.temp:dot(new_grad) / descent)

		return x, {loss}, theta, eta_a, eta_w, 0, 0, 0, 0, 0
	else
		error("Invalid momentype type '" .. mom_type .. "'.")
	end
end
