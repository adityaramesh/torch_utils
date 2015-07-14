--
-- Based on Xiang's code here:
-- https://github.com/zhangxiangxiao/GalaxyZoo/blob/master/model.lua
--
local function get_dropout_prob(model)
	for i, m in ipairs(model.modules) do
		if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
			return m.p
		end
	end
end

--
-- Taken from Xiang's code here:
-- https://github.com/zhangxiangxiao/GalaxyZoo/blob/master/model.lua
--
local function change_dropout_prob(model, prob)
	for i, m in ipairs(model.modules) do
		if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
			m.p = prob
		end
	end
end

local function local_grid_search(eig_func, iters, gran, tol, alpha)
	assert(gran > 0 and gran < 1)
	assert(alpha > 0 and alpha < 1)

	local a = math.max(alpha - 10 * gran, gran)
	local b = alpha + 10 * gran
	local best_alpha = 0
	local best_skew = 1
	local eig = 0

	for cur_alpha = a, b + gran, gran do
		local cur_skew, cur_eig = eig_func(cur_alpha, iters)

		if cur_skew < best_skew then
			if cur_skew < tol then
				return cur_alpha, cur_skew, cur_eig
			end

			best_alpha = cur_alpha
			best_skew = cur_skew
			eig = cur_eig
		end
	end

	return best_alpha, best_skew, eig
end

local function iterative_grid_search(eig_func, outer_iters, inner_iters, tol, alpha, skew)
	assert(tol > 0 and tol < 1)
	assert(alpha > 0 and alpha < 1)

	local exp = math.floor(math.log(alpha) / math.log(10))
	local gran = math.pow(10, exp - 1)
	local best_alpha = alpha
	local best_skew = skew
	local eig = 0

	for i = 1, outer_iters do
		local cur_alpha, cur_skew, cur_eig =
			local_grid_search(eig_func, inner_iters, gran, tol, best_alpha)

		if cur_skew < best_skew then
			if cur_skew < tol then
				return cur_alpha, cur_skew, cur_eig
			end

			best_alpha = cur_alpha
			best_skew = cur_skew
			eig = cur_eig
		end
		gran = gran / 10
	end

	return best_alpha, best_skew, eig
end

--
-- TODO: Refactor this later.
--
local function estimate_max_eigenvalue(grad_func, params, grad_params)
	-- Number of iterations of the power method.
	local inner_iters = 5

	-- This value of `tol` is actually quite forgiving. If `tol` is much
	-- higher than 1e-8, then the corresponding eigenvalue will only be
	-- correct to within an order of magnitude. I wouldn't recommend making
	-- this any smaller.
	local tol = 1e-6

	local alpha_list  = {
		-- For some reason, 6.56e-8 seems to give good results for a
		-- large fraction of the finite differences.
		6.56e-8,
		1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8,
		1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7,
		1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9, 9e-9,
		1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6,
		1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5
	}

	local phi        = torch.CudaTensor(params:size(1))
	-- TODO: Try setting `init_phi` to the current mini-batch gradient.
	local init_phi   = torch.randn(params:size(1)):cuda()
	local tmp_grad   = torch.CudaTensor(params:size(1))
	local tmp_params = params:clone()

	print("AAA: " .. grad_params:norm())

	local eig_func = function(alpha, iters)
		-- "Hot-starting" the iterations using the previous
		-- value of `phi` actually seems to retard progress.
		-- Instead, we use the saved value of `init_phi`.
		phi:copy(init_phi)

		local proj = 0
		local skew = 0
		local norm = 0

		for j = 1, iters do
			-- Because of finite precision arithmetic, we
			-- can't undo the action of adding `-alpha *
			-- phi` to `params` by subtracting the same
			-- quantity. Instead, we save `params` and
			-- restore it after bprop.
			params:add(-alpha, phi)
			grad_func(params, false)
			tmp_grad:copy(grad_params)

			params:copy(tmp_params)
			params:add(alpha, phi)
			grad_func(params, false)
			params:copy(tmp_params)
			grad_params:add(-1, tmp_grad):div(2 * alpha)

			-- How close is `phi` to being an eigenvector?
			norm = grad_params:norm()
			proj = phi:dot(grad_params) / norm
			skew = math.min(math.abs(proj - 1), math.abs(proj + 1))
			if skew < tol then
				break
			end

			phi:copy(grad_params)
			phi:div(norm)
		end

		norm = proj > 0 and norm or -norm
		return skew, norm
	end

	local best_alpha = 0
	local best_skew = 1
	local eig = 0

	local isnan = function(x) return x ~= x end

	-- Note: we might get this to run faster by reinitializing
	-- `init_phi` sooner if we aren't able to compute the maximum
	-- eigenvalue.
	local grid_search_func = function()
		local results = {}

		-- First try all of the `alpha` values that are in
		-- `alpha_list` for the finite difference.
		for j = 1, #alpha_list do
			local cur_alpha = alpha_list[j]
			local cur_skew, cur_eig = eig_func(cur_alpha, inner_iters)

			print(j .. ", " .. cur_skew)
			if not isnan(cur_skew) then
				results[cur_skew] = cur_alpha

				if cur_skew < best_skew then
					best_alpha = cur_alpha
					best_skew = cur_skew
					eig = cur_eig

					if best_skew < tol then
						break
					end
				end
			end
		end

		-- If we weren't able to find a good value for `alpha`,
		-- then run a nested grid search to try to hone in on a
		-- better value.
		if best_skew >= tol then
			local accuracies = {}
			for k in pairs(results) do
				table.insert(accuracies, k)
			end
			table.sort(accuracies)

			for j = 1, math.min(#accuracies, 5) do
				local cur_skew = accuracies[j]
				local cur_alpha = results[cur_skew]

				local new_alpha, new_skew, new_eig =
					iterative_grid_search(eig_func,
						3, inner_iters, tol,
						cur_alpha, cur_skew)

				if new_skew < best_skew then
					best_alpha = new_alpha
					best_skew = new_skew
					eig = new_eig

					if best_skew < tol then
						break
					end
				end
			end
		end
	end

	grid_search_func()

	-- If we still weren't able to compute the maximum eigenvalue
	-- using the nested grid search, then reinitialize `init_phi`
	-- and try again.
	if best_skew >= tol then
		for j = 1, 5 do
			best_alpha = 0
			best_skew = 1
			eig = 0

			init_phi:copy(torch.randn(params:size(1)))
			grid_search_func()

			if best_skew < tol then
				break
			end
		end
	end

	if best_skew < tol then
		-- TODO: log the eigenvalue.
		return eig, phi
	end

	print("Failed to find maximum eigenvalue during optimization.")
	print("Best skew: " .. best_skew .. ".")
	print("Best alpha: " .. best_alpha .. ".")
end

function sopt.sgu_eig_info(func, x, config, state, model, grad_params)
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
	local dropout_prob = get_dropout_prob(model)

	if mom_type == sopt.none then
		error("Not implemented yet.")
		--local loss, grad = func(x)
		--x:add(-cur_lr, grad)

		--local descent = -math.pow(grad:norm(), 2)
		--local theta = 0

		--local new_loss, new_grad = func(x, false)
		--local eta_a = (new_loss - loss) / (cur_lr * descent)
		--local eta_w = math.abs(grad:dot(new_grad) / descent)

		---- TODO update this.
		--return x, {loss}, theta, eta_a, eta_w
	elseif mom_type == sopt.cm then
		error("CM is not implemented for this optimization procedure.")
	elseif mom_type == sopt.nag then
		if not state.temp then
			change_dropout_prob(model, 0)
			local eig, eigvec = estimate_max_eigenvalue(func, x, grad_params)
			change_dropout_prob(model, dropout_prob)

			local loss = func(x)

			state.temp = torch.Tensor():typeAs(x):resizeAs(x):
				copy(grad_params):mul(-cur_lr)
			x:add(state.temp)

			local descent = -math.pow(grad_params:norm(), 2)
			local theta_1 = math.pi

			local new_loss, new_grad = func(x, false)
			local eta_a = (new_loss - loss) / (cur_lr * descent)
			local eta_w = math.abs(grad_params:dot(new_grad) / descent)

			local delta = eigvec:dot(grad_params)
			local theta_2 = math.acos(delta / grad_params:norm())
			if isnan(theta_2) then
				theta_2 = -1
			end

			--change_dropout_prob(model, 0)
			--local new_eig, new_eigvec = estimate_max_eigenvalue(func, x, new_grad)
			--change_dropout_prob(model, dropout_prob)

			--local eig_ratio = new_eig / eig
			--local delta_ratio = new_eigvec:dot(new_grad) / delta

			--return x, {loss}, theta_1, eta_a, eta_w, theta_2, eig_ratio, delta_ratio,
			--	eig, new_eig
			return x, {loss}, theta_1, eta_a, eta_w, theta_2, 0, 0, 0, 0
		end

		change_dropout_prob(model, 0)
		local eig, eigvec = estimate_max_eigenvalue(func, x, grad_params)
		change_dropout_prob(model, dropout_prob)

		-- Evaluate the function at the trial point.
		local cur_mom = mom(k)
		assert(cur_mom > 0 and cur_mom < 1)
		state.temp:mul(cur_mom):add(x)
		local loss = func(state.temp)
		print("A1: " .. grad_params:norm())

		-- Update the parameters.
		state.temp:add(-1, x):add(-cur_lr, grad_params)
		x:add(state.temp)

		local descent = state.temp:dot(grad_params)
		local theta_1 = math.acos(descent / (state.temp:norm() * grad_params:norm()))
		if isnan(theta_1) then
			theta = -1
		end

		local new_loss, new_grad = func(x, false)
		-- We treat the effective learning rate as one in this case.
		local eta_a = (new_loss - loss) / descent
		local eta_w = math.abs(state.temp:dot(new_grad) / descent)

		local delta = eigvec:dot(grad_params)
		local theta_2 = math.acos(delta / grad_params:norm())
		if isnan(theta_2) then
			theta_2 = -1
		end

		--change_dropout_prob(model, 0)
		--local new_eig, new_eigvec = estimate_max_eigenvalue(func, x, new_grad)
		--change_dropout_prob(model, dropout_prob)
		---- XXX
		--new_loss, new_grad = func(x, false)

		--local eig_ratio = new_eig / eig
		--local delta_ratio = new_eigvec:dot(new_grad) / delta

		--return x, {loss}, theta_1, eta_a, eta_w, theta_2, eig_ratio, delta_ratio,
		--	eig, new_eig
		return x, {loss}, theta_1, eta_a, eta_w, theta_2, 0, 0, 0, 0
	else
		error("Invalid momentype type '" .. mom_type .. "'.")
	end
end
