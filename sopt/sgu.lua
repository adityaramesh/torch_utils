SGUOptimizer = {}
SGUOptimizer.__index = SGUOptimizer

local function isnan(x)
	return x ~= x
end

--
-- Note: the `model` parameter here is unused, but kept anyway to preserve API
-- uniformity. Other optimization algorithms may need to use this parameter to
-- perform model-specific operations (e.g. disabling/enabling dropout).
--
function SGUOptimizer.create(model, params, grad_params, grad_func, config, logger)
	local self = {}
	setmetatable(self, SGUOptimizer)

	self.params      = params
	self.grad_params = grad_params
	self.grad_func   = grad_func
	self.state       = config or {}

	self.state.iter = self.state.iter          or 0
	self.lr         = self.state.learning_rate or sopt.constant(1e-3)
	self.mom        = self.state.momentum      or sopt.constant(0.95)
	self.mom_type   = self.state.momentum_type or sopt.none

	if logger then
		self.logger = logger
		assert(config.eig_estimator ~= nil)
		self.eig_estimator = config.eig_estimator

		self.logger:add_fields({"loss", "norm_grad", "theta", "eta_a",
			"eta_w", "min_eig", "max_eig", "phi_1", "phi_2",
			"eig_ratio", "proj_ratio"})
		self.prev_grad_params = torch.Tensor():typeAs(grad_params):
			resizeAs(grad_params)
		self.cur_grad_params = torch.Tensor():typeAs(grad_params):
			resizeAs(grad_params)
	end
	return self
end

function SGUOptimizer:compute_pre_eig_info(input, target)
	if self.logger then
		self.prev_eig_1, self.prev_v_1, self.prev_eig_2, self.prev_v_2 =
			self.eig_estimator:get_min_max_eig(input, target)
	end
end

function SGUOptimizer:compute_post_eig_info(input, target)
	if self.logger then
		self.cur_eig_1, self.cur_v_1, self.cur_eig_2, self.cur_v_2 =
			self.eig_estimator:get_min_max_eig(input, target)
	end
end

function SGUOptimizer:log_eig_info(prev_norm_grad)
	-- If any of the eigenvalue computations could not be completed to
	-- satisfactory accuracy, then we return early.
	if not (self.prev_eig_1 and self.prev_eig_2 and self.cur_eig_1 and
		self.cur_eig_2) then return end

	local cur_norm_grad = self.cur_grad_params:norm()
	local prev_proj     = self.prev_grad_params:dot(self.prev_v_1)
	local cur_proj      = self.cur_grad_params:dot(self.cur_v_1)

	local phi_1      = math.acos(prev_proj / prev_norm_grad)
	local phi_2      = math.acos(cur_proj / cur_norm_grad)
	local eig_ratio  = (self.cur_eig_2 - self.cur_eig_1) /
				(self.prev_eig_2 - self.prev_eig_1)
	local proj_ratio = cur_proj / prev_proj

	if isnan(phi_1) then phi_1 = -1 end
	if isnan(phi_2) then phi_2 = -1 end

	self.logger:log_value("min_eig", self.prev_eig_1)
	self.logger:log_value("max_eig", self.prev_eig_2)
	self.logger:log_value("phi_1", phi_1)
	self.logger:log_value("phi_2", phi_2)
	self.logger:log_value("eig_ratio", eig_ratio)
	self.logger:log_value("proj_ratio", proj_ratio)
end

function SGUOptimizer:log_sgu_info(input, target, cur_lr, loss)
	if self.logger then
		local prev_norm_grad = self.grad_params:norm()
		local descent = -math.pow(prev_norm_grad, 2)
		self.prev_grad_params:copy(self.grad_params)

		local new_loss = self.grad_func(input, target)
		self.cur_grad_params:copy(self.grad_params)
		self:compute_post_eig_info(input, target)

		local eta_a = (new_loss -  loss) / (cur_lr * descent)
		local eta_w = math.abs(self.prev_grad_params:dot(
			self.cur_grad_params) / descent)

		self.logger:log_value("loss", loss)
		self.logger:log_value("norm_grad", prev_norm_grad)
		self.logger:log_value("theta", math.pi)
		self.logger:log_value("eta_a", eta_a)
		self.logger:log_value("eta_w", eta_w)
		self:log_eig_info(prev_norm_grad)
	end
end

function SGUOptimizer:log_sgu_nag_info(input, target, cur_lr, loss)
	if self.logger then
		-- In order to allow for a more direct comparison to SGD, we
		-- make the following observation regarding the NAG update:
		-- 	s_{k + 1} := mu * s_k - eta * hat{g}_{k + 1}
		-- 	           = eta(mu / eta * s_k - hat{g}_{k + 1})
		-- 	          := eta * p_{k + 1}.
		-- Thus p_{k + 1} = (1 / eta) * s_{k + 1}. So we use p_{k + 1}
		-- instead of s_{k + 1} to compute the quantities that we log.
		local prev_norm_grad = self.grad_params:norm()
		local descent = 1 / cur_lr * self.state.temp:dot(self.grad_params)
		self.prev_grad_params:copy(self.grad_params)

		local new_loss = self.grad_func(input, target)
		self.cur_grad_params:copy(self.grad_params)
		self:compute_post_eig_info(input, target)

		-- Note: I folded the `1 / cur_lr` from the denominator into the
		-- numerator to clean things up.
		local theta = math.acos(cur_lr * descent / (self.state.temp:norm() *
			prev_norm_grad))

		-- If this happens, then either the update or the gradient has
		-- a very small magnitude. We report -1 to indicate that the
		-- angle could not be computed.
		if isnan(theta) then theta = -1 end

		local eta_a = (new_loss - loss) / descent
		local eta_w = math.abs(self.state.temp:dot(self.cur_grad_params) / descent)

		self.logger:log_value("loss", loss)
		self.logger:log_value("norm_grad", prev_norm_grad)
		self.logger:log_value("theta", theta)
		self.logger:log_value("eta_a", eta_a)
		self.logger:log_value("eta_w", eta_w)
		self:log_eig_info(prev_norm_grad)
	end
end

function SGUOptimizer:update(input, target)
	local iter = self.state.iter
	self.state.iter = self.state.iter + 1

	self:compute_pre_eig_info(input, target)
	local cur_lr = self.lr(iter)
	assert(cur_lr > 0)

	if self.mom_type == sopt.none then
		local loss = self.grad_func(input, target, true)
		self.params:add(-cur_lr, self.grad_params)
		self:log_sgu_info(input, target, cur_lr, loss)
	elseif self.mom_type == sopt.nag then
		if not self.state.temp then
			local loss = self.grad_func(input, target, true)
			self.state.temp = self.grad_params:clone():mul(-cur_lr)
			self.params:add(self.state.temp)
			self:log_sgu_info(input, target, cur_lr, loss)
			return
		end

		local cur_mom = self.mom(iter)
		assert(cur_mom > 0 and cur_mom < 1)

		-- Evaluate the function at the trial point.
		self.state.temp:mul(cur_mom)
		self.params:add(self.state.temp)
		local loss = self.grad_func(input, target, true)

		---- Update the parameters.
		self.state.temp:add(-cur_lr, self.grad_params)
		self.params:add(-cur_lr, self.grad_params)
		self:log_sgu_nag_info(input, target, cur_lr, loss)
	else
		error("Unsupported momentum type.")
	end
end
