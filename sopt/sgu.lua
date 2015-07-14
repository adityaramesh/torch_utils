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
		self.logger:add_fields({"loss", "norm_grad", "theta", "eta_a", "eta_w"})
		self.prev_grad_params = torch.Tensor():typeAs(grad_params):
			resizeAs(grad_params)
	end
	return self
end

function SGUOptimizer:log_sgu_info(input, target, cur_lr, loss)
	if self.logger then
		local norm_grad = self.grad_params:norm()
		local descent = -math.pow(norm_grad, 2)
		self.prev_grad_params:copy(self.grad_params)
		local new_loss = self.grad_func(input, target)

		local eta_a = (new_loss -  loss) / (cur_lr * descent)
		local eta_w = math.abs(self.prev_grad_params:dot(
			self.grad_params) / descent)

		self.logger:log_value("loss", loss)
		self.logger:log_value("norm_grad", norm_grad)
		self.logger:log_value("theta", math.pi)
		self.logger:log_value("eta_a", eta_a)
		self.logger:log_value("eta_w", eta_w)
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

		local norm_grad = self.grad_params:norm()
		local descent = 1 / cur_lr * self.state.temp:dot(self.grad_params)
		local theta = math.acos(descent / (1 / (cur_lr * cur_lr) *
			self.state.temp:norm() * norm_grad))

		-- If this happens, then either the update or the gradient has
		-- a very small magnitude. We report -1 to indicate that the
		-- angle could not be computed.
		if isnan(theta) then theta = -1 end

		local new_loss = self.grad_func(input, target)
		local eta_a = (new_loss - loss) / descent
		local eta_w = math.abs(self.state.temp:dot(self.grad_params) / descent)

		self.logger:log_value("loss", loss)
		self.logger:log_value("norm_grad", norm_grad)
		self.logger:log_value("theta", theta)
		self.logger:log_value("eta_a", eta_a)
		self.logger:log_value("eta_w", eta_w)
	end
end

function SGUOptimizer:update(input, target)
	local iter = self.state.iter
	self.state.iter = self.state.iter + 1

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
