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
function SGUOptimizer.create(model, params, grad_params, grad_func, state, logger)
	local self = {}
	setmetatable(self, SGUOptimizer)

	self.params      = params
	self.grad_params = grad_params
	self.grad_func   = grad_func
	self.state       = state or {}

	self.state.iter = self.state.iter          or 0
	self.lr         = self.state.learning_rate or sopt.constant(1e-3)
	self.mom        = self.state.momentum      or sopt.constant(0.95)
	self.mom_type   = self.state.momentum_type or sopt.none

	if logger then
		self.logger = logger
		self.logger:add_fields({"loss", "norm_grad", "theta", "eta_a", "eta_w"})
	end
	return self
end

--
-- Log function for SGU without NAG.
--
function SGUOptimizer:log_info(input, target, cur_lr, loss)
	if not self.logger then return end

	if not self.prev_grad_params then
		self.prev_grad_params = torch.Tensor():typeAs(self.grad_params):
			resizeAs(self.grad_params)
	end

	local norm_grad = self.grad_params:norm()
	-- In our case, descent := p_k g_k = -||g_k||^2.
	local descent = -math.pow(norm_grad, 2)
	self.prev_grad_params:copy(self.grad_params)
	local new_loss = self.grad_func(input, target)

	local eta_a = (new_loss -  loss) / (cur_lr * descent)
	local eta_w = math.abs(-self.grad_params:dot(
		self.prev_grad_params) / descent)

	self.logger:log_value("loss", loss)
	self.logger:log_value("norm_grad", norm_grad)
	self.logger:log_value("theta", math.pi)
	self.logger:log_value("eta_a", eta_a)
	self.logger:log_value("eta_w", eta_w)
end

--
-- Log function for SGU with NAG.
--
-- Precondition: `self.grad_params` is *not* modified.
-- Precondition: `self.state.step` =: s_k is defined such that
-- `x_{k + 1} = x_k + s_k`.
--
function SGUOptimizer:log_nag_info(input, target, cur_lr, loss)
	if not self.logger then return end
	assert(self.state.prev_params ~= nil)
	assert(self.state.prev_grad_params ~= nil)

	-- In order to allow for a more direct comparison to SGD, we
	-- make the following observation regarding the NAG update:
	-- 	s_{k + 1} :=  mu * s_k - eta * hat{g}_{k + 1}
	-- 	           =  eta(mu / eta * s_k - hat{g}_{k + 1})
	-- 	           =: eta * p_{k + 1}.
	-- So the analogous notion of "search direction" for SGU with NAG is
	-- p_{k + 1} = (1 / eta) * s_{k + 1}. Thus we use p_{k + 1} instead of
	-- s_{k + 1} to compute the quantities below.

	local norm_grad = self.state.prev_grad_params:norm()
	-- Note that descent := `1 / cur_lr * proj`. Because of cancellation
	-- with `cur_lr` that occurs in the formulas, we don't actually define
	-- it this way.
	local proj = self.state.step:dot(self.state.prev_grad_params)
	-- Note that theta could be NaN. If this happens, then either the update
	-- or the gradient has very small magnitude, so the angle could not be
	-- computed in single precision.
	local theta = math.acos(proj / (self.state.step:norm() * norm_grad))

	local new_loss = self.grad_func(input, target)
	local eta_a = cur_lr * (new_loss - loss) / proj
	local eta_w = math.abs(self.state.step:dot(self.grad_params) / proj)

	self.logger:log_value("loss", loss)
	self.logger:log_value("norm_grad", norm_grad)
	self.logger:log_value("theta", theta)
	self.logger:log_value("eta_a", eta_a)
	self.logger:log_value("eta_w", eta_w)
end

function SGUOptimizer:update(input, target)
	local iter = self.state.iter
	self.state.iter = self.state.iter + 1

	local cur_lr = self.lr(iter)
	assert(cur_lr > 0 and cur_lr <= 1)

	if self.mom_type == sopt.none then
		local loss = self.grad_func(input, target, true)
		self.params:add(-cur_lr, self.grad_params)
		self:log_info(input, target, cur_lr, loss)
	elseif self.mom_type == sopt.nag then
		-- For the first iteration, we just take the direction of
		-- steepest descent.
		if not self.state.step then
			local loss = self.grad_func(input, target, true)
			self.state.step = self.grad_params:clone():mul(-cur_lr)
			self.params:add(self.state.step)

			if self.logger then
				-- Unlike vanilla SGU, `prev_params` and
				-- `grad_params` need to be part of the state,
				-- since the logging function depends on their
				-- values and does not just use them as
				-- temporary buffers.
				self.state.prev_params = self.params:clone()
				self.state.prev_grad_params = self.grad_params:clone()
				self:log_nag_info(input, target, cur_lr, loss)
			end
			return
		end

		local cur_mom = self.mom(iter)
		assert(cur_mom > 0 and cur_mom < 1)

		-- Evaluate the function at the trial point.
		self.state.step:mul(cur_mom)
		self.params:add(self.state.step)
		local loss = self.grad_func(input, target, true)

		-- Update the parameters. We don't multiply the gradient by
		-- `-cur_lr` in advance because the logging function requires
		-- the original value.
		self.state.step:add(-cur_lr, self.grad_params)
		self.params:add(-cur_lr, self.grad_params)

		if self.logger then
			self:log_nag_info(input, target, cur_lr, loss)
			self.state.prev_params:copy(self.params)
			self.grad_func(input, target)
			self.state.prev_grad_params:copy(self.grad_params)
		end
	else
		error("Unsupported momentum type.")
	end
end
