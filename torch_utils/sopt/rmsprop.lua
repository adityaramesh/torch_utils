RMSPropOptimizer = {}
RMSPropOptimizer.__index = RMSPropOptimizer

local function isnan(x)
	return x ~= x
end

--
-- Note: the `model` parameter here is unused, but kept anyway to preserve API
-- uniformity. Other optimization algorithms may need to use this parameter to
-- perform model-specific operations (e.g. disabling/enabling dropout).
--
function RMSPropOptimizer.create(model, params, grad_params, grad_func, state, logger)
	local self = {}
	setmetatable(self, RMSPropOptimizer)

	self.params      = params
	self.grad_params = grad_params
	self.grad_func   = grad_func
	self.state       = state or {}

	self.state.iter = self.state.iter          or 0
	self.eps        = self.state.eps           or 1e-10
	self.lr         = self.state.learning_rate or sopt.constant(1e-3)
	self.mom        = self.state.momentum      or sopt.constant(0.95)
	self.decay      = self.state.decay         or sopt.constant(0.95)
	self.mom_type   = self.state.momentum_type or sopt.none

	if logger then
		self.logger = logger
		self.logger:add_fields({"loss", "norm_grad", "theta", "eta_a", "eta_w"})
	end
	return self
end

--
-- Log function for RMSProp without NAG.
--
function RMSPropOptimizer:log_info(input, target, cur_lr, loss)
	if not self.logger then return end

	if not self.prev_grad_params then
		self.prev_grad_params = torch.Tensor():typeAs(self.grad_params):
			resizeAs(self.grad_params)
	end

	local norm_grad = self.grad_params:norm()
	-- Note that descent := `1 / cur_lr * proj`. Because of cancellation
	-- with `cur_lr` that occurs in the formulas, we don't actually define
	-- it this way.
	local proj = -self.state.temp:dot(self.grad_params)
	local theta = math.acos(proj / (self.state.temp:norm() * norm_grad))

	self.prev_grad_params:copy(self.grad_params)
	local new_loss = self.grad_func(input, target)
	local eta_a = cur_lr * (new_loss - loss) / proj
	local eta_w = math.abs(-self.state.temp:dot(self.grad_params) / proj)

	self.logger:log_value("loss", loss)
	self.logger:log_value("norm_grad", norm_grad)
	self.logger:log_value("theta", theta)
	self.logger:log_value("eta_a", eta_a)
	self.logger:log_value("eta_w", eta_w)
end

--
-- Log function for RMSProp with NAG. Currently the implementation is exactly
-- the same as the one in `sgu.lua`.
--
function RMSPropOptimizer:log_nag_info(input, target, cur_lr, loss)
	if not self.logger then return end
	assert(self.state.prev_params ~= nil)
	assert(self.state.prev_grad_params ~= nil)

	-- See comment for analogous function in `sgu.lua` for more information
	-- regarding the computations performed below.

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

function RMSPropOptimizer:update(input, target)
	local iter = self.state.iter
	self.state.iter = self.state.iter + 1

	local cur_lr = self.lr(iter)
	local cur_decay = self.decay(iter)
	assert(cur_lr > 0 and cur_lr <= 1)
	assert(cur_decay > 0 and cur_decay < 1)

	-- Initializing the parameters here causes the first update to be
	-- multiplied by `(1 - cur_decay)`, since the running average of the
	-- second moment of the gradient will be zero. While it may seem like
	-- using a severe underestimate may impede convergence, I have actually
	-- found that the optimizer converges faster.
	if not self.state.temp then
		-- Used as a buffer to store intermediate results.
		self.state.temp = torch.Tensor():typeAs(self.params):
			resizeAs(self.params):zero()
		-- Estimate of the second moment of the gradient.
		self.state.grad_mom_2 = torch.Tensor():typeAs(self.params):
			resizeAs(self.params):zero()
	end

	if self.mom_type == sopt.none then
		local loss = self.grad_func(input, target, true)

		-- Update the estimate of the second moment of the gradient.
		self.state.temp:pow(self.grad_params, 2)
		self.state.grad_mom_2:mul(cur_decay):add(1 - cur_decay, self.state.temp)
		self.state.temp:add(self.state.grad_mom_2, self.eps):sqrt()

		self.params:addcdiv(-cur_lr, self.grad_params, self.state.temp)
		self:log_info(input, target, cur_lr, loss)
	elseif self.mom_type == sopt.nag then
		if not self.state.step then
			self.state.step = torch.Tensor():typeAs(self.params):
				resizeAs(self.params):zero()

			if self.logger then
				self.state.prev_params = self.params:clone()
				self.state.prev_grad_params = self.grad_params:clone()
			end
		end

		local cur_mom = self.mom(iter)
		assert(cur_mom > 0 and cur_mom < 1)

		-- Evaluate the function at the trial point.
		self.state.step:mul(cur_mom)
		self.params:add(self.state.step)
		local loss = self.grad_func(input, target, true)

		-- Update the estimate of the second moment of the gradient.
		self.state.temp:pow(self.grad_params, 2)
		self.state.grad_mom_2:mul(cur_decay):add(1 - cur_decay, self.state.temp)
		self.state.temp:add(self.state.grad_mom_2, self.eps):sqrt()
		self.state.temp:cdiv(self.grad_params, self.state.temp):mul(-cur_lr)

		-- Update the parameters.
		self.state.step:add(self.state.temp)
		self.params:add(self.state.temp)

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
