AdaDeltaOptimizer = {}
AdaDeltaOptimizer.__index = AdaDeltaOptimizer

local function isnan(x)
	return x ~= x
end

--
-- Note: the `model` parameter here is unused, but kept anyway to preserve API
-- uniformity. Other optimization algorithms may need to use this parameter to
-- perform model-specific operations (e.g. disabling/enabling dropout).
--
function AdaDeltaOptimizer.create(model, params, grad_params, grad_func, state, logger)
	local self = {}
	setmetatable(self, AdaDeltaOptimizer)

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

-- Log function for AdaDelta without NAG.
-- TODO

-- Log function for AdaDelta with NAG.
-- TODO

function AdaDeltaOptimizer:update(input, target)
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
