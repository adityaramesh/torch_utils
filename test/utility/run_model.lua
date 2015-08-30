require "torch"
require "optim"
require "xlua"

require "torch_utils/utility"

local function do_train_epoch(data, context, paths, info)
	local model       = info.model.model
	local criterion   = info.model.criterion
	local params      = context.params
	local grad_params = context.grad_params
	local confusion   = context.confusion

	local train_size  = data.inputs:size(1)
	local opt_method  = info.train.opt_method
	local opt_state   = info.train.opt_state
	local batch_size  = info.train.batch_size

	if info.train.epoch == nil then
		info.train.epoch = 1
	end

	local perm = torch.randperm(train_size)
	model:training()

	print("Starting training epoch " .. info.train.epoch .. ".")
	for i = 1, train_size, batch_size do
		xlua.progress(i, train_size)

		-- Create the mini-batch.
		local cur_batch_size = math.min(batch_size, train_size - i + 1)
		local inputs = {}
		local targets = {}
		for j = i, i + cur_batch_size - 1 do
			local input = data.inputs[{{perm[j]}}]
			local target = data.targets[{{perm[j]}}]
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		inputs = nn.JoinTable(1):forward(inputs):cuda()
		targets = nn.JoinTable(1):forward(targets):cuda()

		-- Create function to obtain the model's output and gradient
		-- with respect to parameters.
		local f = function(x)
			if x ~= params then
				params:copy(x)
			end
			grad_params:zero()

			local outputs = model:forward(inputs)
			local loss = criterion:forward(outputs, targets)
			model:backward(inputs, criterion:backward(outputs, targets))
			confusion:batchAdd(outputs, targets)
			return loss, grad_params
		end

		opt_method(f, params, opt_state)

		if i % 33 == 0 then
			-- This helps prevent out of memory errors on the GPU.
			collectgarbage()
		end
	end

	xlua.progress(train_size, train_size)
	confusion:updateValids()
	local acc = confusion.totalValid
	print("Mean class accuracy (training): " .. 100 * acc .. "%.")
	confusion:zero()

	model_utils.save_train_progress(function(x, y) return x > y end,
		info.train.epoch, acc, paths, info)
	info.train.epoch = info.train.epoch + 1
end

local function do_valid_epoch(data, context, paths, info)
	local model      = info.model.model
	local criterion  = info.model.criterion
	local valid_size = data.inputs:size(1)
	local batch_size = info.train.batch_size
	local confusion  = context.confusion
	model:evaluate()

	print("Performing validation epoch.")
	for i = 1, valid_size, batch_size do
		xlua.progress(i, valid_size)

		-- Create the mini-batch.
		local cur_batch_size = math.min(batch_size, valid_size - i + 1)
		local inputs = {}
		local targets = {}
		for j = i, i + cur_batch_size - 1 do
			local input = data.inputs[{{j}}]
			local target = data.targets[{{j}}]
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		inputs = nn.JoinTable(1):forward(inputs):cuda()
		targets = nn.JoinTable(1):forward(targets):cuda()
		local outputs = model:forward(inputs)
		confusion:batchAdd(outputs, targets)

		if i % 33 == 0 then
			-- This helps prevent out of memory errors on the GPU.
			collectgarbage()
		end
	end

	xlua.progress(valid_size, valid_size)
	confusion:updateValids()
	local acc = confusion.totalValid
	print("Mean class accuracy (validation): " .. 100 * acc .. "%.")
	confusion:zero()

	if info.train.epoch ~= nil then
		model_utils.save_test_progress(function(x, y) return x > y end,
			info.train.epoch - 1, acc, paths, info)
	end
end

function run(model_info_func)
	print("Loading data.")
	local data_dir = "data/preprocessed/"
	local train_data = torch.load(data_dir .. "train_small.t7")
	local valid_data = torch.load(data_dir .. "test.t7")

	local do_train, _, paths, info = model_utils.restore(
		model_info_func, get_train_info, optimization_options)

	print("Configuration options:")
	print(info.options)

	if info.train.epoch ~= nil then
		info.train.epoch = info.train.epoch + 1
	end

	local context = {}
	local max_epochs = info.options.max_epochs or 1000
	context.params, context.grad_params = info.model.model:getParameters()
	context.confusion = optim.ConfusionMatrix(10)

	print("")
	if do_train then
		while info.train.epoch == nil or info.train.epoch <= max_epochs do
			do_train_epoch(train_data, context, paths, info)
			print("")

			local cur_epoch = info.train.epoch - 1
			if cur_epoch % 3 == 0 or cur_epoch >= max_epochs - 5 then
				do_valid_epoch(valid_data, context, paths, info)
				print("")
			end
		end
	else
		do_valid_epoch(valid_data, context, paths, info)
	end
end
