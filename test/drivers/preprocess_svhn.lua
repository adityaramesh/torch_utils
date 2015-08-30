require "torch"
require "image"
require "nn"

local raw_data_dir = "data/raw/"
local pp_data_dir  = "data/preprocessed/"
local train_file   = "train_small.t7"
local test_file    = "test.t7"

local raw_train_file = raw_data_dir .. train_file
local raw_test_file  = raw_data_dir .. test_file
local pp_train_file  = pp_data_dir  .. train_file
local pp_test_file   = pp_data_dir  .. test_file

local function preprocess(input_fn, output_fn, mean, stddev)
	print("Preprocessing `" .. paths.basename(input_fn) .. "`.")
	local data = torch.load(input_fn)
	local images = data.inputs:float()

	print("Converting color space from RGB to YUV.")
	for i = 1, images:size(1) do
		images[i] = image.rgb2yuv(images[i])
	end

	local compute_stats = false
	local channels = {'y', 'u', 'v'}

	print("Performing per-channel global normalization.")
	if mean == nil then
		mean = {}
		stddev = {}
		compute_stats = true
		for i, c in ipairs(channels) do
			mean[i] = images[{{}, i}]:mean()
			stddev[i] = images[{{}, i}]:std()
			images[{{}, i}]:add(-mean[i]):div(stddev[i])
		end
		assert(stddev[1] > 0)
		assert(stddev[2] > 0)
		assert(stddev[3] > 0)
	else
		assert(stddev[1] > 0)
		assert(stddev[2] > 0)
		assert(stddev[3] > 0)
		for i, c in ipairs(channels) do
			images[{{}, i}]:add(-mean[i]):div(stddev[i])
		end
	end
	
	print("Performing per-channel local normalization.")
	local kernel = image.gaussian1D(13)
	local module = nn.SpatialContrastiveNormalization(1, kernel, 1):float()

	for i = 1, images:size(1) do
		for c in ipairs(channels) do
			images[{i, {c}}] = module:forward(images[{i, {c}}])
		end
	end

	for i, c in ipairs(channels) do
		local new_mean = images[{{}, i}]:mean()
		local new_stddev = images[{{}, i}]:std()
		print("Mean of " .. c .. " channel: " .. new_mean)
		print("Standard deviation of " .. c .. " channel: " .. new_stddev)
	end

	local new_data = {inputs = images, targets = data.targets}
	torch.save(output_fn, new_data)
	if compute_stats then
		return mean, stddev
	end
end

local mean, stddev = preprocess(raw_train_file, pp_train_file)
preprocess(raw_test_file, pp_test_file, mean, stddev)
