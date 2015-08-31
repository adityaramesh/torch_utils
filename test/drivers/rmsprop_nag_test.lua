--
-- RMSProp with NAG plateaus very quickly but seems to have shitty behavior
-- close to convergence.
--

package.path = package.path .. ';source'
require "torch_utils/sopt"
require "test/models/cnn_5x5"
require "test/utility/run_model"

function get_train_info()
	return {
		opt_state = {
			learning_rate = sopt.constant(0.0001),
			momentum = sopt.constant(0.95),
			momentum_type = sopt.nag
		},
		opt_method = RMSPropOptimizer,
		batch_size = 100
	}
end

run(get_model_info, get_train_info)
