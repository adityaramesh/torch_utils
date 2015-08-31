require 'rake/clean'

model_dir = 'models'
pp_data_dir = "data/preprocessed"

directory model_dir
directory pp_data_dir

task :preprocess_data => pp_data_dir do
	sh "th test/drivers/preprocess_svhn.lua"
end

task :test_optimizer, [:name, :gpu] => model_dir do |t, args|
	sh "th test/drivers/#{args[:name]}_test.lua -device #{args[:gpu]} " \
		"-model #{args[:name]}_test -task replace"
end
