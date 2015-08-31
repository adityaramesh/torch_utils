function sopt.constant(value)
	function f(iter)
		return value
	end
	return f
end

-- Used to decay the learning rate. This schedule is advocated by Leon Bottou in
-- "Stochastic Gradient Descent Tricks".
function sopt.gentle_decay(init, decay)
	function f(iter)
		return init / (1 + iter * decay)
	end
	return f
end

-- Used to decay momentum. See "On the importance of initialization and momentum
-- in deep learning", by Sutskever et al., for the reasoning behind why this
-- schedule is used.
function sopt.sutskever_blend(max, stretch)
	stretch = stretch or 250
	function f(iter)
		return math.min(1 - math.pow(2, -1 - math.log(
			math.floor(iter / stretch) + 1, 2)), max)
	end
	return f
end
