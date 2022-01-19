using ImageFiltering, OffsetArrays # for moving average, https://stackoverflow.com/a/59589877/855050

function moving_average(x::AbstractVector, σ::Int = max(1, length(x) ÷ 20))
	# https://stackoverflow.com/a/59589877/855050
	kernel = OffsetArray(fill(1/(2σ + 1), 2σ + 1), -σ:σ)
	return imfilter(x, kernel)
end
