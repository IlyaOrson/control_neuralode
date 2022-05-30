# Feller, C., & Ebenbauer, C. (2014).
# Continuous-time linear MPC algorithms based on relaxed logarithmic barrier functions.
# IFAC Proceedings Volumes, 47(3), 2481–2488.
# https://doi.org/10.3182/20140824-6-ZA-1003.01022

function quadratic_relaxation(z, δ)
    return one(typeof(z)) / 2 * (((z - 2δ) / δ)^2 - one(typeof(z))) - log(δ)
end
exponential_relaxation(z, δ) = exp(one(typeof(z)) - z / δ) - one(typeof(z)) - log(δ)
function relaxed_log_barrier(z; δ=0.3f0)
    # return max(z > δ ? -log(z) : exponential_relaxation(z, δ), zero(typeof(z)))
    return z > δ ? -log(z) : exponential_relaxation(z, δ)
end
function relaxed_log_barrier(z, lower, upper; δ=(upper - lower) / convert(typeof(z), 2))
    return relaxed_log_barrier(z - lower; δ) + relaxed_log_barrier(upper - z; δ)
end

indicator_function(z; δ) = min(z, zero(typeof(z)))
quadratic_penalty(z; δ) = indicator_function(z; δ)^2
