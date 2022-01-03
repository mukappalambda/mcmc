from typing import Callable

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def main(func: Callable, maxiters: int = 150, n_samples: int = 100, epsilon: float = 1e-10) -> float:
  """Maximize a function using the Cross-Entropy Method (CEM)"""

  mu = 123
  sigma = 456
  t = 0
  Ne = 10

  while t < maxiters and sigma > epsilon:
    x = stats.norm.rvs(loc=mu, scale=sigma, size=n_samples)
    S = func(x)
    idx = np.argsort(S)
    x = x[idx[::-1]]
    mu = np.mean(x[:Ne])
    sigma = np.std(x[:Ne])
    t = t + 1

  return mu

if __name__ == "__main__":
  # first example
  func = lambda x: np.exp(-(x - 2) ** 2) + 0.8 * np.exp(- (x + 2) ** 2)
  mu = main(func=func)
  res = minimize(fun=lambda x: - func(x), x0=1)
  print(f"Example 1: true: {res.x[0]}; approximated: {mu}; close enough: {np.allclose(res.x[0], mu)}")

  # second example
  func = lambda x: stats.norm.pdf(x)
  mu = main(func=func)
  print(f"Example 2: true: {0}; approximated: {mu}; close enough: {np.allclose(0, mu)}")
