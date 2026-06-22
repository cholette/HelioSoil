"""
Nested-sampling inference for a multi-modal log-normal airborne-dust size
distribution from *binned* particle counts, using the negative-binomial +
multinomial likelihood of Section 3.4 (the PGM model) of the methodology notes.

Supports 1, 2, or 3 modes (set ``n_modes``).

Parametrisation
---------------
Each mode k = 1..K is described by an *amplitude* A_k (an expected count, the
N_i of the notes), a location mu_k = log10(mu) and a width sigma_k. The
expected counts in bin i (edges e_0 < ... < e_Nb in log10 D) are

    lambda_i = sum_k A_k [ Phi((e_{i+1}-mu_k)/sigma_k) - Phi((e_i-mu_k)/sigma_k) ]
             = diff_i  GaussianMixtureModel(A, mu, sigma).cumulative(edges)

and Lambda = sum_i lambda_i is the expected total count in the window. Using
amplitudes (rather than weights-on-the-simplex plus a separate total) removes
the need for a Dirichlet/stick-breaking prior. Each component gets an
informative, physically interpretable prior: Normal on the mean mu_k,
LogNormal on the width sigma_k and on the amplitude A_k (see ``Priors``). The
mixture CDF is evaluated with a single vectorised ``scipy.special.ndtr`` call
over the (Nb+1, K) grid -- no per-component Python loop and no object
construction -- since the forward model dominates the per-evaluation cost.

Likelihood (Section 3.4, Eq. joint_likelihood / joint_bin_counts)
-----------------------------------------------------------------
Marginalising a Gamma(r, r) multiplicative count error gives the closed form

    p(n | lambda) = NB(N_tot | r, r/(r+Lambda)) * Multinomial(n | lambda/Lambda, N_tot)

whose log, with N_tot = sum_i n_i, is

    log L = r ln r - lnGamma(r) + lnGamma(N_tot+r) - (N_tot+r) ln(r+Lambda)
            + sum_i [ n_i ln lambda_i - lnGamma(n_i+1) ].

Identifiability
---------------
* For K > 1 the mode labels are pinned by the *separated, informative Normal
  priors* on the means (default locations ~1.5 decades apart).
* r (NB dispersion) captures period-to-period total-count scatter and is only
  weakly identified from a *single* histogram, so it is fixed by default
  (fit_r=False). To estimate it, fit several histograms that share (A, mu,
  sigma) -- sum the per-period log-likelihoods (see `make_loglike` docstring).

Optional OPC broadening (Section 3.3): set fit_sigma_m to replace sigma_k by
sqrt(sigma_k^2 + sigma_m^2); set fit_delta for a per-period log10 shift of the
diameter scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln, ndtr, ndtri, log_ndtr  # raw normal CDF/inverse/log-CDF
# from scipy.stats import truncnorm                          # only the ordered-mean path


# --------------------------------------------------------------------------- #
#  Model configuration: how many modes, which nuisance params are free
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Defines the parameter vector for a K-mode fit.

    Parameter layout (matches GaussianMixtureModel.from_params ordering for the
    first 3K entries):

        A_1..A_K        amplitudes      (expected counts)
        mu_1..mu_K      locations       (log10 D, ordered)
        sigma_1..sigma_K widths         (log10 D)
        r               NB dispersion   (only if fit_r)
        sigma_m         OPC broadening  (only if fit_sigma_m)
        delta           period shift    (only if fit_delta)
    """
    n_modes: int = 3
    fit_r: bool = False
    fit_sigma_m: bool = False
    fit_delta: bool = False
    r_fixed: float = 1.0
    sigma_m_fixed: float = 0.0
    delta_fixed: float = 0.0
    names: list = field(default_factory=list)
    detected_amplitude: bool = False   # first K params are detected counts c_k (Sec. 3.5)

    def __post_init__(self):
        assert self.n_modes in (1, 2, 3), "n_modes must be 1, 2 or 3."
        K = self.n_modes
        amp = "c" if self.detected_amplitude else "A"
        n = ([f"{amp}{k+1}" for k in range(K)]
             + [f"mu{k+1}" for k in range(K)]
             + [f"sigma{k+1}" for k in range(K)])
        if self.fit_r:       n.append("r")
        if self.fit_sigma_m: n.append("sigma_m")
        if self.fit_delta:   n.append("delta")
        self.names = n

    @property
    def ndim(self):
        return len(self.names)

    def unpack(self, theta):
        """theta -> (A, mu, sigma, r, sigma_m, delta)."""
        theta = np.asarray(theta, float)
        K = self.n_modes
        A = theta[0:K]
        mu = theta[K:2 * K]
        sigma = theta[2 * K:3 * K]
        idx = 3 * K
        if self.fit_r:
            r = theta[idx]; idx += 1
        else:
            r = self.r_fixed
        if self.fit_sigma_m:
            sigma_m = theta[idx]; idx += 1
        else:
            sigma_m = self.sigma_m_fixed
        if self.fit_delta:
            delta = theta[idx]; idx += 1
        else:
            delta = self.delta_fixed
        return A, mu, sigma, r, sigma_m, delta


# --------------------------------------------------------------------------- #
#  Expected bin counts (vectorised ndtr forward model)
# --------------------------------------------------------------------------- #
def expected_bin_counts(edges, A, mu, sigma, sigma_m=0.0, delta=0.0):
    """lambda_i = expected counts per bin.

    Vectorised one ``scipy.special.ndtr`` call over the (Nb+1, K)
    grid, then a single (Nb+1, K) @ (K,) contraction with the amplitudes. Identical
    result to the GaussianMixtureModel class CDF. Broadening adds sigma_m in quadrature; 
    delta shifts the diameter scale. 
    """
    A = np.asarray(A, float)
    mu = np.asarray(mu, float)
    sig_eff = np.sqrt(np.asarray(sigma, float) ** 2 + sigma_m ** 2)
    e = np.asarray(edges, float) - delta
    z = (e[:, None] - mu[None, :]) / sig_eff[None, :]      # (Nb+1, K)
    F = ndtr(z) @ A                                        # (Nb+1,) amplitude-weighted CDF
    return np.diff(F)                                      # (Nb,) expected bin counts

def detected_fraction(edges, mu, sigma, sigma_m=0.0, delta=0.0):
    """f_k = Phi_k(upper) - Phi_k(lower) over the detection window
    [edges[0], edges[-1]] -- the fraction of mode k that is detectable."""
    se = np.sqrt(np.asarray(sigma, float) ** 2 + sigma_m ** 2)
    e  = np.asarray(edges, float) - delta
    mu = np.asarray(mu, float)
    return ndtr((e[-1] - mu) / se) - ndtr((e[0] - mu) / se)


def unpack_full(theta, edges, cfg, f_floor=1e-12):
    A, mu, sigma, r, sigma_m, delta = cfg.unpack(theta)
    if cfg.detected_amplitude:
        f = detected_fraction(edges, mu, sigma, sigma_m, delta)
        A = A / np.maximum(f, f_floor)      # floor avoids c/0 -> inf for invisible modes
    return A, mu, sigma, r, sigma_m, delta
# --------------------------------------------------------------------------- #
#  Log-likelihood
# --------------------------------------------------------------------------- #
def make_loglike(edges, counts, cfg: ModelConfig, penalty: float = -1e25):
    """Build the dynesty log-likelihood closure for one histogram.

    Multiple histograms: build one closure per period (shared cfg) and return
    their sum; e.g.

        lls = [make_loglike(edges, c, cfg) for c in counts_list]
        loglike = lambda th: sum(f(th) for f in lls)

    in which case fit_r=True becomes meaningful (r is shared across periods). 
    run_dynesty handles the single-histogram case automatically.
    """
    edges = np.asarray(edges, float)
    counts = np.asarray(counts, float)
    N_tot = counts.sum()
    pos = counts > 0
    const_data = -np.sum(gammaln(counts + 1.0))

    def loglike(theta):

        A, mu, sigma, r, sigma_m, delta = unpack_full(theta, edges, cfg)
        if np.any(A <= 0) or not np.all(np.isfinite(A)) or np.any(sigma <= 0) or r <= 0:
            return penalty
        lam = expected_bin_counts(edges, A, mu, sigma, sigma_m, delta)
        Lambda = lam.sum()

        if Lambda <= 0 or np.any(lam[pos] <= 0):
            return penalty

        nb = (r * np.log(r) - gammaln(r)
              + gammaln(N_tot + r) - (N_tot + r) * np.log(r + Lambda))
        poisson = np.sum(counts[pos] * np.log(lam[pos]))
        ll = nb + poisson + const_data
        return float(ll) if np.isfinite(ll) else penalty

    return loglike


def make_loglike_multi(edges, counts, cfg: ModelConfig, penalty: float = -1e25):
    """Joint log-likelihood for M histograms sharing (A, mu, sigma, r).

    counts : (M, Nb) stack of per-period bin counts (shared edges & volume).
    Each period has its own flow factor alpha_m ~ Gamma(r, r), marginalised
    inside its NB x multinomial term, so given the shared params the periods are
    independent and the log-likes add. The sum collapses to

        ll = M*(r ln r - lnGamma(r)) + sum_m lnGamma(N_m + r)
             - (N_all + M*r) ln(r + Lambda) + sum_i nbar_i ln(lambda_i) - const

    with N_m the per-period totals, N_all their sum, and nbar the POOLED bin
    counts. The shape (A, mu, sigma) is informed by nbar; r is informed only by
    the spread of the totals via sum_m lnGamma(N_m + r) -- hence r needs M > 1.
    Reduces exactly to make_loglike at M = 1.
    """
    counts = np.atleast_2d(np.asarray(counts, float))   # (M, Nb); promotes 1-D
    M      = counts.shape[0]
    N_m    = counts.sum(axis=1)                          # (M,) per-period totals
    N_all  = N_m.sum()
    nbar   = counts.sum(axis=0)                          # (Nb,) pooled counts
    pos    = nbar > 0
    const  = -np.sum(gammaln(counts + 1.0))

    def loglike(theta):
        A, mu, sigma, r, sigma_m, delta = unpack_full(theta, edges, cfg)
        if np.any(A <= 0) or not np.all(np.isfinite(A)) or np.any(sigma <= 0) or r <= 0:
            return penalty
        lam = expected_bin_counts(edges, A, mu, sigma, sigma_m, delta)
        Lambda = lam.sum()
        if Lambda <= 0 or np.any(lam[pos] <= 0):
            return penalty
        nb = (M * (r * np.log(r) - gammaln(r)) + np.sum(gammaln(N_m + r))
              - (N_all + M * r) * np.log(r + Lambda))
        poisson = np.sum(nbar[pos] * np.log(lam[pos]))
        ll = nb + poisson + const
        return float(ll) if np.isfinite(ll) else penalty

    return loglike

# --------------------------------------------------------------------------- #
#  Priors and prior transform
# --------------------------------------------------------------------------- #
def _normal_ppf(u, loc, scale):
    """Inverse-CDF of Normal(loc, scale) via the raw ``ndtri`` ufunc; loc/scale
    may be arrays."""
    return np.asarray(loc, float) + np.asarray(scale, float) * ndtri(u)


def _lognormal_ppf(u, median, sigma_ln):
    """Inverse-CDF of a LogNormal parametrised by its median and sigma_ln,
    the std of ln(value): value = median * exp(sigma_ln * Phi^{-1}(u)).
    median/sigma_ln may be arrays."""
    return np.asarray(median, float) * np.exp(np.asarray(sigma_ln, float) * ndtri(u))


@dataclass
class Priors:
    """Component-wise informative priors. Per-component entries are length-3
    tuples (fine, accumulation, coarse); the first ``n_modes`` are used.
 
    * mu_k     ~ Normal(mu_loc, mu_scale)            in log10(D)
    * sigma_k  ~ LogNormal(median=sigma_median, sigma_ln=sigma_lnsd)
    * A_k      ~ LogNormal(median=A_median, sigma_ln=A_lnsd)   (per-mode counts)
    * r        ~ log-uniform[10**log10_r_lo, 10**log10_r_hi]   (unchanged)
    * sigma_m  ~ LogNormal(sigma_m_median, sigma_m_lnsd)       (if fit_sigma_m)
    * delta    ~ Normal(0, delta_scale)                        (if fit_delta)
 
    Identifiability of the modes is handled by the separation of the mu priors
    """

    # Component means: Normal(loc, scale) in log10(D), fine/accumulation/coarse.
    mu_loc:   tuple = (-1.8, -0.3, 0.7)
    mu_scale: tuple = (0.3, 0.3, 0.3)
    
    # Component widths: LogNormal(median, sigma_ln) in log10(D).
    sigma_median: tuple = (0.25, 0.25, 0.25)
    sigma_lnsd:   tuple = (0.5, 0.5, 0.5)
    
    # Amplitudes (per-mode expected counts): LogNormal(median, sigma_ln), broad.
    A_median: tuple = (1.0e4, 1.0e4, 1.0e4)
    A_lnsd:   tuple = (1.0, 1.0, 1.0)      
    
    # NB dispersion r: log-uniform.
    log10_r_lo: float = -1.0
    log10_r_hi: float = 4.0
    
    # OPC broadening sigma_m: LogNormal.
    sigma_m_median: float = 0.05
    sigma_m_lnsd:   float = 0.5

    # Per-period diameter shift delta: Normal(0, scale) (cf. delta ~ N(0,sig_d)).
    delta_scale: float = 0.05
 
    @classmethod
    def set(cls, K, edges, n_expected=1.0e4, A_lnsd=1.0, sigma_median=0.28,
            sigma_lnsd=0.4, mu_scale_frac=0.5, margin=0.0, **over):
        """... margin (decades) lets the mode means sit up to `margin` beyond the
        detection window on each side (for the truncated case); margin=0 reproduces
        the in-window behaviour. Beyond ~2-3 typical sigma a mode is undetectable,
        so margin ~ 0.6-0.9 is usually plenty."""
        edges = np.asarray(edges, float)
        lo, hi = float(edges[0]) - margin, float(edges[-1]) + margin   # extend by margin
        span = hi - lo
        mu = np.linspace(lo, hi, K + 2)[1:-1]
        mu_scale = mu_scale_frac * span / (K + 1)
        kw = dict(
            mu_loc=tuple(mu), mu_scale=tuple(np.full(K, mu_scale)),
            sigma_median=tuple(np.full(K, sigma_median)),
            sigma_lnsd=tuple(np.full(K, sigma_lnsd)),
            A_median=tuple(np.full(K, n_expected / K)),
            A_lnsd=tuple(np.full(K, A_lnsd)),
        )
        kw.update(over)
        return cls(**kw)
    
def make_prior_transform(cfg: ModelConfig, pri: Priors):
    """Unit hypercube u in [0,1]^ndim -> parameter vector theta.

    Means: independent Normal(mu_loc, mu_scale) (needs separated priors). 
    Widths and amplitudes are component-wise log-normal; r is log-uniform.
    """
    K = cfg.n_modes
    mu_loc   = np.asarray(pri.mu_loc, float)[:K]
    mu_scale = np.asarray(pri.mu_scale, float)[:K]
    sig_med  = np.asarray(pri.sigma_median, float)[:K]
    sig_lnsd = np.asarray(pri.sigma_lnsd, float)[:K]
    A_med    = np.asarray(pri.A_median, float)[:K]
    A_lnsd   = np.asarray(pri.A_lnsd, float)[:K]

    def prior_transform(u):
        u = np.asarray(u, float)
        x = np.empty(cfg.ndim)
        # amplitudes: log-normal
        x[0:K] = _lognormal_ppf(u[0:K], A_med, A_lnsd)
        
        # means
        x[K:2 * K] = _normal_ppf(u[K:2 * K], mu_loc, mu_scale)
        
        # widths: log-normal
        x[2 * K:3 * K] = _lognormal_ppf(u[2 * K:3 * K], sig_med, sig_lnsd)

        idx = 3 * K
        if cfg.fit_r:
            x[idx] = 10.0 ** (pri.log10_r_lo + u[idx] * (pri.log10_r_hi - pri.log10_r_lo))
            idx += 1
        if cfg.fit_sigma_m:
            x[idx] = _lognormal_ppf(u[idx], pri.sigma_m_median, pri.sigma_m_lnsd)
            idx += 1
        if cfg.fit_delta:
            x[idx] = _normal_ppf(u[idx], 0.0, pri.delta_scale)
            idx += 1
        return x

    return prior_transform

# --------------------------------------------------------------------------- #
#  Sampler driver
# --------------------------------------------------------------------------- #
def run_dynesty(edges, counts, cfg: ModelConfig | None = None,
                pri: Priors | None = None, dynamic: bool = True,
                nlive: int = 500, seed: int | None = 0, **run_kwargs):
    """Fit with dynesty; returns the results object. Requires `pip install dynesty`.

        cfg = ModelConfig(n_modes=2)              # try 1, 2 or 3
        res = run_dynesty(edges, counts, cfg)
        eq = res.samples_equal()                  # equal-weight posterior

    Model comparison across K uses the log-evidence res.logz[-1].
    """
    import dynesty

    cfg = cfg or ModelConfig()
    pri = pri or Priors()
    counts = np.asarray(counts, float)
    
    if counts.ndim == 1:
        loglike = make_loglike(edges, counts, cfg)
    elif counts.ndim == 2:
        loglike = make_loglike_multi(edges, counts, cfg)
    else:
        raise ValueError("counts must be 1-D (one histogram) or 2-D (M, Nb).")
    
    ptform = make_prior_transform(cfg, pri)
    rstate = np.random.default_rng(seed)
    
    # bound='multi' + sample='rwalk': uniform-in-ellipsoid ('auto' at ndim<=10)
    # rejects badly on the correlated mixture ridges; random-walk is far cheaper
    # in likelihood calls here. Override via run_kwargs if needed.
    if dynamic:
        sampler = dynesty.DynamicNestedSampler(
            loglike, ptform, cfg.ndim, rstate=rstate,
            bound="multi", sample="rwalk")
        sampler.run_nested(nlive_init=nlive, **run_kwargs)
    else:
        sampler = dynesty.NestedSampler(
            loglike, ptform, cfg.ndim, nlive=nlive, rstate=rstate,
            bound="multi", sample="rwalk")
        sampler.run_nested(**run_kwargs)
    return sampler.results


# --------------------------------------------------------------------------- #
#  Synthetic-data helper
# --------------------------------------------------------------------------- #
def simulate_counts(edges, A, mu, sigma, r=None, sigma_m=0.0, delta=0.0, rng=None):
    rng = rng or np.random.default_rng()
    lam = expected_bin_counts(edges, A, mu, sigma, sigma_m, delta)
    lam = np.where(np.isfinite(lam), lam, 0.0)      # belt-and-suspenders vs inf/nan
    lam = np.clip(lam, 0.0, None)
    Lambda = lam.sum()
    if not np.isfinite(Lambda) or Lambda <= 0.0:    # whole draw outside the window
        return np.zeros(len(edges) - 1, dtype=int)
    p = lam / Lambda
    N_tot = rng.poisson(Lambda) if r is None else rng.negative_binomial(r, r / (r + Lambda))
    return rng.multinomial(int(N_tot), p)

def posterior_amplitudes(samples, edges, cfg):
    """Full amplitudes N_k for each posterior draw (S, K). Under
    detected_amplitude these are the extrapolated c_k / f_k, with the heavy tail
    discussed in Sec. 3.5; otherwise they're the sampled A_k unchanged."""
    return np.array([unpack_full(th, edges, cfg)[0] for th in samples])

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    edges = np.linspace(-2.2, 1.2, 25)
    A_true, mu_true, sig_true = [1.1e5, 7.0e4, 2.0e4], [-1.6, -0.4, 0.6], [0.25, 0.30, 0.25]
    counts = simulate_counts(edges, A_true, mu_true, sig_true, r=50.0, rng=rng)
    for K in (1, 2, 3):
        cfg = ModelConfig(n_modes=K)
        ll = make_loglike(edges, counts, cfg)
        theta = np.concatenate([A_true[:K], mu_true[:K], sig_true[:K]])
        print(f"K={K}: ndim={cfg.ndim:2d}  names={cfg.names}")
        print(f"      logL(partial truth) = {ll(theta):.2f}")
    print("N_tot =", int(counts.sum()))