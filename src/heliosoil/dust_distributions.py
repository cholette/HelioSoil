from enum import Enum
from typing import Union, Sequence, Tuple

import scipy.stats as sps
import scipy.optimize as spo
import numpy as np
from .utilities import _print_if
from openpyxl import load_workbook
from tqdm import tqdm

import matplotlib.pyplot as plt

NumberArray = Union[np.ndarray, Sequence[float]]


class DistributionKind(Enum):
    """Physical interpretation of a DustDistribution's weights."""

    NUMBER = "number"
    MASS = "mass"
    AREA = "area"


class GaussianMixtureModel:
    """
    Mixture of Gaussian distributions in log10(D/µm) space.

    Parameters are stored as numpy arrays. This class owns the mixture
    model machinery: density, cumulative distribution, mean, and inverse CDF.
    It is used as the internal representation inside DustDistribution.

    Args:
        weights: Component amplitudes, shape (N,). Units depend on the parent
                 distribution type (µg·m⁻³ for mass, cm⁻³ for number, etc.).
        mus:     Component means in log10(D/µm), shape (N,).
        sigmas:  Component standard deviations in log10(D/µm), shape (N,).
    """

    def __init__(self, weights: NumberArray, mus: NumberArray, sigmas: NumberArray) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.mus = np.asarray(mus, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)
        assert len(self.weights) == len(self.mus) == len(self.sigmas), "weights, mus, and sigmas must all have the same length."
        assert np.all(self.sigmas > 0), "All sigma values must be positive."

    @classmethod
    def from_params(cls, params: NumberArray) -> "GaussianMixtureModel":
        """
        Construct from a flat parameter array.

        Args:
            params: Flat array [w0,...,wN-1, mu0,...,muN-1, sig0,...,sigN-1].
        Returns:
            A new GaussianMixtureModel instance.
        """
        params = np.asarray(params, dtype=float)
        N = len(params) // 3
        assert len(params) == 3 * N, "params length must be divisible by 3."
        return cls(params[0:N], params[N : 2 * N], params[2 * N :])

    @property
    def n_components(self) -> int:
        return len(self.weights)

    def to_params(self) -> np.ndarray:
        """Return the flat parameter array [w0,..., mu0,..., sig0,...]."""
        return np.concatenate([self.weights, self.mus, self.sigmas])

    def density(self, log_d: NumberArray) -> np.ndarray:
        """
        Evaluate the mixture density at log10(D/µm) values.

        Args:
            log_d: log10(D/1µm) values at which to evaluate.
        Returns:
            Mixture density values, same shape as log_d.
        """
        return sum(w * sps.norm.pdf(log_d, loc=mu, scale=sig) for w, mu, sig in zip(self.weights, self.mus, self.sigmas))

    def cumulative(self, log_d: NumberArray) -> np.ndarray:
        """
        Evaluate the mixture cumulative distribution at log10(D/µm) values.

        Args:
            log_d: log10(D/1µm) values at which to evaluate.
        Returns:
            Mixture cumulative distribution values, same shape as log_d.
        """
        return sum(w * sps.norm.cdf(log_d, loc=mu, scale=sig) for w, mu, sig in zip(self.weights, self.mus, self.sigmas))

    def mean(self) -> float:
        """Weighted mean in log10(D/1µm) space."""
        return float(np.dot(self.weights / self.weights.sum(), self.mus))

    def icdf(self, p: float) -> float:
        """
        Inverse cumulative distribution via root-finding.

        Args:
            p: Probability in [0, 1].
        Returns:
            log10(D/µm) such that cumulative(log_d) == p.
        """
        (result,) = spo.fsolve(lambda x: self.cumulative(x) - p, self.mean())
        return float(result)

    def __repr__(self) -> str:
        lines = [f"GaussianMixtureModel ({self.n_components} component(s)):"]
        for i, (w, mu, sig) in enumerate(zip(self.weights, self.mus, self.sigmas)):
            lines.append(f"  [{i}]  weight={w:.4g},  mu={mu:.4g},  sigma={sig:.4g}")
        return "\n".join(lines)


class DustDistribution:
    """
    Base class for dust particle size distributions.

    Subclasses (NumberDistribution, MassDistribution, AreaDistribution) each
    set a class-level `kind` and provide `to_number`, `to_mass`, and `to_area`
    conversion methods that return new instances rather than mutating in place.

    This class should not be instantiated directly.

    Args:
        distribution: A GaussianMixtureModel (or compatible object) holding
                      the mixture parameters.
    """

    kind: Union[None, DistributionKind] = None  # overridden by each subclass

    _DENSITY_UNITS = {
        DistributionKind.MASS: r"$\frac{\mu g \cdot m^{-3}}{d(\log D)}$",
        DistributionKind.NUMBER: r"$\frac{\mathrm{cm}^{-3}}{d(\log D)}$",
        DistributionKind.AREA: r"$\frac{\mu m^2 \cdot cm^{-3}}{d(\log D)}$",
    }

    _CUMULATIVE_UNITS = {
        DistributionKind.MASS: r"$\mu g \cdot m^{-3}$",
        DistributionKind.NUMBER: r"$\mathrm{cm}^{-3}$",
        DistributionKind.AREA: r"$\mu m^2 \cdot cm^{-3}$",
    }

    def __init__(self, distribution: GaussianMixtureModel) -> None:
        self.distribution = distribution

    @classmethod
    def from_params(cls, params: NumberArray) -> "DustDistribution":
        """
        Construct from a flat parameter array.

        Args:
            params: Flat array [w0,...,wN-1, mu0,...,muN-1, sig0,...,sigN-1].
        Returns:
            A new instance of the calling subclass.
        """
        return cls(GaussianMixtureModel.from_params(params))

    # ------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------

    @property
    def units(self) -> str:
        """LaTeX units string for the density of the current distribution kind."""
        return self._DENSITY_UNITS.get(self.kind, "")

    @property
    def cumulative_units(self) -> str:
        """LaTeX units string for the cumulative of the current distribution kind."""
        return self._CUMULATIVE_UNITS.get(self.kind, "")

    # ------------------------------------------------------------------
    # Mixture model pass-throughs
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        return self.distribution.n_components

    def density(self, log_d: NumberArray) -> np.ndarray:
        """
        Evaluate the mixture density at log10(D/µm) values.

        Args:
            log_d: log10(D/µm) values at which to evaluate.
        Returns:
            Mixture density values, same shape as log_d.
        """
        return self.distribution.density(log_d)

    def cumulative(self, log_d: NumberArray) -> np.ndarray:
        """
        Evaluate the mixture cumulative distribution at log10(D/µm) values.

        Args:
            log_d: log10(D/µm) values at which to evaluate.
        Returns:
            Mixture cumulative distribution values, same shape as log_d.
        """
        return self.distribution.cumulative(log_d)

    def mean(self) -> float:
        """Weighted mean in log10(D/µm) space."""
        return self.distribution.mean()

    def icdf(self, p: float) -> float:
        """
        Inverse cumulative distribution via root-finding.

        Args:
            p: Probability in [0, 1].
        Returns:
            log10(D/µm) such that cumulative(log_d) == p.
        """
        return self.distribution.icdf(p)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _sse(distribution: GaussianMixtureModel, log_diameter_values: NumberArray, pm_values: NumberArray) -> float:
        """
        Sum of squared errors between the mixture CDF and empirical cumulative values.

        Static to prevent the optimiser from mutating the instance during fitting.
        """
        residuals = distribution.cumulative(log_diameter_values) - np.asarray(pm_values)
        return float(np.sum(residuals**2))

    @classmethod
    def fit(
        cls, params0: NumberArray, log_diameter_values: NumberArray, cumulative_values: NumberArray, tol: float = 1e-3
    ) -> Tuple["DustDistribution", spo.OptimizeResult]:
        """
        Fit a mixture to cumulative data using bound-constrained least squares.

        Call on a subclass to produce the correctly typed result, e.g.:
            dist, res = NumberDistribution.fit(params0, log_d, data)

        Args:
            params0:             Initial guess, flat array [w, mu, sigma].
            log_diameter_values: Sorted log10(D/µm) values.
            cumulative_values:   Empirical cumulative values at those points.
            tol:                 Lower bound on sigma for each component.
        Returns:
            Tuple of (fitted instance of the calling subclass, scipy OptimizeResult).
        """
        params0 = np.asarray(params0, dtype=float)
        N = len(params0) // 3
        assert len(params0) == 3 * N, "params0 length must be divisible by 3."

        def objective(params: np.ndarray) -> float:
            return DustDistribution._sse(GaussianMixtureModel.from_params(params), log_diameter_values, cumulative_values)

        lb = [0.0] * N + [-np.inf] * N + [tol] * N
        ub = [np.inf] * (3 * N)
        bnds = spo.Bounds(lb=lb, ub=ub, keep_feasible=True)
        res = spo.minimize(objective, params0, bounds=bnds, tol=1e-8)

        return cls(GaussianMixtureModel.from_params(res.x)), res

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write_to_file(self, file_name: str, sheet_name: str, kind: DistributionKind = None, rho: float = None, verbose: bool = True) -> None:
        """
        Write the distribution to an Excel file, converting kind if needed.

        Args:
            file_name:  Path to .xlsx file.
            sheet_name: Sheet containing dust parameters.
            kind:       DistributionKind to write. Defaults to the current kind.
            rho:        Particle density in g·cm⁻³, required for some conversions.
            verbose:    Whether to print status.
        """
        _print_if("Writing dust distribution to file " + file_name, verbose)

        if kind is None or kind == self.kind:
            target = self
        elif kind == DistributionKind.NUMBER:
            target = self.to_number(rho)
        elif kind == DistributionKind.MASS:
            target = self.to_mass(rho)
        elif kind == DistributionKind.AREA:
            target = self.to_area(rho)
        else:
            raise ValueError(f"Unrecognised kind: {kind}.")

        weight_str = ";".join(str(w) for w in target.distribution.weights)
        mu_str = ";".join(str(10**mu) for mu in target.distribution.mus)
        sig_str = ";".join(str(10**sigma) for sigma in target.distribution.sigmas)

        wb = load_workbook(file_name)
        ws = wb[sheet_name]
        for cell in ws["A"]:
            if cell.value == "N_size":
                ws.cell(row=cell.row, column=2).value = target.n_components
                ws.cell(row=cell.row, column=4).value = ""
            elif cell.value == "Nd":
                ws.cell(row=cell.row, column=2).value = weight_str
                ws.cell(row=cell.row, column=4).value = ""
            elif cell.value == "mu":
                ws.cell(row=cell.row, column=2).value = mu_str
                ws.cell(row=cell.row, column=4).value = ""
            elif cell.value == "sigma":
                ws.cell(row=cell.row, column=2).value = sig_str
                ws.cell(row=cell.row, column=4).value = ""
        wb.save(filename=file_name)
        wb.close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def make_probability(self) -> "DustDistribution":
        """
        Return a new instance with component weights normalised to sum to 1.
        """
        total = self.distribution.weights.sum()
        new_gmm = GaussianMixtureModel(self.distribution.weights / total, self.distribution.mus, self.distribution.sigmas)
        return type(self)(new_gmm)

    def sample(self, sample_volume):
        N = self.cumulative(np.inf)
        Δv = sample_volume
        N = sps.poisson.rvs(N * Δv)

        w = self.distribution.weights
        w /= sum(w)
        mus, sigmas = self.distribution.mus, self.distribution.sigmas
        n_components = len(w)
        comp = np.random.choice(n_components, N, p=w)
        samples = np.zeros(N)
        for ii in tqdm(range(N)):
            c = comp[ii]
            samples[ii] = sps.norm.rvs(loc=mus[c], scale=sigmas[c], size=1)
        return samples

    def plot(self, npts=1000, ax=None, lb=1e-4, ub=1.0 - 1e-4, mplkwds={}):

        if ax is None:
            fig, ax = plt.subplots()

        maxN = np.sum(self.distribution.weights)
        XL, XU = self.icdf(lb * maxN), self.icdf(ub * maxN)
        x = np.linspace(XL, XU, npts)

        ax.semilogx(10**x, self.density(x), **mplkwds)
        ax.set_xlabel("Diameter")
        ax.set_ylabel(f"Density {self.units}")

        return ax

    def __repr__(self) -> str:
        return f"{type(self).__name__} (kind={self.kind}, density units='{self.units}')\n{self.distribution}"


# ----------------------------------------------------------------------
# Concrete subclasses
# ----------------------------------------------------------------------


class NumberDistribution(DustDistribution):
    """
    Dust distribution whose weights are number concentrations (cm⁻³).

    Conversions:
        to_mass(rho)  →  MassDistribution   (requires particle density)
        to_area()     →  AreaDistribution
    """

    kind = DistributionKind.NUMBER

    def to_number(self, rho: float = None) -> "NumberDistribution":
        return self

    def to_mass(self, rho: float = None) -> "MassDistribution":
        """
        Convert to mass distribution.

        Args:
            rho: Particle density in g·cm⁻³.
        """
        assert isinstance(rho, float), "rho must be a scalar float."
        ln10 = np.log10(np.e)
        ws, mus, sigs = self.distribution.weights, self.distribution.mus, self.distribution.sigmas
        b = 2 * mus + 6 * sigs**2 / ln10
        new_mus = b / 2.0
        new_weights = ws * np.pi * rho / 6 * np.exp(-(mus**2 - 0.25 * b**2) / 2 / sigs**2)
        return MassDistribution(GaussianMixtureModel(new_weights, new_mus, sigs))

    def to_area(self) -> "AreaDistribution":
        """Convert to cross-sectional area distribution (µm²·cm⁻³)."""
        ln10 = np.log10(np.e)
        ws, mus, sigs = self.distribution.weights, self.distribution.mus, self.distribution.sigmas
        new_mus = mus + 2 * sigs**2 / ln10
        new_weights = ws * np.pi / 4 * np.exp(2 * mus / ln10 + 2 / (ln10**2) * sigs**2)
        return AreaDistribution(GaussianMixtureModel(new_weights, new_mus, sigs))


class MassDistribution(DustDistribution):
    """
    Dust distribution whose weights are mass concentrations (µg·m⁻³).

    Conversions:
        to_number(rho)  →  NumberDistribution  (requires particle density in g·cm⁻³)
        to_area(rho)    →  AreaDistribution     (requires particle density in g·cm⁻³)
    """

    kind = DistributionKind.MASS

    def to_number(self, rho: float = None) -> NumberDistribution:
        """
        Convert to number distribution.

        Args:
            rho: Particle density in g·cm⁻³.
        """
        assert isinstance(rho, float), "rho must be a scalar float."
        ln10 = np.log10(np.e)
        ws, mus, sigs = self.distribution.weights, self.distribution.mus, self.distribution.sigmas
        new_mus = mus - 3 * sigs**2 / ln10
        b = 2 * new_mus + 6 * sigs**2 / ln10
        new_weights = 6 * ws / np.pi / rho * np.exp((new_mus**2 - 0.25 * b**2) / 2 / sigs**2)
        return NumberDistribution(GaussianMixtureModel(new_weights, new_mus, sigs))

    def to_mass(self, rho: float = None) -> "MassDistribution":
        return self

    def to_area(self, rho: float = None) -> "AreaDistribution":
        """
        Convert to cross-sectional area distribution (routes through number).

        Args:
            rho: Particle density in g·cm⁻³.
        """
        return self.to_number(rho).to_area()


class AreaDistribution(DustDistribution):
    """
    Dust distribution whose weights are cross-sectional area concentrations (µm²·cm⁻³).

    Conversions:
        to_number()     →  NumberDistribution
        to_mass(rho)    →  MassDistribution    (requires particle density; routes via number)
    """

    kind = DistributionKind.AREA

    def to_number(self, rho: float = None) -> NumberDistribution:
        """Convert to number distribution."""
        ln10 = np.log10(np.e)
        ws, mus, sigs = self.distribution.weights, self.distribution.mus, self.distribution.sigmas
        new_mus = mus - 2 * sigs**2 / ln10
        new_weights = ws / np.pi * 4 * np.exp(-2 * new_mus / ln10 - 2 / (ln10**2) * sigs**2)
        return NumberDistribution(GaussianMixtureModel(new_weights, new_mus, sigs))

    def to_mass(self, rho: float = None) -> MassDistribution:
        """
        Convert to mass distribution (routes through number).

        Args:
            rho: Particle density in g·cm⁻³.
        """
        return self.to_number().to_mass(rho)

    def to_area(self, rho: float = None) -> "AreaDistribution":
        return self
