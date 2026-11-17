"""Campaign economics: what a retention offer costs and what a saved customer is worth."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Economics:
    """Assumptions, not data: change them to match a real campaign."""

    offer_cost: float = 100.0  # $ per customer contacted (discount + outreach)
    save_rate: float = 0.30  # share of would-be churners the offer keeps
    horizon: int = 12  # months of revenue a saved customer is worth

    def expected_value(self, p, monthly):
        """Expected $ from contacting a customer: P(churn) x P(saved | churner) x their value, minus the offer."""
        return np.asarray(p) * self.save_rate * self.horizon * np.asarray(monthly) - self.offer_cost

    def realised(self, target, churned, monthly):
        """$ each targeting decision actually earns on labelled customers."""
        return np.asarray(target) * (np.asarray(churned) * self.save_rate * self.horizon * np.asarray(monthly) - self.offer_cost)

    def revenue_at_risk(self, p, monthly):
        """Expected revenue lost to churn over the horizon if nobody is contacted. Not profit: no save rate, no cost."""
        return float(np.sum(np.asarray(p) * self.horizon * np.asarray(monthly)))

    def dict(self):
        return asdict(self)


MARGINS = np.arange(-100, 101, 5.0)  # $ margins searched when tuning the decision threshold


def best_margin(ev, churned, monthly, econ, grid=MARGINS, b=2000, seed=0):
    """Cost-sensitive threshold. $0 is the Bayes-optimal margin for calibrated probabilities; the best margin on the
    grid replaces it only if it earns more with a paired-bootstrap 95% lower bound above zero, so the threshold never
    chases noise. Returns (margin, profit by margin, the test of best vs $0)."""
    profits = [econ.realised(ev > m, churned, monthly).sum() for m in grid]
    best = max(profits)
    tuned = float(min((m for m, p in zip(grid, profits, strict=True) if p == best), key=abs))
    gain = econ.realised(ev > tuned, churned, monthly) - econ.realised(ev > 0, churned, monthly)
    idx = np.random.default_rng(seed).integers(0, len(gain), (b, len(gain)))
    low = float(np.percentile(gain[idx].sum(axis=1), 2.5))
    test = {"best_margin": tuned, "gain_over_zero": round(float(gain.sum()), 0), "gain_ci_low": round(low, 0)}
    return (tuned if low > 0 else 0.0), dict(zip(grid.tolist(), profits, strict=True)), test
