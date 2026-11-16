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


def best_margin(ev, churned, monthly, econ, grid=MARGINS):
    """Cost-sensitive threshold: the EV margin that maximises realised profit (ties go to the margin nearest 0,
    the Bayes-optimal margin for calibrated probabilities)."""
    profits = [econ.realised(ev > m, churned, monthly).sum() for m in grid]
    best = max(profits)
    return float(min((m for m, p in zip(grid, profits, strict=True) if p == best), key=abs)), dict(
        zip(grid.tolist(), profits, strict=True)
    )
