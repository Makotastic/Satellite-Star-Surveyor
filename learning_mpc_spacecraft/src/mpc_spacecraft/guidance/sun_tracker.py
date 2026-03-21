from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeAlias, cast

import numpy as np
from astropy.coordinates import GCRS, get_sun
from astropy.time import Time

from mpc_spacecraft.simulation.clock import SimulationClock
from mpc_spacecraft.utilities.utils import Vec3, unit


TimeLike: TypeAlias = Time | datetime | str | float


@dataclass(slots=True)
class AstropySunDirectionModel:
    """Compute Sun direction in an Earth-centered inertial frame (GCRS).

    Returned vector is unit-length in ECI-like inertial coordinates.
    """

    clock: SimulationClock

    def sun_dir_eci(self, epoch_utc: TimeLike | None = None) -> Vec3:
        if epoch_utc is None:
            epoch_utc = self.clock.now()

        t = _to_astropy_time(epoch_utc)
        sun_gcrs = cast(Any, get_sun(t).transform_to(GCRS(obstime=t)))
        sun_vec = np.asarray(sun_gcrs.cartesian.xyz.to_value(), dtype=np.float64)
        return unit(sun_vec)


def _to_astropy_time(epoch_utc: TimeLike) -> Time:
    if isinstance(epoch_utc, Time):
        return epoch_utc

    if isinstance(epoch_utc, datetime):
        return Time(epoch_utc, scale="utc")

    if isinstance(epoch_utc, str):
        return Time(epoch_utc, scale="utc")

    raise TypeError("epoch_utc must be astropy.Time, datetime, str")
