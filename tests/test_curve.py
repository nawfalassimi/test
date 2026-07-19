from __future__ import annotations

import pytest

from fxbacktest.data.schema import TENOR_YEARS
from fxbacktest.data.synthetic import SyntheticFxDataGenerator
from fxbacktest.market.curve import FxForwardCurve


def test_implied_carry_round_trips_at_pillars():
    gen = SyntheticFxDataGenerator(start="2022-01-01", end="2022-06-30", seed=7)
    spot = gen.generate_spot_path()
    carry = gen.generate_carry_path()
    fwd_df = gen.generate_forward_points(spot, carry)

    date = spot.index[10]
    known_carry = carry.loc[date]
    day_fwd = fwd_df[fwd_df["date"] == date].set_index("tenor")["fwd_points"]
    curve = FxForwardCurve(spot=spot.loc[date], tenor_fwd_points=day_fwd.to_dict())

    for tenor, T in TENOR_YEARS.items():
        assert curve.implied_carry(T) == pytest.approx(known_carry, abs=1e-9)


def test_forward_interpolation_between_pillars_is_between_pillar_values():
    gen = SyntheticFxDataGenerator(start="2022-01-01", end="2022-06-30", seed=7)
    spot = gen.generate_spot_path()
    carry = gen.generate_carry_path()
    fwd_df = gen.generate_forward_points(spot, carry)

    date = spot.index[10]
    day_fwd = fwd_df[fwd_df["date"] == date].set_index("tenor")["fwd_points"]
    curve = FxForwardCurve(spot=spot.loc[date], tenor_fwd_points=day_fwd.to_dict())

    T_1m, T_3m = TENOR_YEARS["1M"], TENOR_YEARS["3M"]
    mid_T = (T_1m + T_3m) / 2
    f_1m, f_mid, f_3m = curve.forward(T_1m), curve.forward(mid_T), curve.forward(T_3m)
    assert min(f_1m, f_3m) <= f_mid <= max(f_1m, f_3m)
