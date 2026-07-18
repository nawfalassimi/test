from __future__ import annotations

import pytest

from fxbacktest.execution.transaction_costs import (
    OptionCostSpec,
    PairCostSpec,
    SpotCostSpec,
    TransactionCostModel,
)


def test_zero_model_has_no_cost():
    model = TransactionCostModel.zero()
    assert model.option_cost("EURUSD", "buy", fair_price=1000.0, vega=50000.0) == 0.0
    assert model.spot_cost("EURUSD", "sell", notional=1_000_000) == 0.0


def test_vol_spread_cost_scales_with_vega_and_sign():
    spec = PairCostSpec(option=OptionCostSpec(kind="vol_spread", vol_spread_bp=50.0))
    model = TransactionCostModel(by_pair={"EURUSD": spec})

    buy_cost = model.option_cost("EURUSD", "buy", fair_price=1000.0, vega=50000.0)
    sell_cost = model.option_cost("EURUSD", "sell", fair_price=1000.0, vega=50000.0)

    expected = 0.5 * (50.0 / 10_000.0) * 50000.0
    assert buy_cost == pytest.approx(expected)
    assert sell_cost == pytest.approx(-expected)


def test_premium_spread_cost_scales_with_premium_not_vega():
    spec = PairCostSpec(option=OptionCostSpec(kind="premium_spread", premium_spread_bp=20.0))
    model = TransactionCostModel(by_pair={"EURUSD": spec})

    buy_cost = model.option_cost("EURUSD", "buy", fair_price=10_000.0, vega=999_999.0)
    expected = 0.5 * (20.0 / 10_000.0) * 10_000.0
    assert buy_cost == pytest.approx(expected)


def test_spot_cost_scales_with_notional_and_pips():
    spec = PairCostSpec(spot=SpotCostSpec(spread_pips=2.0))
    model = TransactionCostModel(by_pair={"USDJPY": spec})

    buy_cost = model.spot_cost("USDJPY", "buy", notional=1_000_000)
    expected = 0.5 * (2.0 / 10_000.0) * 1_000_000
    assert buy_cost == pytest.approx(expected)


def test_unlisted_pair_falls_back_to_default():
    default_spec = PairCostSpec(option=OptionCostSpec(kind="vol_spread", vol_spread_bp=75.0))
    model = TransactionCostModel(by_pair={"EURUSD": PairCostSpec()}, default=default_spec)

    cost = model.option_cost("GBPUSD", "buy", fair_price=1000.0, vega=10000.0)
    expected = 0.5 * (75.0 / 10_000.0) * 10000.0
    assert cost == pytest.approx(expected)


def test_from_config_builds_per_pair_and_default_specs():
    config = {
        "EURUSD": {"option": {"kind": "vol_spread", "vol_spread_bp": 50.0},
                   "spot": {"spread_pips": 1.0}},
        "USDJPY": {"option": {"kind": "premium_spread", "premium_spread_bp": 15.0},
                   "spot": {"spread_pips": 3.0}},
        "default": {"option": {"kind": "vol_spread", "vol_spread_bp": 100.0},
                    "spot": {"spread_pips": 5.0}},
    }
    model = TransactionCostModel.from_config(config)

    assert model.for_pair("EURUSD").option.kind == "vol_spread"
    assert model.for_pair("EURUSD").option.vol_spread_bp == 50.0
    assert model.for_pair("USDJPY").option.kind == "premium_spread"
    assert model.for_pair("USDJPY").option.premium_spread_bp == 15.0
    assert model.for_pair("GBPUSD").option.vol_spread_bp == 100.0  # falls back to default
    assert model.for_pair("GBPUSD").spot.spread_pips == 5.0


def test_unknown_cost_kind_raises():
    spec = PairCostSpec(option=OptionCostSpec(kind="bogus"))
    model = TransactionCostModel(by_pair={"EURUSD": spec})
    with pytest.raises(ValueError):
        model.option_cost("EURUSD", "buy", fair_price=1000.0, vega=1000.0)
