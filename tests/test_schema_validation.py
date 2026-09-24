"""Unit tests for serving/schemas.py — no Flink, no models required."""
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serving"))

import schemas


def test_parse_valid_event():
    ev, err = schemas.parse_event(json.dumps({"TransactionID": 1, "card1": 1000}))
    assert err is None and ev["TransactionID"] == 1


def test_parse_malformed_json_goes_to_dlq():
    ev, err = schemas.parse_event('{"TransactionID": 1, "card1"')
    assert ev is None and "json_parse_error" in err
    rec = json.loads(schemas.make_dlq_record('{"broken', err))
    assert rec["error"] == err and rec["raw"].startswith('{"broken')


def test_parse_non_object_rejected():
    for bad in ("[1,2,3]", '"str"', "42", "{}"):
        ev, err = schemas.parse_event(bad)
        assert ev is None, bad


def test_hyphen_identity_keys_normalized():
    ev, _ = schemas.parse_event(json.dumps({"TransactionID": 5, "id-01": -5.0,
                                            "id-31": "chrome"}))
    assert "id_01" in ev and "id_31" in ev and "id-01" not in ev


def test_coerce_and_complete_fills_missing_and_coerces_types():
    row, warns = schemas.coerce_and_complete(
        {"TransactionID": 7, "TransactionAmt": "49.5", "card4": 123,
         "C1": "not-a-number", "extra_field": 1},
        numeric_cols=["TransactionAmt", "C1", "C2"],
        categorical_cols=["card4", "DeviceType"],
    )
    assert row["TransactionAmt"] == 49.5          # str -> float
    assert row["C1"] is None                      # uncoercible -> None + warning
    assert row["C2"] is None                      # missing -> None (imputed later)
    assert row["card4"] == "123"                  # non-null categorical -> str
    assert row["DeviceType"] is None
    assert any(w.startswith("uncoercible_numeric:C1") for w in warns)
    assert any(w.startswith("ignored_unknown_fields") for w in warns)


def test_nan_and_empty_are_null_not_errors():
    row, warns = schemas.coerce_and_complete(
        {"TransactionAmt": float("nan"), "C1": ""},
        numeric_cols=["TransactionAmt", "C1"], categorical_cols=[])
    assert row["TransactionAmt"] is None and row["C1"] is None
    assert not any(w.startswith("uncoercible") for w in warns)


def test_transaction_id_extraction():
    assert schemas.extract_transaction_id({"TransactionID": 987}) == "987"
    assert schemas.extract_transaction_id({"TransactionID": "  "}) is None
    assert schemas.extract_transaction_id({}) is None
