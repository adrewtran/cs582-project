from src.crm_workflow import priority_group


def test_priority_group_thresholds_are_explicit():
    assert priority_group(0.70) == "High"
    assert priority_group(0.69) == "Medium"
    assert priority_group(0.40) == "Medium"
    assert priority_group(0.39) == "Low"


def test_priority_group_rejects_invalid_probability():
    for invalid in (-0.01, 1.01):
        try:
            priority_group(invalid)
        except ValueError as error:
            assert "between 0 and 1" in str(error)
        else:
            raise AssertionError("invalid probability should be rejected")
