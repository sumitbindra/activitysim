from asim_core.logsums import logsum_keys, logsum_keys_distinct


def test_reference_is_1x_rule():
    assert logsum_keys(set(), has_segments=True) == ["out_period", "in_period"]
    assert logsum_keys(set(), has_segments=False) == ["out_period", "in_period", "duration"]


def test_distinct_inputs_uses_only_what_formulas_read():
    # ARC: no segments file; mode choice reads the periods (skims), not duration
    assert logsum_keys_distinct({"out_period", "in_period", "age"}, False) == ["out_period", "in_period"]
    assert logsum_keys_distinct({"duration", "out_period"}, False) == ["out_period", "duration"]
    # with a segments file, representative times depend on both periods
    assert logsum_keys_distinct({"duration"}, True) == ["out_period", "in_period"]
    assert logsum_keys_distinct({"in_period"}, True) == ["in_period"]
    assert logsum_keys_distinct(set(), True) == []  # one logsum per tour
