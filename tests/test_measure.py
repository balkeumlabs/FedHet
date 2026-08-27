"""Cost accounting: uplink bytes and the energy-source labelling."""
import measure


def test_uplink_payload_is_the_published_44_bytes():
    # 10 features + bias, float32 -> the "44 bytes per home per round" claim.
    assert measure.model_bytes(10) == 44


def test_model_bytes_scales_with_dimension():
    assert measure.model_bytes(0) == 4
    assert measure.model_bytes(3, dtype_bytes=8) == 32


def test_rapl_status_reports_a_reason_string():
    ok, reason = measure.rapl_status()
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason


def test_measure_returns_a_labelled_energy_source():
    result, cost = measure.measure(lambda a, b: a + b, 1, b=2)
    assert result == 3
    assert cost.energy_source in {"rapl", "tdp_estimate"}
    assert cost.wall_s >= 0.0 and cost.cpu_s >= 0.0 and cost.energy_j >= 0.0


def test_tdp_fallback_is_labelled_and_warns(monkeypatch, capsys):
    """A TDP estimate must never be reported as if it were a measurement."""
    monkeypatch.setattr(measure, "_rapl_domains", lambda: [])
    monkeypatch.setattr(measure, "_WARNED", False)
    _, cost = measure.measure(lambda: None)
    assert cost.energy_source == "tdp_estimate"
    assert "ENERGY IS ESTIMATED, NOT MEASURED" in capsys.readouterr().err


def test_rapl_reading_is_labelled_rapl(monkeypatch):
    energies = iter([1_000_000.0, 3_500_000.0])  # microjoules
    monkeypatch.setattr(measure, "_rapl_domains", lambda: ["/fake/intel-rapl:0"])
    monkeypatch.setattr(measure, "_read_energy_uj", lambda d: next(energies))
    _, cost = measure.measure(lambda: None)
    assert cost.energy_source == "rapl"
    assert cost.energy_j == 2.5


def test_counter_wraparound_falls_back_instead_of_reporting_negative_energy():
    """RAPL counters wrap; a wrapped interval must not become a bogus measurement."""
    import itertools
    energies = itertools.cycle([9_000_000.0, 1_000_000.0])
    orig = measure._read_energy_uj
    try:
        measure._read_energy_uj = lambda d: next(energies)
        measure._rapl_domains_orig = measure._rapl_domains
        measure._rapl_domains = lambda: ["/fake/intel-rapl:0"]
        _, cost = measure.measure(lambda: None)
    finally:
        measure._read_energy_uj = orig
        measure._rapl_domains = measure._rapl_domains_orig
    assert cost.energy_source == "tdp_estimate"
    assert cost.energy_j >= 0.0
