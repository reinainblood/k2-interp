import pytest

from k2_interp.cli import panel, parser
from k2_interp.schema import CaptureSpec, Intervention, RunSpec, Sample


def test_contracts_reject_silent_mistakes():
    with pytest.raises(ValueError):
        Sample(id="../escape", text="test")
    with pytest.raises(ValueError):
        Sample(id="x", text="test", messages=[])
    with pytest.raises(ValueError):
        RunSpec(run_id="x", samples=[Sample(id="s", text="a"), Sample(id="s", text="b")])
    with pytest.raises(ValueError):
        CaptureSpec(sites=["residual-ish"])
    with pytest.raises(ValueError):
        Intervention(layer=3, site="expert_ids", positions=[0], operation="scale")
    assert len(panel(["all"])) == 11
    assert len(panel(["spine"])) == 6
    assert parser().parse_args(["deploy", "--checkpoints", "all"]).command == "deploy"
