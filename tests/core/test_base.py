import numpy as np
import pytest

from medaugment.core import MedVolume, Transform


class Identity(Transform):
    def apply(self, volume):
        return volume


def test_p_must_be_in_range():
    with pytest.raises(ValueError):
        Identity(p=1.5)
    with pytest.raises(ValueError):
        Identity(p=-0.1)


def test_p_zero_skips_application():
    class Sentinel(Transform):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.called = False

        def apply(self, volume):
            self.called = True
            return volume

    t = Sentinel(p=0.0, seed=0)
    vol = MedVolume(image=np.zeros((4, 4), dtype=np.float32))
    t(vol)
    assert t.called is False


def test_call_rejects_non_volume():
    with pytest.raises(TypeError):
        Identity()(np.zeros(4))  # type: ignore[arg-type]


def test_to_dict_returns_params():
    class Mine(Transform):
        def __init__(self, alpha=1.0, **kw):
            super().__init__(**kw)
            self.alpha = alpha

        def apply(self, v):
            return v

    d = Mine(alpha=3.14, p=0.5).to_dict()
    assert d["name"] == "Mine"
    assert d["params"]["alpha"] == 3.14
    assert d["params"]["p"] == 0.5
    assert "rng" not in d["params"]
