"""Studio catalog metadata and dependency-free rendering."""

import struct
import zlib

import numpy as np
import pytest

from medaugmentx import MedVolume
from medaugmentx.serialization import REGISTRY, from_dict
from medaugmentx.studio import render
from medaugmentx.studio.catalog import (
    CATEGORY_ORDER,
    catalog,
    describe_transform,
    parse_args_section,
    plain_text,
)
from medaugmentx.studio.session import StudioSession


def test_catalog_covers_registry_with_documented_parameters():
    entries = {entry["name"]: entry for entry in catalog()}
    assert set(entries) == set(REGISTRY)
    for entry in entries.values():
        assert entry["category"] in CATEGORY_ORDER
        assert entry["summary"]
        for param in entry["params"]:
            assert param["description"], f"{entry['name']}.{param['name']} lacks a description"
            assert "``" not in param["description"]
    assert entries["Compose"]["container"] == "transforms"
    assert entries["Guard"]["container"] == "transform"
    assert entries["PerChannel"]["container"] == "transform"
    assert entries["RandomAffine"]["card"]["implementation"].endswith("affine")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_choice_constructs_and_every_step_is_buildable(name):
    session = StudioSession()
    session.load_phantom("dbt")
    node = session.new_step(name)
    assert from_dict(node).to_dict()["name"] == name
    for param in describe_transform(name)["params"]:
        for choice in param.get("choices", []):
            from_dict({"name": name, "params": {**node["params"], param["name"]: choice}})


def test_docstring_parsing_and_plain_text():
    doc = """Summary.

    Args:
        alpha (float): First line
            continues here.
        **kwargs: Extra.

    Returns:
        Nothing.
    """
    import inspect

    parsed = parse_args_section(inspect.cleandoc(doc))
    assert parsed == {"alpha": "First line continues here.", "kwargs": "Extra."}
    assert parse_args_section("No sections") == {}
    text = "Use :class:`~medaugmentx.Resize`, :func:`numpy.pad` and ``(lo, hi)`` **now**"
    assert plain_text(text) == "Use Resize, numpy.pad and (lo, hi) now"


def _decode_png(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        chunks.setdefault(tag, b"")
        chunks[tag] += body
        pos += 12 + length
    width, height, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    channels = 1 if colour == 0 else 3
    raw = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), dtype=np.uint8)
    rows = raw.reshape(height, 1 + width * channels)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].reshape(height, width, channels).squeeze(), depth


@pytest.mark.parametrize("shape", [(5, 7), (4, 6, 3)])
def test_png_round_trip(shape):
    pixels = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    decoded, depth = _decode_png(render.encode_png(pixels))
    assert depth == 8
    np.testing.assert_array_equal(decoded, pixels)
    with pytest.raises(ValueError):
        render.encode_png(pixels.astype(np.float32))
    with pytest.raises(ValueError):
        render.encode_png(np.zeros((0, 3), np.uint8))


def test_windowing_and_label_overlay_edge_cases():
    assert render.display_window(np.full((4, 4), 3.0)) == (3.0, 4.0)
    assert render.display_window(np.full((2, 2), np.nan)) == (0.0, 1.0)
    gray = render.to_uint8(np.array([[0.0, 0.5], [1.0, np.nan]]), 0.0, 1.0)
    np.testing.assert_array_equal(gray, [[0, 128], [255, 0]])
    labels = np.array([[0, 2**64 - 2], [1, 13]], dtype=np.uint64)
    rgb = render.overlay_labels(np.zeros((2, 2), np.uint8), labels, 1.0)
    assert (rgb[0, 0] == 0).all()
    np.testing.assert_array_equal(rgb[1, 0], render.LABEL_COLOURS[0])
    np.testing.assert_array_equal(rgb[1, 1], render.LABEL_COLOURS[0])  # 13 wraps to 1
    boolean = render.overlay_labels(np.zeros((1, 2), np.uint8), np.array([[True, False]]), 1.0)
    np.testing.assert_array_equal(boolean[0, 0], render.LABEL_COLOURS[0])
    assert render.difference_image(np.zeros((2, 2)), np.zeros((3, 3))) is None
    assert render.slice_index(10, 2.0) == 9 and render.slice_index(10, -1) == 0


def test_targets_projected_onto_the_displayed_slice():
    volume = MedVolume(
        np.zeros((4, 6, 8), np.float32),
        spacing=(2.0, 1.0, 0.5),
        keypoints=np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
        keypoint_labels=np.array(["a", "b"]),
        bboxes=np.array([[0.0, 1.0, 2.0, 2.0, 3.0, 4.0]]),
    )
    targets = render.project_targets(volume, 0, 1)
    assert targets["keypoints"] == [{"row": 2.0, "col": 3.0, "label": "a"}]
    assert targets["boxes"][0] == {
        "row0": 1.0,
        "col0": 2.0,
        "row1": 3.0,
        "col1": 4.0,
        "label": None,
    }
    assert render.project_targets(volume, 0, 3.0)["boxes"] == []
    sagittal = render.project_targets(volume, 2, 5)
    assert sagittal["keypoints"][0]["row"] == 3.0 and sagittal["keypoints"][0]["col"] == 4.0
    assert render.pixel_aspect(volume, 2) == 2.0
    view = render.render_view(
        volume, axis=1, position=0.5, channel=None, window=(0.0, 1.0), mask_alpha=0.5
    )
    assert view["png"].startswith("data:image/png;base64,")
    assert (view["height"], view["width"], view["index"]) == (4, 8, 3)
