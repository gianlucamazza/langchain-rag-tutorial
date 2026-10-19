"""Static checks for tutorial dependency pins."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines(filename):
    lines = []
    for raw in (ROOT / filename).read_text().splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if stripped:
            lines.append(stripped)
    return lines


def test_networkx_keeps_python_310_via_environment_markers():
    """NetworkX 3.5+ requires Python >=3.11; 3.10 stays on 3.4.x via a marker."""
    lines = [line for line in _requirement_lines("requirements.txt") if line.startswith("networkx")]
    py310 = [line for line in lines if 'python_version < "3.11"' in line]
    py311 = [line for line in lines if 'python_version >= "3.11"' in line]

    assert len(py310) == 1, lines
    assert len(py311) == 1, lines
    assert ">=3.4.2" in py310[0].replace(" ", "")
    assert "<3.5" in py310[0].replace(" ", "")
    assert ">=3.6.1" in py311[0].replace(" ", "")
    assert "<3.7" in py311[0].replace(" ", "")
