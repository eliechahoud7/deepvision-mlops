from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def requirement_lines(name: str) -> list[str]:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_serving_requirements_are_a_subset_of_runtime_requirements():
    runtime = set(requirement_lines("requirements.txt"))
    serving = requirement_lines("requirements-serve.txt")
    assert serving
    assert [line for line in serving if line not in runtime] == []
