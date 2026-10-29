import re
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


def package(line: str) -> str:
    return re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].lower()


def test_frontend_requirements_agree_with_the_other_files():
    frontend = requirement_lines("requirements-frontend.txt")
    others = {
        package(line): line
        for name in ("requirements.txt", "requirements-dev.txt")
        for line in requirement_lines(name)
        if not line.startswith("-r")
    }
    assert {package(line) for line in frontend} == {"streamlit", "httpx", "pillow"}
    # A package pinned in both places must use the same range.
    assert [
        line for line in frontend if package(line) in others and others[package(line)] != line
    ] == []
    # The frontend never needs the model stack.
    assert not {"torch", "torchvision"} & {package(line) for line in frontend}


def test_dev_requirements_include_the_frontend():
    assert "-r requirements-frontend.txt" in requirement_lines("requirements-dev.txt")
