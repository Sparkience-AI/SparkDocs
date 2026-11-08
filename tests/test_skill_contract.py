import json
from pathlib import Path

SKILL_ROOT = Path(__file__).parents[1] / ".agents" / "skills" / "sparkdocs-grounded-qa"


def test_skill_bundle_and_routing_cases() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    cases = json.loads((SKILL_ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))

    assert skill.startswith("---\nname: sparkdocs-grounded-qa\n")
    assert (SKILL_ROOT / "agents" / "openai.yaml").is_file()
    assert {case["should_trigger"] for case in cases} == {True, False}
    assert {"positive", "negative", "adversarial", "boundary", "missing-evidence"} <= {
        case["category"] for case in cases
    }
    assert all("expected_tool_order" in case for case in cases)
    assert len({case["id"] for case in cases}) == len(cases)
