"""Suite Pytest de Auditoria Adversarial Red-Team (MoS + SCache O(1) — Custo vs Qualidade)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adversarial_mos_scache_suite import run_full_adversarial_suite


def test_full_adversarial_mos_scache_cost_and_quality(tmp_path: Path):
    out_json = tmp_path / "adversarial_report.json"
    rep = run_full_adversarial_suite(output_json_path=out_json)
    assert rep["all_passed"] is True
    assert out_json.exists()
    for k in (
        "attack_0_anti_facade",
        "attack_1_mos_multidomain_outliers",
        "attack_2_context_horizon_bomb",
        "attack_3_poisoned_haystack_antighosting",
        "attack_4_noise_flood_dos_immunity",
        "attack_5_combined_fullstack_mos_scache",
        "attack_6_colossal_72b_streaming_proof",
    ):
        assert rep["attacks"][k]["passed"] is True
