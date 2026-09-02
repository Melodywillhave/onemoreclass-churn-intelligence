from __future__ import annotations

import subprocess
import sys


PIPELINE_STEPS = [
    ("Generate synthetic data", "src.data.generate_data"),
    ("Validate raw data", "src.data.validate_data"),
    ("Build churn labels", "scripts.build_labels"),
    ("Build modeling features", "scripts.build_features"),
    ("Run model evaluation", "scripts.run_evaluation"),
    ("Score customers", "scripts.run_scoring"),
    ("Build retention decisions", "scripts.run_retention_decisions"),
    ("Initialize database", "scripts.init_database"),
    ("Load customer risk", "scripts.load_customer_risk"),
    ("Load retention decisions", "scripts.load_retention_decisions"),
]


def run_step(step_number: int, name: str, module: str) -> None:
    """Run one pipeline step and stop immediately if it fails."""
    print("\n" + "=" * 70)
    print(
        f"STEP {step_number}/{len(PIPELINE_STEPS)}: {name}"
    )
    print("=" * 70)

    subprocess.run(
        [sys.executable, "-m", module],
        check=True,
    )


def main() -> None:
    """Build the complete local churn intelligence demo."""
    print("Starting churn intelligence demo setup...")

    for step_number, (name, module) in enumerate(
        PIPELINE_STEPS,
        start=1,
    ):
        run_step(
            step_number=step_number,
            name=name,
            module=module,
        )

    print("\n" + "=" * 70)
    print("Demo setup completed successfully.")
    print("=" * 70)
    print(
        "\nNext:"
        "\n1. Start Ollama: ollama serve"
        "\n2. Start API: uvicorn src.api.app:app --reload"
        "\n3. Start UI: streamlit run src/ui/app.py"
    )


if __name__ == "__main__":
    main()