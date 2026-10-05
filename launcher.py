import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    scheduler_process = subprocess.Popen(
        [
            sys.executable,
            str(BASE_DIR / "model_generator.py"),
        ],
        cwd=BASE_DIR,
    )

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(BASE_DIR / "app.py"),
            ],
            cwd=BASE_DIR,
            check=True,
        )
    except KeyboardInterrupt:
        print("\n[Launcher] Encerrando aplicação...")
    finally:
        scheduler_process.terminate()

        try:
            scheduler_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            scheduler_process.kill()


if __name__ == "__main__":
    main()