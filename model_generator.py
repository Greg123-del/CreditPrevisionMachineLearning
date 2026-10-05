from datetime import datetime
from pathlib import Path
import asyncio

import h2o
from h2o.automl import H2OAutoML


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = ( BASE_DIR/ "dados"/ "dados_treino_credito_200.csv")
MODELS_DIR = BASE_DIR / "modelos"
CURRENT_MODEL_FILE = MODELS_DIR / "modelo_atual.txt"


def iniciar_h2o() -> None:
    """
    Inicia um cluster H2O local ou reutiliza um já iniciado.
    """
    print("[H2O] Inicializando cluster...")

    h2o.init(
        ip="localhost",
        port=54321,
        start_h2o=True,
    )

    print("[H2O] Cluster conectado com sucesso.")


def generate_model() -> str:
    print("[Modelo] Iniciando treinamento...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {DATA_PATH}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Esta chamada precisa ocorrer antes de import_file().
    iniciar_h2o()

    print(f"[Modelo] Carregando CSV: {DATA_PATH}")

    data = h2o.import_file(
        path=str(DATA_PATH.resolve())
    )

    target = "apto_credito"

    if target not in data.columns:
        raise ValueError(
            f"A coluna '{target}' não foi encontrada. "
            f"Colunas disponíveis: {data.columns}"
        )

    # Informa ao H2O que a previsão é uma classificação.
    data[target] = data[target].asfactor()

    treino, teste = data.split_frame(
        ratios=[0.7],
        seed=42,
    )

    features = [
        coluna
        for coluna in data.columns
        if coluna not in {
            target,
            "cliente_id",
        }
    ]

    print(f"[Modelo] Variáveis de entrada: {features}")
    print(f"[Modelo] Variável-alvo: {target}")

    automl = H2OAutoML(
        max_runtime_secs=180,
        sort_metric="AUTO",
        seed=42,
    )

    automl.train(
        x=features,
        y=target,
        training_frame=treino,
        leaderboard_frame=teste,
    )

    modelo = automl.leader

    if modelo is None:
        raise RuntimeError(
            "O H2O AutoML não retornou um modelo."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_directory = (MODELS_DIR/ timestamp)

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_model_path = h2o.save_model(
        model=modelo,
        path=str(model_directory.resolve()),
        force=True,
    )

    CURRENT_MODEL_FILE.write_text(
        saved_model_path,
        encoding="utf-8",
    )

    leaderboard_path = (
        MODELS_DIR
        / f"leaderboard_{timestamp}.csv"
    )

    leaderboard = (
        automl
        .leaderboard
        .as_data_frame()
    )

    leaderboard.to_csv(
        leaderboard_path,
        index=False,
    )

    performance = modelo.model_performance(
        teste
    )

    print("[Modelo] Treinamento concluído.")
    print(f"[Modelo] ID: {modelo.model_id}")
    print(
        f"[Modelo] Arquivo salvo em: "
        f"{saved_model_path}"
    )
    print(
        f"[Modelo] Leaderboard salvo em: "
        f"{leaderboard_path}"
    )
    print(performance)

    return saved_model_path


def seconds_until_next_day10() -> float:
    now = datetime.now()

    scheduled_this_month = now.replace(
        day=10,
        hour=8,
        minute=0,
        second=0,
        microsecond=0,
    )

    if now < scheduled_this_month:
        next_run = scheduled_this_month

    elif now.month == 12:
        next_run = now.replace(
            year=now.year + 1,
            month=1,
            day=10,
            hour=8,
            minute=0,
            second=0,
            microsecond=0,
        )

    else:
        next_run = now.replace(
            month=now.month + 1,
            day=10,
            hour=8,
            minute=0,
            second=0,
            microsecond=0,
        )

    delta = max(
        (next_run - now).total_seconds(),
        0,
    )

    print(
        "[Agendador] Próxima execução: "
        f"{next_run:%d/%m/%Y às %H:%M:%S}"
    )

    return delta


async def scheduler() -> None:
    print("[Agendador] Iniciado.")

    while True:
        wait_seconds = seconds_until_next_day10()

        await asyncio.sleep(wait_seconds)

        try:
            await asyncio.to_thread(
                generate_model
            )

        except Exception as error:
            print(
                "[Agendador] Erro durante "
                f"o treinamento: {error}"
            )


if __name__ == "__main__":
    asyncio.run(scheduler())