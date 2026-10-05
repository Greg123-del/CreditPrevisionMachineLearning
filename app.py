from pathlib import Path
from typing import Any

import h2o
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "modelos"
CURRENT_MODEL_FILE = MODELS_DIR / "modelo_atual.txt"


st.set_page_config(
    page_title="Análise de crédito",
    page_icon="💳",
    layout="centered",
)


# ============================================================
# INICIALIZAÇÃO E CARREGAMENTO DO MODELO
# ============================================================

@st.cache_resource
def iniciar_h2o() -> bool:
    """
    Inicializa ou conecta ao servidor local do H2O.

    O cache evita que o Streamlit tente inicializar o H2O
    novamente a cada interação do usuário.
    """
    h2o.init()
    return True


def obter_caminho_modelo() -> str:
    """
    Lê o arquivo modelo_atual.txt e retorna o caminho
    do modelo H2O mais recente.
    """
    if not CURRENT_MODEL_FILE.exists():
        raise FileNotFoundError(
            "O arquivo modelos/modelo_atual.txt não foi encontrado. "
            "Gere um modelo antes de iniciar as previsões."
        )

    model_path = CURRENT_MODEL_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not model_path:
        raise ValueError(
            "O arquivo modelos/modelo_atual.txt está vazio."
        )

    path = Path(model_path)

    # Caso o caminho salvo seja relativo, considera-o relativo
    # à pasta raiz do projeto.
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(
            f"O modelo registrado não foi encontrado: {path}"
        )

    return str(path.resolve())


@st.cache_resource
def carregar_modelo(model_path: str):
    """
    Carrega o modelo H2O.

    O model_path é usado como argumento do cache. Quando o arquivo
    modelo_atual.txt apontar para outro modelo, um novo modelo será
    carregado.
    """
    iniciar_h2o()
    return h2o.load_model(model_path)


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def calcular_comprometimento_renda(
    renda_mensal: float,
    divida_atual: float,
    valor_solicitado: float,
    prazo_meses: int,
) -> float:
    """
    Calcula uma estimativa do comprometimento da renda.

    Essa fórmula deve ser igual à utilizada na criação do CSV
    de treinamento.
    """
    if renda_mensal <= 0:
        raise ValueError("A renda mensal deve ser maior que zero.")

    if prazo_meses <= 0:
        raise ValueError("O prazo deve ser maior que zero.")

    parcela_divida_atual = divida_atual / 24
    parcela_novo_credito = valor_solicitado / prazo_meses

    comprometimento = (
        parcela_divida_atual + parcela_novo_credito
    ) / renda_mensal

    return round(comprometimento, 4)


def criar_dados_cliente(
    renda_mensal: float,
    tempo_emprego_meses: int,
    divida_atual: float,
    valor_solicitado: float,
    prazo_meses: int,
    score_credito: int,
    atrasos_12_meses: int,
    possui_imovel: str,
    tipo_emprego: str,
) -> dict[str, Any]:
    """
    Cria o registro que será enviado para o modelo.
    """
    comprometimento_renda = calcular_comprometimento_renda(
        renda_mensal=renda_mensal,
        divida_atual=divida_atual,
        valor_solicitado=valor_solicitado,
        prazo_meses=prazo_meses,
    )

    return {
        "renda_mensal": float(renda_mensal),
        "tempo_emprego_meses": int(tempo_emprego_meses),
        "divida_atual": float(divida_atual),
        "valor_solicitado": float(valor_solicitado),
        "prazo_meses": int(prazo_meses),
        "score_credito": int(score_credito),
        "atrasos_12_meses": int(atrasos_12_meses),
        "comprometimento_renda": float(comprometimento_renda),
        "possui_imovel": possui_imovel,
        "tipo_emprego": tipo_emprego,
    }


# ============================================================
# PREVISÃO
# ============================================================

def realizar_previsao(
    dados_cliente: dict[str, Any],
) -> tuple[str, float | None, pd.DataFrame]:
    """
    Envia os dados ao modelo H2O e retorna:

    - classe prevista;
    - probabilidade da classe prevista;
    - DataFrame completo retornado pelo H2O.
    """
    model_path = obter_caminho_modelo()
    modelo = carregar_modelo(model_path)

    dados_pandas = pd.DataFrame([dados_cliente])

    iniciar_h2o()

    dados_h2o = h2o.H2OFrame(dados_pandas)

    resultado_h2o = modelo.predict(dados_h2o)

    resultado_pandas = resultado_h2o.as_data_frame(
        use_multi_thread=True
    )

    if resultado_pandas.empty:
        raise RuntimeError(
            "O modelo não retornou nenhuma previsão."
        )

    primeira_linha = resultado_pandas.iloc[0]

    if "predict" not in primeira_linha.index:
        raise RuntimeError(
            "A resposta do modelo não contém a coluna 'predict'."
        )

    previsao = str(primeira_linha["predict"])

    probabilidade = extrair_probabilidade(
        resultado=primeira_linha,
        classe_prevista=previsao,
    )

    return previsao, probabilidade, resultado_pandas


def extrair_probabilidade(
    resultado: pd.Series,
    classe_prevista: str,
) -> float | None:
    """
    Tenta encontrar a probabilidade da classe prevista.

    Em um modelo binário H2O, o resultado costuma ser parecido com:

    predict | Nao | Sim
    """
    nomes_colunas = {
        str(coluna).strip().lower(): coluna
        for coluna in resultado.index
    }

    classe_normalizada = classe_prevista.strip().lower()

    if classe_normalizada in nomes_colunas:
        coluna_original = nomes_colunas[classe_normalizada]

        try:
            return float(resultado[coluna_original])
        except (TypeError, ValueError):
            return None

    # Alguns modelos podem retornar p0 e p1.
    if classe_normalizada == "nao" and "p0" in nomes_colunas:
        try:
            return float(resultado[nomes_colunas["p0"]])
        except (TypeError, ValueError):
            return None

    if classe_normalizada == "sim" and "p1" in nomes_colunas:
        try:
            return float(resultado[nomes_colunas["p1"]])
        except (TypeError, ValueError):
            return None

    return None


def normalizar_previsao(previsao: str) -> str:
    """
    Normaliza possíveis formas retornadas pelo modelo.
    """
    valor = previsao.strip().lower()

    valores_positivos = {
        "sim",
        "apto",
        "aprovado",
        "1",
        "true",
    }

    valores_negativos = {
        "nao",
        "não",
        "não apto",
        "nao apto",
        "reprovado",
        "0",
        "false",
    }

    if valor in valores_positivos:
        return "APTO"

    if valor in valores_negativos:
        return "NÃO APTO"

    return previsao.upper()


# ============================================================
# INTERFACE
# ============================================================

st.title("💳 Análise de crédito")

st.write(
    "Preencha os dados do cliente para enviar uma solicitação "
    "de previsão ao modelo de machine learning."
)

st.info(
    "Este projeto utiliza dados sintéticos e possui finalidade didática. "
    "Não utilize esta previsão como decisão automática em um sistema real."
)


with st.form("formulario_credito"):
    st.subheader("Informações do cliente")

    coluna_esquerda, coluna_direita = st.columns(2)

    with coluna_esquerda:
        renda_mensal = st.number_input(
            label="Renda mensal",
            min_value=1.0,
            max_value=1_000_000.0,
            value=5000.0,
            step=100.0,
            format="%.2f",
            help="Renda mensal do cliente em reais.",
        )

        divida_atual = st.number_input(
            label="Dívida atual",
            min_value=0.0,
            max_value=10_000_000.0,
            value=1000.0,
            step=100.0,
            format="%.2f",
            help="Valor total aproximado das dívidas atuais.",
        )

        valor_solicitado = st.number_input(
            label="Valor solicitado",
            min_value=1.0,
            max_value=10_000_000.0,
            value=10000.0,
            step=500.0,
            format="%.2f",
        )

        prazo_meses = st.selectbox(
            label="Prazo do empréstimo",
            options=[6, 12, 18, 24, 36, 48, 60],
            index=3,
            format_func=lambda valor: f"{valor} meses",
        )

        possui_imovel = st.selectbox(
            label="Possui imóvel?",
            options=["Nao", "Sim"],
            format_func=lambda valor: (
                "Não" if valor == "Nao" else "Sim"
            ),
        )

    with coluna_direita:
        score_credito = st.number_input(
            label="Score de crédito",
            min_value=300,
            max_value=950,
            value=700,
            step=1,
        )

        atrasos_12_meses = st.number_input(
            label="Atrasos nos últimos 12 meses",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
        )

        tempo_emprego_meses = st.number_input(
            label="Tempo de emprego",
            min_value=0,
            max_value=600,
            value=24,
            step=1,
            help="Informe o tempo de emprego em meses.",
        )

        tipo_emprego = st.selectbox(
            label="Tipo de emprego",
            options=[
                "CLT",
                "Autonomo",
                "Servidor_Publico",
                "Empresario",
            ],
            format_func=lambda valor: {
                "CLT": "CLT",
                "Autonomo": "Autônomo",
                "Servidor_Publico": "Servidor público",
                "Empresario": "Empresário",
            }[valor],
        )

    enviar = st.form_submit_button(
        label="Analisar crédito",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# RESULTADO
# ============================================================

if enviar:
    try:
        dados_cliente = criar_dados_cliente(
            renda_mensal=float(renda_mensal),
            tempo_emprego_meses=int(tempo_emprego_meses),
            divida_atual=float(divida_atual),
            valor_solicitado=float(valor_solicitado),
            prazo_meses=int(prazo_meses),
            score_credito=int(score_credito),
            atrasos_12_meses=int(atrasos_12_meses),
            possui_imovel=possui_imovel,
            tipo_emprego=tipo_emprego,
        )

        comprometimento = dados_cliente[
            "comprometimento_renda"
        ]

        if comprometimento > 1:
            st.warning(
                "O comprometimento estimado da renda é superior a 100%. "
                "Confira os valores informados."
            )

        with st.spinner("Consultando o modelo..."):
            previsao, probabilidade, resultado_completo = (
                realizar_previsao(dados_cliente)
            )

        resultado_normalizado = normalizar_previsao(previsao)

        st.divider()
        st.subheader("Resultado da análise")

        if resultado_normalizado == "APTO":
            st.success("Cliente classificado como APTO.")
        elif resultado_normalizado == "NÃO APTO":
            st.error("Cliente classificado como NÃO APTO.")
        else:
            st.warning(
                f"Classe retornada pelo modelo: {resultado_normalizado}"
            )

        coluna_resultado, coluna_comprometimento = st.columns(2)

        with coluna_resultado:
            if probabilidade is not None:
                st.metric(
                    label="Probabilidade da previsão",
                    value=f"{probabilidade * 100:.2f}%",
                )
            else:
                st.metric(
                    label="Classe prevista",
                    value=resultado_normalizado,
                )

        with coluna_comprometimento:
            st.metric(
                label="Comprometimento da renda",
                value=f"{comprometimento * 100:.2f}%",
            )

        with st.expander("Dados enviados ao modelo"):
            dados_exibicao = pd.DataFrame([dados_cliente])

            st.dataframe(
                dados_exibicao,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Resposta completa do H2O"):
            st.dataframe(
                resultado_completo,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "A previsão representa somente o resultado estatístico "
            "do modelo treinado."
        )

    except FileNotFoundError as erro:
        st.error(str(erro))

        st.code(
            'python -c "from model_generator import '
            'generate_model; generate_model()"',
            language="bash",
        )

    except ValueError as erro:
        st.error(f"Dados inválidos: {erro}")

    except ConnectionError as erro:
        st.error(
            "Não foi possível conectar ao servidor H2O. "
            f"Detalhes: {erro}"
        )

    except Exception as erro:
        st.error(
            "Ocorreu um erro durante a previsão."
        )

        with st.expander("Detalhes técnicos do erro"):
            st.exception(erro)


# ============================================================
# INFORMAÇÕES LATERAIS
# ============================================================

with st.sidebar:
    st.header("Status")

    if CURRENT_MODEL_FILE.exists():
        try:
            caminho_modelo = obter_caminho_modelo()

            st.success("Modelo disponível")

            st.caption("Modelo carregado:")

            st.code(
                Path(caminho_modelo).name,
                language=None,
            )

        except Exception as erro:
            st.warning(f"Modelo indisponível: {erro}")
    else:
        st.warning("Nenhum modelo foi gerado.")

    st.divider()

    st.write(
        "O treinamento mensal deve ser executado pelo "
        "`model_generator.py` em um processo separado."
    )