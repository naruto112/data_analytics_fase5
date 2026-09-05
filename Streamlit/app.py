import io
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------------
st.set_page_config(page_title="Risco de Defasagem — Passos Mágicos", page_icon="📚")

st.title("📚 Previsão de Risco de Defasagem")
st.write(
    "Preencha as informações do aluno abaixo para estimar a probabilidade de "
    "ele aumentar a defasagem escolar no próximo ciclo."
)

BASE_DIR = Path(__file__).resolve().parent
RAIZ_PROJETO = BASE_DIR.parent

# O modelo mora em Model/, na raiz do projeto (um nível acima do app.py).
# O caminho é resolvido a partir da pasta do app.py, e não do diretório de onde
# o comando foi executado — assim o app funciona mesmo rodando
# `streamlit run caminho/para/app.py` de qualquer lugar.
#
# A variável de ambiente MODELO_PATH, se definida, tem prioridade (útil em
# deploys onde o arquivo fica em outro lugar).
NOME_MODELO = "modelo_risco_defasagem.joblib"

MODELO_PATH = Path(os.getenv("MODELO_PATH") or RAIZ_PROJETO / "Model" / NOME_MODELO)
if not MODELO_PATH.is_absolute():
    MODELO_PATH = BASE_DIR / MODELO_PATH


# ------------------------------------------------------------------
# Carregamento do modelo treinado (pipeline completo + calibração).
# Cacheado para não recarregar a cada interação.
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def carregar_modelo() -> dict:
    if not MODELO_PATH.exists():
        raise FileNotFoundError(MODELO_PATH.name)
    return joblib.load(MODELO_PATH)


try:
    with st.spinner("Carregando modelo..."):
        artefato = carregar_modelo()
except FileNotFoundError:
    st.error(
        f"Arquivo **{NOME_MODELO}** não encontrado em `{MODELO_PATH}`.\n\n"
        "Coloque o arquivo nesse caminho ou defina a variável de ambiente "
        "`MODELO_PATH` com a localização dele."
    )
    st.stop()
except ModuleNotFoundError as erro:
    st.error(
        f"O modelo depende do pacote **{erro.name}**, que não está instalado "
        f"no Python que está rodando o app (`{sys.executable}`).\n\n"
        "Rode o app pelo ambiente virtual do projeto, que já tem todas as "
        "dependências:\n\n"
        "```\n.venv/bin/streamlit run Streamlit/app.py\n```\n\n"
        "Ou instale as dependências no ambiente atual: "
        "`pip install -r requirements.txt`."
    )
    st.stop()
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: `{type(erro).__name__}: {erro}`")
    st.stop()

MODELO = artefato["modelo"]
FEATURES = artefato["features"]
FAIXAS_RISCO = artefato["faixas_risco"]
LIMIAR_PADRAO = artefato["limiar"]

# ------------------------------------------------------------------
# Domínios do formulário.
# Os rótulos de gênero e instituição seguem exatamente as categorias
# padronizadas que o modelo viu no treino (ver Doc/Base_Conhecimento_Modelo.md,
# seção 2.0). Enviar um valor fora desta lista faz o encoder ignorá-lo
# silenciosamente e degrada a previsão.
# ------------------------------------------------------------------
DOMINIOS = {
    "genero": ["Feminino", "Masculino"],
    "instituicao": [
        "Pública",
        "Privada",
        "Privada - Programa de Apadrinhamento",
        "Privada *Parcerias com Bolsa 100%",
        "Privada - Pagamento por *Empresa Parceira",
        "Concluiu o 3º EM",
    ],
}

# Fase cursada: rótulo exibido -> valor numérico (fase_ordem)
FASES = {
    "ALFA (1º e 2º ano)": 0,
    "Fase 1 (3º e 4º ano)": 1,
    "Fase 2 (5º e 6º ano)": 2,
    "Fase 3 (7º e 8º ano)": 3,
    "Fase 4 (9º ano)": 4,
    "Fase 5 (1º EM)": 5,
    "Fase 6 (2º EM)": 6,
    "Fase 7 (3º EM)": 7,
    "Fase 8 (Universitários)": 8,
}

# Idade -> fase ideal, derivado da própria base PEDE (confere com a
# Tabela 4 do documento de indicadores).
FASE_IDEAL_POR_IDADE = {
    7: 0, 8: 0, 9: 1, 10: 2, 11: 2, 12: 3, 13: 3, 14: 4,
    15: 5, 16: 6, 17: 7,
}

# Faixas de INDE que definem a Pedra (conceito PEDE)
FAIXAS_PEDRA = [
    ("Quartzo", 0.000, 6.110),
    ("Ágata", 6.110, 7.154),
    ("Ametista", 7.154, 8.198),
    ("Topázio", 8.198, 10.001),
]


# ------------------------------------------------------------------
# Regras de negócio derivadas (calculadas, não digitadas)
# ------------------------------------------------------------------
def fase_ideal_para_idade(idade: int) -> int:
    """Fase esperada para a idade do aluno. A partir de 18 anos, Fase 8."""
    return FASE_IDEAL_POR_IDADE.get(int(idade), 8)


def calcular_defasagem(fase_ordem: int, idade: int) -> int:
    """D = Fase Efetiva - Fase Ideal. Negativo = aluno atrasado."""
    return int(fase_ordem) - fase_ideal_para_idade(idade)


def pedra_para_inde(inde: float) -> str:
    """Classificação Pedra a partir do INDE."""
    for nome, minimo, maximo in FAIXAS_PEDRA:
        if minimo <= inde < maximo:
            return nome
    return "Topázio"


def faixa_de_risco(probabilidade: float):
    """Retorna (nome da faixa, ação sugerida) para uma probabilidade."""
    for nome, minimo, maximo, acao in FAIXAS_RISCO:
        if minimo <= probabilidade < maximo:
            return nome, acao
    return "Alto", "Prioridade de acompanhamento"


CORES_FAIXA = {"Alto": "🔴", "Médio": "🟡", "Baixo": "🟢"}


def montar_registro(
    idade, genero, fase_ordem, ano_ingresso, instituicao,
    ida, ieg, iaa, ips, ipv, inde,
) -> dict:
    """Monta o registro no formato esperado pelo modelo, calculando os
    campos derivados (defasagem e pedra)."""
    return {
        "defasagem": calcular_defasagem(fase_ordem, idade),
        "fase_ordem": int(fase_ordem),
        "idade": int(idade),
        "ano_ingresso": int(ano_ingresso),
        "ida": float(ida),
        "ieg": float(ieg),
        "iaa": float(iaa),
        "ips": float(ips),
        "ipv": float(ipv),
        "inde": float(inde),
        "genero": genero,
        "instituicao": instituicao,
        "pedra": pedra_para_inde(inde),
    }


def prever(registros: pd.DataFrame):
    """Retorna as probabilidades de aumento de defasagem."""
    return MODELO.predict_proba(registros[FEATURES])[:, 1]


def ler_csv_utf8(arquivo) -> pd.DataFrame:
    """Lê o CSV enviado exigindo codificação UTF-8.

    Arquivos em Latin-1/Windows-1252 (o que o Excel em português gera ao
    salvar como "CSV separado por vírgulas") são REJEITADOS em vez de lidos
    com os acentos corrompidos — "Pública" viraria "PÃºblica" e deixaria de
    casar com as categorias que o modelo viu no treino.

    O BOM que o Excel escreve ao salvar como "CSV UTF-8" é aceito e removido,
    pois é UTF-8 válido; mantê-lo corromperia o nome da primeira coluna.

    Levanta ValueError com uma mensagem pronta para exibição.
    """
    bruto = arquivo.getvalue()

    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError as erro:
        # Localiza o problema para o usuário: linha e byte ofensivo.
        linha = bruto.count(b"\n", 0, erro.start) + 1
        byte_invalido = bruto[erro.start]
        raise ValueError(
            f"O arquivo não está em **UTF-8**: o byte `0x{byte_invalido:02X}` "
            f"na linha {linha} não é válido nessa codificação.\n\n"
            "Provavelmente ele foi salvo em Latin-1 / Windows-1252. "
            "Salve novamente em UTF-8 e envie de novo:\n\n"
            "- **Excel:** Arquivo → Salvar como → *CSV UTF-8 (delimitado por vírgulas)*\n"
            "- **Google Planilhas:** Arquivo → Fazer download → *CSV* (já sai em UTF-8)\n"
            "- **VS Code:** clique na codificação na barra inferior → "
            "*Save with Encoding* → *UTF-8*"
        ) from erro

    return pd.read_csv(io.StringIO(texto))


def exibir_resultado(probabilidade: float, registro: dict):
    """Exibe a faixa de risco, a leitura de frequência e o porquê."""
    faixa, acao = faixa_de_risco(probabilidade)
    emoji = CORES_FAIXA.get(faixa, "⚪")

    st.subheader(f"{emoji} Risco {faixa}")

    col1, col2 = st.columns(2)
    col1.metric("Probabilidade estimada", f"~{probabilidade:.0%}")
    col2.metric("Ação sugerida", acao)

    st.info(
        f"**Como ler:** de cada 100 alunos com um perfil parecido com este, "
        f"cerca de **{probabilidade * 100:.0f}** aumentam a defasagem no ano "
        "seguinte. O valor é uma estimativa de frequência, não uma certeza "
        "sobre este aluno específico."
    )

    with st.expander("Por que este resultado?"):
        st.write(
            "As variáveis com maior peso no modelo, e os valores informados "
            "para este aluno:"
        )
        principais = pd.DataFrame(
            [
                {"Variável": "Defasagem", "Valor": registro["defasagem"]},
                {"Variável": "Idade", "Valor": registro["idade"]},
                {"Variável": "Ano de ingresso", "Valor": registro["ano_ingresso"]},
                {"Variável": "IPV (Ponto de Virada)", "Valor": registro["ipv"]},
            ]
        )
        st.dataframe(principais, hide_index=True, use_container_width=True)

        if registro["defasagem"] >= 0:
            st.write(
                "O aluno está **na fase esperada para a idade ou acima dela**. "
                "Atenção: na base histórica, esse é justamente o grupo com maior "
                "chance de escorregar — como a fase ideal sobe a cada ano, quem "
                "está em dia perde posição se não avançar de fase."
            )
        elif registro["defasagem"] == -1:
            st.write(
                "O aluno está **1 fase abaixo** do esperado. Historicamente, esse "
                "grupo tem risco intermediário de aumentar a defasagem."
            )
        else:
            st.write(
                f"O aluno já está **{abs(registro['defasagem'])} fases abaixo** do "
                "esperado. Na base histórica, alunos nesta situação raramente se "
                "defasam ainda mais — o que **não significa ausência de risco "
                "pedagógico**, apenas que o indicador específico de *aumento* da "
                "defasagem tende a ser baixo. Avalie também os demais indicadores."
            )

    st.caption(
        "Este resultado é um apoio à decisão pedagógica, não um veredito "
        "automático. A avaliação da equipe continua sendo essencial."
    )


# ------------------------------------------------------------------
# Abas: consulta individual e análise em lote
# ------------------------------------------------------------------
# aba_individual, aba_lote = st.tabs(["👤 Aluno individual", "📄 Análise em lote (CSV)"])

# ------------------------------------------------------------------
# Aba 1 — Formulário de aluno individual
# ------------------------------------------------------------------
# with aba_individual:
with st.form("formulario_aluno"):
    st.subheader("Dados do aluno")

    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input("Idade", min_value=7, max_value=27, value=12, step=1)
        genero = st.selectbox("Gênero", DOMINIOS["genero"])
        ano_ingresso = st.number_input(
            "Ano de ingresso na Passos Mágicos",
            min_value=2016, max_value=2030, value=2022, step=1,
        )
    with col2:
        fase_rotulo = st.selectbox("Fase cursada atualmente", list(FASES.keys()), index=2)
        instituicao = st.selectbox("Instituição de ensino", DOMINIOS["instituicao"])

    st.subheader("Indicadores acadêmicos")

    col3, col4 = st.columns(2)
    with col3:
        ida = st.slider("IDA — Desempenho acadêmico", 0.0, 10.0, 6.9, 0.1)
    with col4:
        ieg = st.slider("IEG — Engajamento", 0.0, 10.0, 8.9, 0.1)

    st.subheader("Indicadores socioemocionais")

    col5, col6, col7 = st.columns(3)
    with col5:
        iaa = st.slider("IAA — Autoavaliação", 0.0, 10.0, 8.8, 0.1)
    with col6:
        ips = st.slider("IPS — Psicossocial", 0.0, 10.0, 6.9, 0.1)
    with col7:
        ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 7.8, 0.1)

    st.subheader("Índice geral")
    inde = st.slider("INDE — Índice de Desenvolvimento Educacional", 0.0, 10.0, 7.5, 0.01)

    enviado = st.form_submit_button("Calcular risco", use_container_width=True)

# Prévia dos campos calculados automaticamente
fase_ordem = FASES[fase_rotulo]
defasagem_calc = calcular_defasagem(fase_ordem, idade)
pedra_calc = pedra_para_inde(inde)

st.caption(
    f"Campos calculados automaticamente — "
    f"**Fase ideal para a idade:** {fase_ideal_para_idade(idade)} | "
    f"**Defasagem:** {defasagem_calc:+d} | "
    f"**Pedra:** {pedra_calc}"
)

if enviado:
    registro = montar_registro(
        idade, genero, fase_ordem, ano_ingresso, instituicao,
        ida, ieg, iaa, ips, ipv, inde,
    )
    try:
        with st.spinner("Calculando..."):
            probabilidade = prever(pd.DataFrame([registro]))[0]
        st.divider()
        exibir_resultado(probabilidade, registro)
    except Exception:
        st.error(
            "Não foi possível calcular o risco com os dados informados. "
            "Revise os campos e tente novamente."
        )