import os
import time

import pandas as pd
import requests
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

# ------------------------------------------------------------------
# Serviço de previsão
#
# O modelo não roda mais dentro do app: ele é servido por uma API externa.
# A variável de ambiente API_URL aponta para outro host (útil para testar
# contra um backend local).
#
# O serviço hiberna quando fica ocioso e leva até um minuto para acordar,
# por isso cada chamada é repetida 3 vezes com 30 s de intervalo.
# ------------------------------------------------------------------
API_URL = os.getenv(
    "API_URL", "https://backend-ml-defasagem-fase-5.onrender.com"
).rstrip("/")

ROTA_REGISTROS = "/api/v1/defasagem-risk-records"
ROTA_DOMINIOS = "/api/v1/domains"

TENTATIVAS = 3
INTERVALO_S = 30
TIMEOUT_S = 60


class ErroAPI(Exception):
    """Falha ao falar com a API.

    `campos` traz os nomes dos campos recusados quando a resposta é 422,
    para que a interface consiga apontar o que precisa ser corrigido."""

    def __init__(self, mensagem: str, campos: list[str] | None = None):
        super().__init__(mensagem)
        self.campos = campos or []


def chamar_api(metodo: str, rota: str, corpo: dict | None = None, aviso=None):
    """Chama a API repetindo em falha de rede, timeout e erro 5xx.

    Respostas 4xx voltam sem nova tentativa: são erro do dado enviado, e
    repetir só faria quem preenche esperar 90 s por uma resposta que não vai
    mudar. `aviso` recebe uma mensagem de progresso a cada espera."""
    url = f"{API_URL}{rota}"
    ultima_causa = "motivo desconhecido"

    for tentativa in range(1, TENTATIVAS + 1):
        if tentativa > 1:
            if aviso:
                aviso(
                    f"Sem resposta ({ultima_causa}). Nova tentativa em "
                    f"{INTERVALO_S}s — {tentativa} de {TENTATIVAS}."
                )
            time.sleep(INTERVALO_S)

        try:
            resposta = requests.request(metodo, url, json=corpo, timeout=TIMEOUT_S)
        except requests.Timeout:
            ultima_causa = f"o serviço não respondeu em {TIMEOUT_S}s"
            continue
        except requests.RequestException as erro:
            ultima_causa = f"falha de conexão ({type(erro).__name__})"
            continue

        if resposta.status_code < 500:
            return resposta

        ultima_causa = f"erro {resposta.status_code} no servidor"

    raise ErroAPI(
        f"O serviço de previsão não respondeu após {TENTATIVAS} tentativas: "
        f"{ultima_causa}."
    )


def _corpo_json(resposta) -> dict:
    """Lê o JSON da resposta, tolerando corpo vazio ou malformado."""
    try:
        return resposta.json()
    except ValueError:
        return {}


# ------------------------------------------------------------------
# Domínios do formulário.
#
# A fonte da verdade é o /domains da API — assim o formulário não pode
# divergir do que o modelo aceita. A lista abaixo é apenas a reserva usada
# quando o serviço está fora; ela reflete as categorias padronizadas do
# treino (ver Doc/Base_Conhecimento_Modelo.md, seção 2.0).
# ------------------------------------------------------------------
DOMINIOS_RESERVA = {
    "genero": ["Feminino", "Masculino"],
    "instituicao": [
        "Pública",
        "Privada",
        "Privada - Programa de Apadrinhamento",
        "Privada *Parcerias com Bolsa 100%",
        "Privada - Pagamento por *Empresa Parceira",
        "Concluiu o 3º EM",
    ],
    # Fase cursada: rótulo exibido -> valor numérico (fase_ordem)
    "fases": {
        "ALFA (1º e 2º ano)": 0,
        "Fase 1 (3º e 4º ano)": 1,
        "Fase 2 (5º e 6º ano)": 2,
        "Fase 3 (7º e 8º ano)": 3,
        "Fase 4 (9º ano)": 4,
        "Fase 5 (1º ano EM)": 5,
        "Fase 6 (2º ano EM)": 6,
        "Fase 7 (3º ano EM)": 7,
        "Fase 8 (Universitários)": 8,
    },
}

# Rótulos usados para traduzir os erros de validação da API.
ROTULOS_CAMPOS = {
    "defasagem": "Defasagem",
    "fase_ordem": "Fase cursada atualmente",
    "idade": "Idade",
    "ano_ingresso": "Ano de ingresso na Passos Mágicos",
    "ida": "IDA — Desempenho acadêmico",
    "ieg": "IEG — Engajamento",
    "iaa": "IAA — Autoavaliação",
    "ips": "IPS — Psicossocial",
    "ipv": "IPV — Ponto de Virada",
    "inde": "INDE — Índice de Desenvolvimento Educacional",
    "genero": "Gênero",
    "instituicao": "Instituição de ensino",
}


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dominios(_aviso=None) -> dict:
    """Monta as opções do formulário a partir do /domains da API.

    Levanta ErroAPI em caso de falha, em vez de devolver a reserva: o
    st.cache_data não guarda chamada que levantou exceção, então a próxima
    interação tenta de novo em vez de ficar uma hora presa à lista local.
    Quem chama é que decide o que fazer sem a API.

    O `_` do parâmetro é exigência do st.cache_data: argumentos com esse
    prefixo ficam fora da chave do cache, e uma função de callback não teria
    como ser transformada em chave."""
    resposta = chamar_api("GET", ROTA_DOMINIOS, aviso=_aviso)
    if resposta.status_code != 200:
        raise ErroAPI(f"O serviço devolveu HTTP {resposta.status_code} ao listar os domínios.")

    campos = {c["field"]: c for c in _corpo_json(resposta).get("data", [])}

    def opcoes(campo: str) -> list:
        itens = sorted(campos.get(campo, {}).get("options", []),
                       key=lambda o: o.get("order", 0))
        return [o["value"] for o in itens]

    fases_api = sorted(campos.get("fase_ordem", {}).get("options", []),
                       key=lambda o: o.get("order", 0))

    dominios = {
        "genero": opcoes("genero"),
        "instituicao": opcoes("instituicao"),
        "fases": {o["label"]: o["value"] for o in fases_api},
    }

    # Um domínio vazio deixaria o formulário sem opções; nesse caso a reserva
    # é mais útil que uma tela quebrada.
    for chave, valor in dominios.items():
        if not valor:
            dominios[chave] = DOMINIOS_RESERVA[chave]

    return dominios


def avaliar_risco(payload: dict, aviso=None) -> dict:
    """Envia o registro à API e devolve o resultado da previsão.

    Fluxo: POST cria o registro e devolve o `id`; se a resposta da criação já
    trouxer a probabilidade, ela é usada direto e o GET é dispensado."""
    resposta = chamar_api("POST", ROTA_REGISTROS, corpo=payload, aviso=aviso)
    corpo = _corpo_json(resposta)

    if resposta.status_code == 422:
        campos = [e.get("field", "") for e in corpo.get("errors", []) if e.get("field")]
        raise ErroAPI(corpo.get("detail", "Um ou mais campos foram recusados."), campos)

    if resposta.status_code != 201:
        raise ErroAPI(
            corpo.get("detail")
            or f"O serviço devolveu HTTP {resposta.status_code} ao criar o registro."
        )

    if corpo.get("probabilidade") is not None:
        return corpo

    registro_id = corpo.get("id")
    if not registro_id:
        raise ErroAPI("O serviço criou o registro mas não devolveu a probabilidade nem o `id`.")

    resposta = chamar_api("GET", f"{ROTA_REGISTROS}/{registro_id}", aviso=aviso)
    if resposta.status_code != 200:
        raise ErroAPI(
            f"O registro foi criado, mas a consulta devolveu HTTP {resposta.status_code}."
        )

    corpo = _corpo_json(resposta)
    if corpo.get("probabilidade") is None:
        raise ErroAPI("O serviço devolveu o registro sem a probabilidade.")

    return corpo


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


CORES_FAIXA = {"Alto": "🔴", "Médio": "🟡", "Baixo": "🟢"}


def montar_payload(
    idade, genero, fase_ordem, ano_ingresso, instituicao,
    ida, ieg, iaa, ips, ipv, inde, defasagem,
) -> dict:
    """Monta os 12 campos que a API espera.

    `defasagem` é informada por quem preenche, e não derivada de `fase_ordem`
    e `idade`: o modelo foi treinado com a coluna medida da planilha do PEDE,
    que diverge da fórmula em 10% das linhas da base. Como essa é a variável
    de maior peso do modelo (~30%), derivá-la mudaria a entrada justamente
    onde ela mais importa.

    `pedra` não entra: foi removida do modelo por ser o INDE em faixas (ver
    seção 2.4 do Doc/Base_Conhecimento_Modelo.md) e a API recusa campos
    desconhecidos. A classificação continua exibida junto do resultado."""
    return {
        "defasagem": int(defasagem),
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
    }


def exibir_resultado(resultado: dict, payload: dict):
    """Exibe a faixa de risco, a leitura de frequência e o porquê."""
    probabilidade = float(resultado["probabilidade"])
    faixa = resultado.get("faixa_risco", "Alto")
    acao = resultado.get("acao_sugerida", "Prioridade de acompanhamento")
    emoji = CORES_FAIXA.get(faixa, "⚪")

    st.subheader(f"{emoji} Risco {faixa}")

    col1, col2 = st.columns(2)
    col1.metric("Probabilidade estimada", f"~{probabilidade:.0%}")
    col2.metric("Ação sugerida", acao)

    # Exibida aqui, junto do resultado, e não como prévia abaixo do formulário:
    # widgets dentro de um st.form só atualizam no envio, então uma legenda
    # antes do submit mostraria o valor do envio anterior.
    st.caption(
        f"Classificação **Pedra:** {pedra_para_inde(payload['inde'])} "
        f"(faixa do INDE {payload['inde']:.2f} informado). "
        "Referência do PEDE — não entra no cálculo do risco."
    )

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
                {"Variável": "Defasagem", "Valor": payload["defasagem"]},
                {"Variável": "Idade", "Valor": payload["idade"]},
                {"Variável": "Ano de ingresso", "Valor": payload["ano_ingresso"]},
                {"Variável": "IPV (Ponto de Virada)", "Valor": payload["ipv"]},
            ]
        )
        st.dataframe(principais, hide_index=True, width="stretch")

        if payload["defasagem"] >= 0:
            st.write(
                "O aluno está **na fase esperada para a idade ou acima dela**. "
                "Atenção: na base histórica, esse é justamente o grupo com maior "
                "chance de escorregar — como a fase ideal sobe a cada ano, quem "
                "está em dia perde posição se não avançar de fase."
            )
        elif payload["defasagem"] == -1:
            st.write(
                "O aluno está **1 fase abaixo** do esperado. Historicamente, esse "
                "grupo tem risco intermediário de aumentar a defasagem."
            )
        else:
            st.write(
                f"O aluno já está **{abs(payload['defasagem'])} fases abaixo** do "
                "esperado. Na base histórica, alunos nesta situação raramente se "
                "defasam ainda mais — o que **não significa ausência de risco "
                "pedagógico**, apenas que o indicador específico de *aumento* da "
                "defasagem tende a ser baixo. Avalie também os demais indicadores."
            )

    st.caption(
        "Este resultado é um apoio à decisão pedagógica, não um veredito "
        "automático. A avaliação da equipe continua sendo essencial."
    )

    if resultado.get("id"):
        st.caption(f"Identificador da consulta no serviço: `{resultado['id']}`")


# ------------------------------------------------------------------
# Abertura: busca os domínios e, de quebra, acorda o serviço — assim a
# espera do cold start acontece aqui, com aviso na tela, e não no clique
# em "Calcular risco".
# ------------------------------------------------------------------
with st.status("Conectando ao serviço de previsão...", expanded=False) as status:
    try:
        dominios = carregar_dominios(_aviso=status.write)
        status.update(label="Serviço de previsão conectado.", state="complete")
        dominios_da_api = True
    except ErroAPI as erro:
        dominios = DOMINIOS_RESERVA
        dominios_da_api = False
        status.update(label="Serviço de previsão indisponível.", state="error")
        status.write(str(erro))

if not dominios_da_api:
    st.warning(
        "Não foi possível consultar o serviço de previsão. O formulário está "
        "usando as opções locais de reserva — você pode preencher normalmente, "
        "mas o cálculo do risco só funciona com o serviço no ar."
    )

FASES = dominios["fases"]

# ------------------------------------------------------------------
# Formulário de consulta individual
# ------------------------------------------------------------------
with st.form("formulario_aluno"):
    st.subheader("Dados do aluno")

    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input("Idade", min_value=7, max_value=30, value=12, step=1)
        genero = st.selectbox("Gênero", dominios["genero"])
        ano_ingresso = st.number_input(
            "Ano de ingresso na Passos Mágicos",
            min_value=2016, max_value=2030, value=2022, step=1,
        )
    with col2:
        fase_rotulo = st.selectbox("Fase cursada atualmente", list(FASES.keys()), index=2)
        instituicao = st.selectbox("Instituição de ensino", dominios["instituicao"])
        defasagem = st.number_input(
            "Defasagem",
            min_value=-8, max_value=8, value=0, step=1,
            help=(
                "Fase Efetiva − Fase Ideal, como registrado no PEDE. "
                "Negativo = aluno atrasado."
            ),
        )

    st.caption(
        "ℹ️ **Sobre a defasagem** — é a variável de maior peso do modelo "
        "(cerca de 31% da decisão), então vale conferir o valor. "
        "**Negativo = atrasado** (`Fase Efetiva − Fase Ideal`). "
        "Atenção à direção, que é contraintuitiva: quem está **em dia** é quem "
        "mais escorrega (27,9% pioram no ano seguinte, contra 2,5% entre os que "
        "já estão em −2), porque a fase ideal sobe a cada ano e quem não avança "
        "de fase perde posição automaticamente."
    )

    st.subheader("Indicadores acadêmicos")

    # O teto é 10,1, e não 10,0: o IPV real do PEDE chega a 10,01 por
    # arredondamento da metodologia, e o modelo foi treinado com esses valores.
    col3, col4 = st.columns(2)
    with col3:
        ida = st.slider("IDA — Desempenho acadêmico", 0.0, 10.1, 6.9, 0.1)
    with col4:
        ieg = st.slider("IEG — Engajamento", 0.0, 10.1, 8.9, 0.1)

    st.subheader("Indicadores socioemocionais")

    col5, col6, col7 = st.columns(3)
    with col5:
        iaa = st.slider("IAA — Autoavaliação", 0.0, 10.1, 8.8, 0.1)
    with col6:
        ips = st.slider("IPS — Psicossocial", 0.0, 10.1, 6.9, 0.1)
    with col7:
        ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.1, 7.8, 0.1)

    st.subheader("Índice geral")
    inde = st.slider("INDE — Índice de Desenvolvimento Educacional", 0.0, 10.1, 7.5, 0.01)

    enviado = st.form_submit_button("Calcular risco", width="stretch")

fase_ordem = FASES[fase_rotulo]

if enviado:
    payload = montar_payload(
        idade, genero, fase_ordem, ano_ingresso, instituicao,
        ida, ieg, iaa, ips, ipv, inde, defasagem,
    )

    # Conferência não bloqueante: 10% da base real diverge da fórmula, então a
    # divergência é plausível — o aviso confirma o valor, não o rejeita.
    # Feita aqui, e não como prévia ao vivo, porque widgets dentro de um
    # st.form só atualizam no envio.
    defasagem_esperada = calcular_defasagem(fase_ordem, idade)
    if int(defasagem) != defasagem_esperada:
        st.warning(
            f"Você informou defasagem **{int(defasagem):+d}**, mas a idade "
            f"({idade} anos) e a fase cursada sugerem **{defasagem_esperada:+d}** "
            f"(fase ideal para a idade: {fase_ideal_para_idade(idade)}). "
            "Confirme o valor — divergências são normais e acontecem em cerca de "
            "10% dos alunos da base, mas vale checar se não foi engano."
        )

    try:
        with st.status("Consultando o serviço de previsão...", expanded=True) as status:
            resultado = avaliar_risco(payload, aviso=status.write)
            status.update(label="Previsão recebida.", state="complete", expanded=False)
        st.divider()
        exibir_resultado(resultado, payload)
    except ErroAPI as erro:
        if erro.campos:
            nomes = "\n".join(
                f"- **{ROTULOS_CAMPOS.get(campo, campo)}**" for campo in erro.campos
            )
            st.error(
                "O serviço recusou os valores informados nos campos abaixo:\n\n"
                f"{nomes}\n\nRevise e envie novamente."
            )
        else:
            st.error(
                f"{erro}\n\nO serviço hiberna quando fica ocioso — se este foi o "
                "primeiro acesso do dia, tente novamente em alguns instantes."
            )
