# Previsão de Risco de Defasagem — Associação Passos Mágicos

Projeto de análise de dados e machine learning sobre a **PEDE** (Pesquisa Extensiva do
Desenvolvimento Educacional) da Associação Passos Mágicos, ciclos 2022–2024.

O modelo estima a **probabilidade de um aluno aumentar sua defasagem escolar no ciclo
seguinte**, permitindo que a equipe pedagógica priorize acompanhamento preventivo. O resultado é
entregue em uma aplicação Streamlit de consulta individual.

O modelo não roda dentro do app: ele é servido por uma **API externa**, que o Streamlit
consulta. O `.joblib` continua no repositório como artefato de referência — é o que o notebook
gera e o que o serviço executa — mas **não é mais carregado pela aplicação**.

> 📘 **[Base de Conhecimento do Modelo](Doc/Base_Conhecimento_Modelo.md)** — a documentação
> técnica completa: decisões de escopo, escolha do algoritmo, métricas e calibração, variáveis
> mais influentes, o guia de implementação do serviço e as limitações a comunicar.
>
> Este README cobre **como usar** o projeto. A Base de Conhecimento cobre **por que o modelo é
> assim e o quanto se pode confiar nele** — é a leitura indicada para quem vai avaliar, manter
> ou hospedar o modelo.

---

## Estrutura do repositório

```
.
├── requirements.txt                            # Dependências
│
├── Streamlit/
│   └── app.py                                  # Aplicação Streamlit
│
├── Model/
│   ├── Modelo_Risco_Defasagem_PEDE.ipynb       # Treino, avaliação e exportação do modelo
│   └── modelo_risco_defasagem.joblib           # Modelo treinado — servido pela API, não pelo app
│
├── Eda/
│   ├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx # Base de dados (só para os notebooks)
│   └── EDA_ajustado_e_modelos.ipynb            # Análise exploratória
│
└── Doc/
    ├── Base_Conhecimento_Modelo.md             # Documentação do modelo e contrato do serviço
    ├── Dicionário Dados Datathon.pdf           # Material de origem
    ├── PEDE_ Pontos importantes.docx           # Material de origem
    └── Relatório PEDE2020/2021/2022.pdf        # Material de origem
```

---

## Como rodar o Streamlit localmente

### Pré-requisitos

- Python 3.10 ou superior (testado em 3.12)
- `pip` instalado

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd <pasta-do-repositorio>
```

### 2. Crie e ative um ambiente virtual

Recomendado para não misturar as dependências com as do seu sistema.

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute a aplicação

```bash
.venv/bin/streamlit run Streamlit/app.py
```

O app abre automaticamente no navegador em `http://localhost:8501`. Se não abrir, acesse o
endereço manualmente.

Para encerrar, pressione `Ctrl + C` no terminal.

> **Onde fica o modelo:** em uma API externa, que o app consulta. Não é preciso treinar nem
> baixar nada para usar a aplicação.
>
> O endereço padrão já vem configurado. Para apontar para outro ambiente — um backend rodando
> na sua máquina, por exemplo — defina a variável de ambiente `API_URL`:
>
> ```bash
> API_URL=http://localhost:8000 .venv/bin/streamlit run Streamlit/app.py
> ```
>
> **A primeira consulta pode demorar até um minuto.** O serviço hiberna quando fica ocioso, e
> acordar leva tempo. O app repete cada chamada **3 vezes com 30 s de intervalo** e mostra o
> progresso na tela. Para adiantar essa espera, ele já consulta o serviço na abertura da
> página, ao carregar as opções do formulário — quando você clica em *Calcular risco*, o
> serviço normalmente já está acordado.

## Como usar a aplicação

Preencha o formulário com os dados do aluno e clique em **Calcular risco**. Todos os campos são
obrigatórios.

As opções de **gênero**, **instituição** e **fase** vêm do próprio serviço, para que o
formulário não possa divergir do que o modelo aceita. Se o serviço estiver fora do ar, o app
avisa e usa uma lista local de reserva — dá para preencher, mas o cálculo só sai com o serviço
no ar.

A **defasagem** (`Fase Efetiva − Fase Ideal`, negativo = atrasado) é informada por quem
preenche, e não derivada da idade e da fase. O modelo foi treinado com o valor medido no PEDE,
que diverge da fórmula em 10% dos alunos da base — e como essa é a variável de maior peso do
modelo (~30% da decisão), derivá-la mudaria a entrada justamente onde ela mais importa. Se o
valor informado divergir do que idade e fase sugerem, o app avisa sem bloquear o cálculo.

Junto do resultado o app mostra a classificação **Pedra**, calculada a partir do INDE. Ela é
uma referência do PEDE para a equipe e **não entra no cálculo do risco** — foi removida do
modelo por ser apenas o INDE em faixas.

O resultado mostra a faixa de risco (🟢 Baixo / 🟡 Médio / 🔴 Alto), a probabilidade estimada e
uma explicação dos fatores que mais pesaram.

---

## Como interpretar o resultado

A probabilidade deve ser lida como **frequência**, não como certeza sobre um aluno específico:

> "De cada 100 alunos com um perfil parecido com este, cerca de N aumentam a defasagem no ano
> seguinte."

Duas ressalvas importantes:

1. **A probabilidade individual tem margem de erro.** Por isso a interface exibe faixas, não
   valores com casas decimais. A ordenação entre alunos é mais confiável que o valor absoluto.
2. **Risco baixo em aluno já muito defasado não significa que ele esteja bem.** Alunos com
   defasagem de −3 ou mais raramente se defasam ainda mais, então o modelo aponta risco baixo —
   mas isso se refere apenas ao *aumento* da defasagem. Para esses casos, avalie os demais
   indicadores (IDA, IEG, IPV). Detalhes na seção 6.1 da
   [Base de Conhecimento](Doc/Base_Conhecimento_Modelo.md).

O resultado é **apoio à decisão pedagógica**, não um veredito automático.

---

## Como reexecutar os notebooks (opcional)

Necessário apenas para refazer a análise ou re-treinar o modelo.

```bash
pip install -r requirements.txt   # inclui as dependências de notebook
jupyter notebook
```

Abra e execute:

- `Eda/EDA_ajustado_e_modelos.ipynb` — análise exploratória
- `Model/Modelo_Risco_Defasagem_PEDE.ipynb` — treino do modelo (regenera o `.joblib`)

Os notebooks localizam a base em `Eda/` automaticamente, tanto rodando a partir da pasta do
próprio notebook quanto da raiz do projeto.

---

## O modelo em resumo

| Item | Valor |
|---|---|
| Algoritmo | GradientBoosting + SMOTENC + calibração isotônica |
| Alvo | `defasagem(t+1) < defasagem(t)` |
| Validação | Split agrupado por aluno (`RA`), 25% teste |
| Amostra | 1.365 pares aluno-ano (17,3% de eventos positivos) |
| Recall | 0,809 |
| ROC-AUC | 0,881 |
| PR-AUC | 0,662 (piso do acaso: 0,173) |

Variáveis mais influentes: `defasagem`, `idade`, `ano_ingresso` e `ipv`.

Como cada número foi obtido, por que este algoritmo e o que o modelo **não** consegue responder:
[Base de Conhecimento do Modelo](Doc/Base_Conhecimento_Modelo.md).

---

## Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| A abertura da página demora quase um minuto | O serviço estava hibernando e precisou acordar | Normal no primeiro acesso. As consultas seguintes são rápidas |
| "O serviço de previsão não respondeu após 3 tentativas" | Serviço fora do ar, ou `API_URL` apontando para o lugar errado | Confira o endereço; teste com `curl $API_URL/api/v1/domains`. O app não calcula localmente |
| "Usando as opções locais de reserva" | O `/domains` não respondeu na abertura | O formulário funciona, mas o cálculo precisa do serviço. Recarregue a página quando ele voltar |
| "O serviço recusou os valores informados" | Algum campo caiu fora da faixa aceita pela API | A mensagem nomeia o campo. Confira o valor e reenvie |
| `No module named 'requests'` | Dependências não instaladas no Python em uso | Use `.venv/bin/streamlit run Streamlit/app.py`, que ignora o `PATH` |
| `command not found: streamlit` | Ambiente virtual não ativado — ou ativado, mas sobreposto no `PATH` por outro Python (o instalador do python.org escreve no `~/.zprofile`, que é lido depois da ativação) | Confirme com `which streamlit`. Se não apontar para o `.venv`, chame pelo caminho: `.venv/bin/streamlit run Streamlit/app.py` |
| Porta 8501 ocupada | Outra instância rodando | `streamlit run Streamlit/app.py --server.port 8502` |
