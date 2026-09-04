# Previsão de Risco de Defasagem — Associação Passos Mágicos

Projeto de análise de dados e machine learning sobre a **PEDE** (Pesquisa Extensiva do
Desenvolvimento Educacional) da Associação Passos Mágicos, ciclos 2022–2024.

O modelo estima a **probabilidade de um aluno aumentar sua defasagem escolar no ciclo
seguinte**, permitindo que a equipe pedagógica priorize acompanhamento preventivo. O resultado é
entregue em uma aplicação Streamlit com consulta individual e análise em lote.

---

## Estrutura do repositório

```
.
├── app.py                                 # Aplicação Streamlit
├── requirements.txt                       # Dependências
├── Model/
│   └── modelo_risco_defasagem.joblib      # Modelo treinado (pipeline + calibração)
│
├── EDA_Passos_Magicos_PEDE_2022_2024_v2.ipynb   # Análise exploratória (10 perguntas do case)
├── Modelo_Risco_Defasagem_PEDE.ipynb            # Treino, avaliação e exportação do modelo
│
├── 01_Analise_Direcionamento_PEDE.md      # Cruzamento dos arquivos-fonte e direcionamento
├── 02_Base_Conhecimento_PEDE.md           # Glossário, fórmulas e regras de negócio da PEDE
├── 03_Base_Conhecimento_Modelo.md         # Documentação do modelo e guia do Streamlit
│
└── BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx      # Base de dados (necessária só p/ notebooks)
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

> **Onde fica o modelo:** o `app.py` procura o `modelo_risco_defasagem.joblib` automaticamente
> na própria pasta e nas subpastas `Model/`, `model/`, `models/` e `modelo/`. Ele já vem no
> repositório — não é necessário treinar nada para usar o app.
>
> Os caminhos são resolvidos a partir da pasta do `app.py`, então o comando funciona mesmo
> executado de outro diretório (ex.: `streamlit run projeto/app.py`).

## Como usar a aplicação

A interface tem duas abas.

### 👤 Aluno individual

Preencha o formulário com os dados do aluno e clique em **Calcular risco**. Todos os campos são
obrigatórios.

Dois campos são **calculados automaticamente** e exibidos abaixo do formulário:

| Campo | Como é calculado |
|---|---|
| `defasagem` | Fase cursada − fase ideal para a idade |
| `pedra` | Faixa correspondente ao INDE informado |

O resultado mostra a faixa de risco (🟢 Baixo / 🟡 Médio / 🔴 Alto), a probabilidade estimada e
uma explicação dos fatores que mais pesaram.

### 📄 Análise em lote (CSV)

Envie um CSV com uma linha por aluno para gerar o ranking de prioridade. Colunas obrigatórias:

```
idade, genero, fase_ordem, ano_ingresso, instituicao, ida, ieg, iaa, ips, ipv, inde
```

A coluna `ra` é opcional e serve apenas para identificar o aluno no resultado. Há um CSV de
exemplo disponível para download dentro da própria aba.

Nessa aba também é possível ajustar o **limiar de sinalização**, definindo quantos alunos entram
na lista de acompanhamento conforme a capacidade da equipe.

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
   indicadores (IDA, IEG, IPV). Detalhes na seção 6.1 de `03_Base_Conhecimento_Modelo.md`.

O resultado é **apoio à decisão pedagógica**, não um veredito automático.

---

## Como reexecutar os notebooks (opcional)

Necessário apenas para refazer a análise ou re-treinar o modelo.

```bash
pip install -r requirements.txt   # inclui as dependências de notebook
jupyter notebook
```

Abra e execute:

- `EDA_Passos_Magicos_PEDE_2022_2024_v2.ipynb` — análise exploratória
- `Modelo_Risco_Defasagem_PEDE.ipynb` — treino do modelo (regenera o `.joblib`)

O arquivo `BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx` precisa estar na mesma pasta dos notebooks.

---

## O modelo em resumo

| Item | Valor |
|---|---|
| Algoritmo | GradientBoosting + SMOTE + calibração isotônica |
| Alvo | `defasagem(t+1) < defasagem(t)` |
| Validação | Split agrupado por aluno (`RA`), 25% teste |
| Amostra | 1.365 pares aluno-ano (17,3% de eventos positivos) |
| Recall | 0,735 |
| ROC-AUC | 0,874 |
| PR-AUC | 0,695 (piso do acaso: 0,173) |

Variáveis mais influentes: `defasagem`, `idade`, `ano_ingresso` e `ipv`.

Documentação completa das decisões, métricas e limitações em
[`03_Base_Conhecimento_Modelo.md`](03_Base_Conhecimento_Modelo.md).

---

## Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `modelo_risco_defasagem.joblib não encontrado` | O `.joblib` não está em nenhuma das pastas procuradas | A mensagem de erro lista os locais verificados. Mova o arquivo para um deles ou use `MODELO_PATH` |
| Erro ou aviso ao carregar o modelo | Versão de `scikit-learn` ou `imbalanced-learn` diferente | Use as versões fixadas no `requirements.txt` ou re-treine pelo notebook |
| `command not found: streamlit` | Ambiente virtual não ativado | Ative o `.venv` (passo 2) |
| Porta 8501 ocupada | Outra instância rodando | `streamlit run app.py --server.port 8502` |
| Previsão parece estranha em lote | Categoria fora do domínio esperado | Confira `genero` e `instituicao` (valores aceitos na seção 7.3 de `03_Base_Conhecimento_Modelo.md`) |
