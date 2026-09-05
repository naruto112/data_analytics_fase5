# Base de Conhecimento — Modelo de Risco de Defasagem

> Documento de referência do modelo: por que ele é assim, como ler seus números e o que é
> preciso saber para construir o Streamlit. Complementa `01_Analise_Direcionamento_PEDE.md` e
> `02_Base_Conhecimento_PEDE.md`.

---

## 1. O que o modelo entrega

**Pergunta de negócio:** quais alunos têm maior risco de **aumentar a defasagem escolar no
próximo ciclo**, para permitir intervenção preventiva?

**Alvo:**
```
aumento_defasagem = 1  se  defasagem(t+1) < defasagem(t)
                     0  caso contrário
```
onde `defasagem = Fase Efetiva − Fase Ideal` (mais negativo = mais atrasado).

**Entregável principal: a probabilidade.** O case pede *"um modelo preditivo que mostre uma
**probabilidade** do aluno ou aluna entrar em risco de defasagem"*. A classificação binária
(sinalizar / não sinalizar) é um complemento operacional, não o produto.

| | Papel |
|---|---|
| **Probabilidade** (`predict_proba`) | **O entregável.** "Qual o risco deste aluno?" |
| **Limiar** | Ferramenta de operação. "A partir de quanto a equipe age?" |

---

## 2. Decisões de escopo

### 2.0 Padronização das categorias

A planilha usa rótulos diferentes por ciclo para o mesmo conceito. Sem tratamento, o encoder
criaria categorias duplicadas:

| Campo | Antes | Depois |
|---|---|---|
| `genero` | "Feminino", "Masculino", "Menina", "Menino" (4) | "Feminino", "Masculino" (2) |
| `instituicao` | 11 categorias, com "Pública" e "Escola Pública" separadas | 6 categorias |

Mapeamento aplicado: `Menina`→`Feminino`, `Menino`→`Masculino`, `Escola Pública`→`Pública`,
`Rede Decisão`/`Escola JP II`→`Privada`, `Nenhuma das opções acima`→nulo.

O impacto em performance é neutro (ROC-AUC 0,8805 → 0,8835; PR-AUC praticamente igual), mas o
ganho é de **consistência**: o formulário do Streamlit passa a ter opções limpas e sem
duplicatas, e o modelo deixa de dividir o sinal entre rótulos sinônimos.

### 2.1 Por que só defasagem, e não também queda de desempenho

O enunciado menciona *"queda no desempenho **ou** aumento da defasagem"*, mas a frase que pede
o modelo restringe a *"risco de defasagem"*. Além disso, juntar os dois fenômenos num único
rótulo tornaria a explicação ambígua: ao sinalizar um aluno, não se saberia se foi por risco de
nota ou de atraso de fase.

### 2.2 Por que `defasagem` e não `IAN`

`IAN` é uma transformação **com perda** da defasagem, em três faixas fixas:

| Defasagem (D) | IAN |
|---|---|
| D ≥ 0 | 10 |
| −2 ≤ D < 0 | 5 |
| D < −2 | 2,5 |

Um aluno que vai de D = −1 para D = −2 piorou de verdade, mas continuaria com IAN = 5 nos dois
anos — o evento sumiria. Pelo mesmo motivo, `IAN` foi **excluída das features** (é função
determinística de `defasagem`, portanto redundante).

### 2.3 Por que validação agrupada por RA

| Estratégia | PR-AUC | Linhas de treino | Avaliação |
|---|---|---|---|
| Temporal (2022→23 treina, 2023→24 testa) | 0,564 | 600 | Rigorosa, mas descarta metade dos dados |
| Aleatório simples | 0,617 | ~1.023 | **Otimista** — 468 alunos caem nos dois lados |
| **Aleatório agrupado por RA** | **0,602** | **~1.033** | **Escolhida** — usa tudo, sem vazamento |

### 2.4 Features excluídas

| Variável | Motivo |
|---|---|
| `ian` | Redundante — função determinística de `defasagem`. |
| `pedra` | Redundante — é o `inde` cortado em 4 faixas fixas, mesmo critério aplicado ao `ian`. |
| `ipp` | 100% nula em 2022 (coluna não existe naquela aba). |
| `ano` | Não generaliza: aprender "em 2022 acontece X" não serve para 2025. |

Sobre a `pedra`: ela era usada nas primeiras versões, mas com importância por permutação de
0,0033 — indistinguível de zero. Removê-la não custou desempenho (PR-AUC de 0,6444 para 0,6478
na média de 5 splits, dentro do desvio de ±0,07) e alinha o modelo ao critério que já havia
excluído o `ian`. A Pedra continua exibida no Streamlit como referência do PEDE, apenas não
alimenta o modelo.

---

## 3. O modelo: GradientBoosting + SMOTENC + calibração isotônica

### 3.0 Por que SMOTENC, e não SMOTE

O SMOTE cria alunos sintéticos interpolando entre vizinhos da classe minoritária. Nas versões
anteriores ele rodava **depois** da codificação, sobre as colunas já transformadas em one-hot —
o que produzia alunos com `instituicao = 0,4` num campo que só admite 0 ou 1, e `pedra = 2,7`
numa escala de inteiros. São perfis impossíveis, e o modelo aprendia com eles.

O `SMOTENC` reamostra no espaço **bruto**, antes da codificação: interpola as colunas numéricas
e resolve as categóricas pela moda dos vizinhos, produzindo sempre categorias válidas. A ordem
do pipeline passou a ser **imputação → SMOTENC → codificação → modelo**.

O ganho não é de métrica — as diferenças ficam dentro do ruído entre splits — mas de validade:
o modelo deixa de ser treinado com registros que não podem existir.

### 3.1 GradientBoosting vs. ExtraTrees

| Métrica | **GradientBoosting** | ExtraTrees tunado |
|---|---|---|
| Acurácia | 0,8554 | **0,8598** |
| Acurácia balanceada | **0,7686** | 0,7559 |
| Precisão | 0,5580 | **0,5763** |
| **Recall** | **0,6382** | 0,6004 |
| F1 | **0,5945** | 0,5864 |
| ROC-AUC | 0,8831 | 0,8831 |
| PR-AUC | 0,6419 | 0,6429 |

**GradientBoosting** segue escolhido, mas com o SMOTENC a diferença estreitou: ROC-AUC empatou
(0,8831 nos dois) e o PR-AUC ficou tecnicamente empatado — a vantagem de 0,001 do ExtraTrees é
uma fração do desvio entre splits (±0,07), portanto ruído.

O que continua separando os dois é o **recall**: 0,638 contra 0,600. Como o erro mais caro aqui
é deixar de identificar um aluno em risco, essa diferença decide. O ExtraTrees leva acurácia
bruta e precisão, ambas por margens pequenas e ambas menos relevantes para o objetivo.

### 3.2 Por que a calibração

O SMOTENC treina o modelo num universo artificialmente balanceado (metade dos alunos piorando),
o que **infla as probabilidades** em relação ao mundo real, onde só 17,3% pioram.

A calibração isotônica corrige os números **sem alterar a ordenação** dos alunos — como ajustar
a escala de um termômetro que distingue bem quente de frio, mas marca alguns graus a mais.

| | Brier (↓) | Desvio de calibração (↓) | PR-AUC |
|---|---|---|---|
| Sem calibração | 0,1018 | 0,059 | 0,642 |
| **Calibrado** | **0,0929** | **0,027** | 0,648 |

O erro de calibração caiu pela metade, sem custo na capacidade de ordenação.

### 3.3 Modelos testados e descartados

| Modelo | PR-AUC | Por que não |
|---|---|---|
| Árvore de Decisão | ~0,26 | Instável sozinha |
| Regressão Logística | ~0,43 | Não captura não-linearidades |
| XGBoost | ~0,48 | Abaixo dos concorrentes nesta base |
| ExtraTrees (config. inicial) | 0,602 | Superado pela versão tunada |
| Stacking (ET+RF+GB) | 0,623 | Mais complexo **e pior** que o GB sozinho |
| RandomForest tunado | 0,627 | Bom, mas abaixo do GB |
| **GradientBoosting** | **0,646** | **Escolhido** |

---

## 4. As métricas e como explicá-las

| Métrica | O que responde | Papel |
|---|---|---|
| **Acurácia** | % de acertos no total | ⚠️ **Enganosa**: "ninguém está em risco" já acerta 82,7%. Só referência. |
| **Acurácia balanceada** | Acerto médio em cada classe | Corrige a distorção acima. |
| **Precisão** | Dos sinalizados, quantos pioraram? | Custo operacional da equipe. |
| **Recall** | Dos que pioraram, quantos sinalizei? | **Crítica** — recall baixo = aluno em risco passa despercebido. |
| **F1** | Equilíbrio precisão × recall | Escolha do limiar. |
| **ROC-AUC** | Capacidade de ordenar risco | Independe do limiar. Piso = 0,50. |
| **PR-AUC** | Ordenação focando na classe rara | **Principal para ordenação.** Piso = 0,173. |
| **KS** | Separação entre distribuições | Padrão em modelos de risco. |
| **Brier** | Erro da probabilidade em valor absoluto | **Principal para calibração.** |

### ⚠️ O erro de leitura mais comum

**PR-AUC não se compara com uma meta de "75%".** Diferente da acurácia e do ROC-AUC (piso
0,50), o piso do PR-AUC é a **taxa de eventos**: 0,173.

Um PR-AUC de 0,685 significa **~4× melhor que o acaso**, não "68% de acerto".

Para explicar a não-técnicos: se apenas 17 em 100 alunos pioram, escolher ao acaso acerta 17%.
O modelo quase quadruplica essa capacidade.

### Métricas do modelo final

Medidas no conjunto de teste, com o limiar escolhido de **0,20**:

| Métrica | Valor | Leitura |
|---|---|---|
| Acurácia | 0,807 | 81% de acertos totais |
| Acurácia balanceada | 0,808 | Desempenho equilibrado nas duas classes |
| Precisão | 0,500 | De cada 10 sinalizados, ~5 pioram |
| **Recall** | **0,809** | **Captura ~8 de cada 10 alunos que vão piorar** |
| F1 | 0,618 | Bom equilíbrio |
| ROC-AUC | 0,881 | Forte capacidade de ordenação |
| PR-AUC | 0,662 | ~4× melhor que o acaso (piso 0,173) |
| KS | 0,639 | Boa separação |
| Brier | 0,102 | Probabilidades confiáveis |

> **Comparando com a versão anterior** (SMOTE + `pedra`, limiar 0,25): o recall subiu de 0,735
> para 0,809 e a precisão caiu de 0,532 para 0,500 — deslocamento explicado pela mudança do
> limiar ótimo, não pelo modelo. O F1 ficou praticamente igual (0,617 → 0,618).
>
> O PR-AUC deste split caiu de 0,695 para 0,662, mas **isso é ruído de amostragem, não perda
> real**: medido em 5 splits, o PR-AUC médio foi 0,6444 antes e 0,6478 depois, com desvio de
> ±0,07 entre splits. Métricas de um split único desta base variam mais do que a diferença
> observada — daí a comparação de 5 splits ser a referência.

---

## 5. Calibração: por que o número é confiável

**Uma probabilidade individual não pode ser verificada.** Se o modelo diz 30% e o aluno piora,
ele acertou ou errou? O aluno piorou ou não piorou — nunca "30%". A única forma de validar é
**em grupo**.

Previsto vs. realidade, no conjunto de teste:

| Faixa | Alunos | Previsto | Aconteceu |
|---|---|---|---|
| 0–10% | 205 | 3,3% | 4,4% ✅ |
| 10–20% | 38 | 14,3% | 10,5% |
| 20–30% | 29 | 24,7% | 34,5% |
| 30–50% | 32 | 37,5% | 43,8% |
| 50–70% | 25 | 60,0% | 40,0% |
| 70–100% | 24 | 80,6% | 87,5% ✅ |

Probabilidade média prevista **18,6%** vs. taxa real **19,3%** — praticamente idêntica.

**Isso autoriza a leitura de frequência no Streamlit:**

> "De cada 100 alunos com este perfil, cerca de X aumentam a defasagem no ano seguinte."

E **não**: "este aluno tem X% de chance" — afirmação impossível de verificar.

---

## 6. Features importantes

| # | Feature | Importância | Interpretação |
|---|---|---|---|
| 1 | `defasagem` | **+0,222** | Quem já está defasado tende a se defasar mais — efeito acumulativo. |
| 2 | `idade` | **+0,164** | A fase ideal sobe com a idade; mais velhos têm risco estrutural maior. |
| 3 | `ano_ingresso` | **+0,112** | Tempo de casa na Passos Mágicos. |
| 4 | `ipv` | **+0,089** | Ponto de Virada — indicador socioemocional mais preditivo. |
| 5 | `fase_ordem` | +0,029 | Fase cursada, contribuição pequena. |
| 6–12 | `ieg`, `ips`, `ida`, `instituicao`, `inde`, `genero`, `iaa` | < 0,03 | Contribuição pequena. |

⚠️ **Cuidado ao ler esta tabela variável a variável.** A importância por permutação embaralha uma
coluna de cada vez; quando duas variáveis são correlacionadas, o modelo compensa a perda de uma
usando a outra e a importância se **dilui** entre elas. Como `ida`, `ieg`, `ipv` e `inde` têm
correlação de 0,45 a 0,82 entre si, o `ipv` acaba levando o crédito do fator comum e os demais
aparecem perto de zero — não por serem irrelevantes, mas por serem substituíveis entre si.
A leitura correta é **por bloco**: o bloco estrutural (`defasagem`, `idade`, `ano_ingresso`)
concentra ~50% da importância, e o bloco de indicadores responde pelo restante.

**Nota de equidade:** `genero` tem importância praticamente nula — o modelo não está
discriminando por gênero.

**Para o Streamlit:** destacar `defasagem`, `idade`, `ano_ingresso` e `ipv` na explicação de
"por que este aluno foi sinalizado" — elas concentram a maior parte da capacidade preditiva.

---

## 6.1 ⚠️ Como o modelo usa a defasagem (contraintuitivo)

`defasagem` é a variável mais importante, mas a relação com o risco é **inversa** ao esperado:

| Defasagem atual | Alunos | Taxa real de piora no ano seguinte |
|---|---|---|
| −3 / −4 | 16 | **0,0%** |
| −2 | 200 | 2,5% |
| −1 | 590 | 10,7% |
| **0 (em dia)** | 513 | **27,9%** |
| +1 | 38 | 47,4% |
| +2 | 8 | 87,5% |

**Quem já está muito defasado quase não se defasa mais; quem está em dia é quem mais escorrega.**

A explicação é mecânica: a fase ideal sobe conforme a idade avança. Um aluno em dia que não
avança de fase no ano seguinte automaticamente cai para −1. Já um aluno em −3 precisaria de uma
queda adicional grande para piorar, o que é raro.

**Consequência para a leitura do resultado:** um risco baixo em aluno muito defasado **não
significa ausência de problema pedagógico** — significa apenas que o indicador específico de
*aumento* da defasagem tende a ser baixo naquele grupo. Para esses alunos, a atenção deve vir
dos demais indicadores (IDA, IEG, IPV), não deste modelo.

Essa ressalva está implementada no app: a explicação exibida muda conforme a faixa de defasagem
do aluno, para não induzir a equipe a concluir que "risco baixo = aluno bem".


## 7. Guia de implementação do Streamlit

### 7.1 Carregando o artefato

```python
import joblib

art = joblib.load("modelo_risco_defasagem.joblib")
modelo   = art["modelo"]        # pipeline completo + calibração
limiar   = art["limiar"]        # sugerido
features = art["features"]      # 12 colunas obrigatórias
faixas   = art["faixas_risco"]  # [(nome, min, max, ação), ...]
```

Chaves disponíveis: `modelo`, `calibrado`, `limiar`, `limiar_recall75`, `faixas_risco`,
`features`, `features_numericas`, `features_nominais`, `features_categoricas`,
`campos_obrigatorios`, `alvo`, `algoritmo`, `validacao`, `metricas_teste`,
`importancia_features`, `taxa_positiva_base`, `n_amostras`, `data_treino`,
`leitura_probabilidade`.

### 7.2 Fazendo a previsão

```python
proba = modelo.predict_proba(df[features])[:, 1]

def faixa_risco(p):
    for nome, minimo, maximo, acao in faixas:
        if minimo <= p < maximo:
            return nome, acao
    return "Alto", "Prioridade de acompanhamento"
```

O DataFrame precisa apenas das colunas **brutas** — imputação, padronização e encoding já estão
dentro do pipeline.

### 7.3 Campos do formulário (todos obrigatórios)

> As opções de `genero` e `instituicao` abaixo já refletem a **padronização** descrita na
> seção 2.0 — use exatamente estes rótulos, pois são os que o modelo reconhece.

| Campo | Tipo | Domínio |
|---|---|---|
| `defasagem` | numérico | Fase Efetiva − Fase Ideal (negativo = atrasado) |
| `fase_ordem` | numérico | 0 (ALFA) a 8 |
| `idade` | numérico | Anos |
| `ano_ingresso` | numérico | Ano de entrada na Passos Mágicos |
| `ida`, `ieg`, `iaa`, `ips`, `ipv`, `inde` | numérico | 0 a 10 |
| `genero` | seleção | "Feminino" / "Masculino" |
| `instituicao` | seleção | 6 opções: "Pública", "Privada", "Privada - Programa de Apadrinhamento", "Privada *Parcerias com Bolsa 100%", "Privada - Pagamento por *Empresa Parceira", "Concluiu o 3º EM" |

`pedra` **não** é mais campo do modelo (ver seção 2.4). O Streamlit continua exibindo a
classificação, calculada a partir do INDE, como referência do PEDE para a equipe.

### ⚠️ 7.4 Por que a obrigatoriedade importa

O pipeline **nunca falha** por campo vazio: o `SimpleImputer` preenche a lacuna com a mediana
(ou a moda, nas categóricas) da base de treino. Essa é justamente a armadilha — em vez de um
erro visível, o modelo devolve uma probabilidade de aparência normal, calculada sobre um aluno
que não é o que está na sua frente.

O efeito é substituir o aluno pelo **aluno médio** naquele campo. Como o modelo aprendeu a
distinguir alunos justamente pelo que os afasta da média, apagar essa distância desloca a
previsão para o centro da distribuição — ou, quando o campo é um dos de maior peso, para o
extremo oposto ao real.

Duas propriedades tornam isso grave:

- **A distorção não é proporcional ao número de campos.** Um único campo em branco basta para
  mudar a faixa de risco, porque os quatro primeiros lugares em importância concentram a maior
  parte da decisão (ver seção 6).
- **A direção do erro é imprevisível.** Dependendo do campo omitido e de quanto o aluno se
  afasta da mediana naquele indicador, o mesmo aluno pode aparecer como risco alto ou como
  risco baixo. Não existe um "lado seguro" para errar.

Por isso a validação de formulário completo é **requisito de confiabilidade, não preferência
de interface**: sem ela, a interface produz números que parecem válidos e não são — o pior
resultado possível para uma ferramenta de apoio à decisão.

> Se for útil quantificar o efeito para a equipe, o teste é direto: pegue um aluno da base,
> calcule a probabilidade completa e recalcule zerando um campo de cada vez, começando pelos de
> maior importância. Refaça a cada re-treino — os valores mudam com o modelo.

### 7.5 Como exibir o resultado

A probabilidade individual é uma **estimativa com incerteza**. O mesmo aluno, com o modelo
treinado sobre 5 divisões diferentes, recebeu de 18,8% a 28,0% — amplitude de ~9 pontos.

Por isso:

1. **Exibir faixas**, não decimais. "Risco Médio" comunica melhor que "68,4%", que sugere uma
   precisão inexistente.

   | Faixa | Intervalo | % da base | Taxa real de piora | Ação |
   |---|---|---|---|---|
   | Baixo | < 25% | 73,7% | 7,3% | Sem sinal de alerta |
   | Médio | 25–50% | 12,5% | 40,9% | Monitorar |
   | Alto | ≥ 50% | 13,9% | 63,3% | Prioridade |

   A coluna "taxa real de piora" cresce de forma consistente entre as faixas (7,3% → 40,9% →
   63,3%) — é a validação prática da calibração e o que sustenta a frase mostrada ao usuário.
2. **Ranking de prioridade** — ordenar por probabilidade decrescente para atendimento em lote.
   A ordenação é a parte mais robusta do modelo.
3. **Leitura de frequência** — "de cada 100 alunos assim, ~30 pioram".
4. **Limiar ajustável** — slider para a coordenação calibrar conforme a capacidade de
   atendimento, mostrando quantos alunos seriam sinalizados.
5. **Explicação por aluno** — exibir `defasagem`, `idade`, `ipv` e `ano_ingresso`.
6. **Aviso de uso** — apoio à decisão pedagógica, não veredito automático.

---

## 8. Limitações a comunicar

1. **Base pequena:** 1.365 pares aluno-ano, ~236 eventos positivos. Métricas variam ±0,04 a
   ±0,06 no PR-AUC entre divisões.
2. **Incerteza individual:** amplitude de ~9 pontos percentuais para um mesmo aluno entre
   modelos treinados em divisões diferentes. Daí a recomendação de faixas.
3. **`IPP` fora do modelo:** ausente em 2022. Se a coleta for padronizada, vale re-treinar
   incluindo essa variável.
4. **Features de tendência não ajudaram:** testamos a variação dos indicadores entre anos e o
   PR-AUC **caiu** (0,602 → 0,570), porque as linhas de 2022 não têm histórico anterior. Só
   valeria com mais ciclos.
5. **Teto de performance:** ~0,65 de PR-AUC com esta base. Parte da variação depende de fatores
   não capturados (mudança de escola, contexto familiar, saúde).
6. **Re-treino:** recomendável a cada novo ciclo do PEDE.
