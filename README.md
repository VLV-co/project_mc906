# 2048 AI

O jogo 2048, apesar de suas regras simples, apresenta uma dinâmica estratégica complexa que o torna um ambiente relevante para o estudo de algoritmos de tomada de decisão sob incerteza. Este projeto explora como variações topológicas do tabuleiro, como conectividade hexagonal e triangular, afetam a dificuldade do jogo e o desempenho de algoritmos de busca.

Modelamos cada topologia como um grafo, analisando propriedades estruturais como o grau médio de conexão. Utilizando o algoritmo Expectimax, conduzimos experimentos que revelam o impacto dessas variações na mobilidade das peças e na eficácia da busca. A análise estatística, baseada em regressão linear, identificou o número de movimentos possíveis e o grau de conexão como variáveis chave na determinação da dificuldade. Os resultados destacam a importância da estrutura espacial na projeção de agentes inteligentes e variantes de jogos baseados em grids.


## Requisitos

- Python 3.8+
- Recomendado o uso de ambiente virtual (`venv`)

Instale as dependências:

```bash
pip install -r 2048_AI/requirements.txt
```

## Como executar

Para jogar manualmente:

```bash
python3 2048_AI/game.py
```

Para rodar o jogo com a IA:

```bash
bash 2048_AI/run_experiments.sh <board_variant> <size>
```

Board variants disponíveis: `square`, `hex`, `triangle`

Exemplo:

```bash
bash 2048_AI/run_experiments.sh square 4
```

## Análise de dados

Os notebooks dentro de `analysis/` permitem analisar o desempenho da IA em diferentes condições. Você pode usar o Jupyter Notebook ou JupyterLab para abri-los.
