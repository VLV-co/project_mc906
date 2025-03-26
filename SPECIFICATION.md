## **1. Objetivo do Projeto**
O objetivo do projeto é criar uma IA capaz de jogar o jogo **2048** de maneira otimizada usando **Deep Q-Learning (DQN)**, um algoritmo de Aprendizado por Reforço. A IA aprenderá a fazer jogadas no tabuleiro 4x4 com o objetivo de alcançar a maior pontuação possível, considerando que uma peça aleatória será adicionada ao tabuleiro a cada rodada.

---

## **2. Pacotes e Dependências**

### **2.1. Pacotes principais**

- **Gym**: Biblioteca para criar o ambiente de aprendizado, com uma interface padrão para simular o jogo 2048.
  - Instalação: `pip install gym`
  
- **TensorFlow ou PyTorch**: Para a construção e treinamento da rede neural.
  - Instalação: `pip install tensorflow` (ou `pip install torch` para PyTorch)
  
- **NumPy**: Para manipulação de arrays e operações matemáticas.
  - Instalação: `pip install numpy`
  
- **Matplotlib**: Para visualização dos resultados do treinamento, gráficos de recompensas e a progressão do modelo.
  - Instalação: `pip install matplotlib`
  
- **OpenAI Baselines (opcional)**: Para usar implementações de RL já testadas e rodar benchmarks.
  - Instalação: `pip install stable-baselines3`

### **2.2. Pacotes para Interface Gráfica (Opcional)**

Se você quiser implementar uma interface gráfica para interagir com o jogo ou visualizar os estados do jogo:

- **Pygame**: Biblioteca para criar interfaces gráficas para jogos, útil para mostrar o tabuleiro 2048.
  - Instalação: `pip install pygame`
  
- **Tkinter (opcional)**: Se quiser algo mais simples para exibição gráfica.
  - Instalação: Já incluído no Python.

---

## **3. Estrutura do Projeto**

Aqui está uma estrutura proposta para o projeto:

```
2048_AI/
│
├── game/
│   ├── __init__.py              # Inicialização do módulo do jogo
│   ├── game_environment.py      # Implementação do ambiente 2048 para o RL
│   ├── game_logic.py            # Lógica do jogo (movimentos, combinações, etc.)
│   └── game_renderer.py         # Funções para renderização do jogo (gráfico ou terminal)
│
├── agent/
│   ├── __init__.py              # Inicialização do módulo do agente
│   ├── dqn_agent.py             # Implementação do agente DQN
│   └── replay_buffer.py         # Implementação do buffer de experiência (ReplayBuffer)
│
├── training/
│   ├── train.py                 # Script principal para o treinamento do agente
│   └── utils.py                 # Funções auxiliares (ex. salvar modelo, log)
│
├── graphics/                    # Pasta para gráficos e interface gráfica
│   └── display.py               # Exibição do jogo para o usuário
│
├── results/                     # Pasta para resultados do treinamento
│   └── logs/                    # Logs de treinamento e checkpoints
│
└── README.md                    # Documentação do projeto
```

---

## **4. Processo de Treinamento**

### **4.1. Ambiente do Jogo (Gym)**

- O jogo será modelado como um ambiente do Gym, onde o tabuleiro 4x4 e a mecânica de adicionar peças aleatórias serão simulados.
- O tabuleiro é uma grade 4x4, onde:
  - Cada célula pode conter um número (0 para vazio, ou potências de 2 começando de 2)
  - O objetivo é alcançar a peça 2048 combinando peças com o mesmo valor
  - O jogo começa com duas peças (2 ou 4) colocadas aleatoriamente no tabuleiro
- Cada estado do ambiente será uma matriz 4x4 representando a posição das peças no tabuleiro. Cada número na matriz indicará o valor da peça ou zero se não houver peça naquele local.
- A recompensa será calculada com base na pontuação após cada jogada. A estrutura de recompensa é:
  - Penalidade de -10 pontos para jogadas inválidas (que não alteram o estado do tabuleiro)
  - Recompensa baseada no aumento da pontuação após a jogada (quando peças são combinadas)
  - Bônus de 0.1 pontos para cada peça não-vazia no tabuleiro
  - A recompensa total é calculada como: `reward = score_increase + (number_of_tiles * 0.1)`
  
### **4.2. Agente DQN**

- **Deep Q-Network (DQN)**: O agente usará uma rede neural para estimar a função Q, ou seja, o valor de cada ação em cada estado do jogo.
  
  - **Estado:** O estado é a configuração atual do tabuleiro (representado como uma matriz 4x4 ou vetor de tamanho 16).
  - **Ação:** As ações possíveis são as 4 direções: cima, baixo, esquerda, direita.
  - **Modelo:** Uma rede neural com uma camada de entrada que recebe o estado (tabuleiro), camadas ocultas (por exemplo, 2 ou 3 camadas densas), e uma camada de saída com 4 valores Q, um para cada direção.
  
  O modelo DQN será treinado por:
  1. **Exploração vs Exploração**: Durante o treinamento, o agente irá explorar (escolher ações aleatórias) e explorar (escolher a melhor ação com base no modelo) com base em uma política epsilon-greedy.
  2. **Replay Buffer**: O agente usará um buffer de replay para armazenar experiências passadas (estado, ação, recompensa, próximo estado) e amostrar aleatoriamente esses dados durante o treinamento, quebrando as dependências temporais entre as experiências.
  3. **Atualização da Rede Neural**: O modelo será atualizado utilizando o algoritmo de atualização do Q-Learning, com a rede neural sendo treinada para minimizar a diferença entre o valor Q estimado e o valor Q real.

### **4.3. Parâmetros de Treinamento**

- **Taxa de aprendizado (learning rate)**: `0.001`
- **Gamma (desconto futuro)**: `0.99`
- **Tamanho do batch**: `32`
- **Epsilon inicial**: `1.0` (exploração), que decresce ao longo do tempo.
- **Tamanho do replay buffer**: `10,000`
- **Número de episódios de treinamento**: `10,000`

### **4.4. Monitoramento e Salvamento**

- O modelo será treinado em episódios, e o progresso será monitorado a cada 100 episódios, gerando gráficos de evolução da pontuação média.
- A cada 500 episódios, o modelo será salvo em checkpoints para garantir que o treinamento não seja perdido.

---

## **5. Interface Gráfica (opcional)**

### **5.1. Renderização do Jogo**

- O jogo pode ser renderizado utilizando **Pygame** ou **Tkinter** para exibir a grade 4x4 e as peças. O tabuleiro será mostrado com as peças se movendo de acordo com a política do agente.
  
### **5.2. Interface com o Usuário**

- O usuário pode jogar contra o agente ou observar o treinamento. Uma interface gráfica simples pode ser criada com:
  - Um painel mostrando o estado atual do tabuleiro.
  - Um painel mostrando a pontuação atual.
  - Botões para fazer jogadas manuais (se desejar que o usuário interaja).
  - Exibição dos gráficos de progresso do treinamento.

### **5.3. Visualização dos Resultados**

- Durante e após o treinamento, gráficos de progresso da pontuação média, recompensas, e o valor da função Q podem ser exibidos para avaliar o desempenho do agente.

---

## **6. Implementação**

### **6.1. Funções principais**

- **Treinamento do agente (train.py)**:
  - Inicialização do ambiente e do agente DQN.
  - Loop de treinamento, onde o agente interage com o ambiente e aprende a maximizar a recompensa.
  - Atualização dos pesos da rede neural.
  
- **Renderização do jogo (game_renderer.py)**:
  - Função para desenhar o estado do tabuleiro.
  
- **Funções auxiliares (utils.py)**:
  - Funções para salvar modelos, carregar checkpoints, plotar gráficos, etc.

---

## **7. Resultados Esperados**

- O agente deve ser capaz de jogar o jogo 2048 de maneira eficiente, buscando maximizar a pontuação.
- Durante o treinamento, a pontuação do agente deve melhorar gradualmente à medida que ele aprende a jogar melhor.