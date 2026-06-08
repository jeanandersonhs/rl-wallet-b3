# 📈 RL-Wallet-B3: Gestão de Portfólio com Aprendizado por Reforço

Este projeto implementa agentes de Aprendizado por Reforço (Reinforcement Learning - RL) para a gestão inteligente e autônoma de um portfólio de investimentos no mercado financeiro brasileiro (B3).

O objetivo principal é maximizar o retorno financeiro ajustado ao risco (Sharpe Ratio), decidindo dinamicamente como distribuir o capital entre diferentes ativos ao longo do tempo.

## Funcionalidades

- **Ambiente Customizado (PortfolioEnv)**:  modelando simples o mercado da B3 com custos de transação, volatilidade, drawdowns e taxas livres de risco (CDI).
- **Agentes de Reinforcement Learning**:
  - **Q-Learning Tabular** (Off-policy TD control)
  - **TD-Learning** (On-policy TD control)
- **Discretização de Estados Contínuos**:  converter 18 dimensões contínuas (retornos, médias móveis, excesso sobre CDI, volatilidade) em espaços discretos para aprendizado tabular.
- **Benchmarks Integrados**: Comparação automática de desempenho contra:
  - 100% CDI
  - Estratégia "Buy & Hold" (1/3 em cada ativo)
- **Varredura de Hiperparâmetros (Sweeps)**: Scripts para testar sensibilidade do agente em relação à discretização, taxas de decaimento (epsilon, alpha), custos de corretagem e rebalanceamento.

## Ativos Selecionados

Para evitar a explosão combinatória do espaço de estados, o escopo foi restrito a 3 ativos altamente líquidos da B3, com comportamentos distintos:
- **ITUB4**: Ação do setor financeiro (Itaú Unibanco).
- **BBAS3**: Ação do setor financeiro (Banco do Brasil).
- **BOVA11**: ETF que replica o Ibovespa (Exposição ampla ao mercado).

## 🛠️ Tecnologias e Bibliotecas

- **Linguagem**: Python 3.12+
- **Gerenciador de Dependências**: `pip` via `requirements.txt`
- **Core ML & Dados**: `numpy`, `pandas`
- **Visualização**: `matplotlib`, `seaborn`, Jupyter Notebooks

## ⚙️ Como Instalar

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/jeanandersonhs/rl-wallet-b3.git
   cd rl-wallet-b3
   ```

2. **Crie e ative um ambiente virtual**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## Como Usar

### Executar Treinamento Principal

Você pode rodar os arquivos principais para treinar e avaliar um agente específico. Por exemplo, para rodar o agente **(TD-Learning)**:

```bash
python main_td.py
```

Para rodar o agente **Q-Learning**:

```bash
python main.py
```
Isso irá treinar o agente (default: 500 episódios) e ao final salvará os logs e modelos na pasta `resultados/`.

### Rodar Experimentos (Hyperparameter Sweeps)

O projeto contém scripts dedicados para varredura de hiperparâmetros, essenciais para analisar o comportamento do agente sob diferentes condições (ou taxas de corretagem):

```bash
# Experimentos com TD-Learning
python experiments_td.py

# Experimentos com Q-Learning
python experiments_q_learning.py
```
*Você pode abrir os arquivos de experimento e ajustar a variável `N_EPISODES` para `500` antes de executar uma bateria completa.*

### Visualização e Métricas

Os resultados salvos podem ser analisados em profundidade através do Jupyter Notebook presente na pasta `notebooks/`:

```bash
python -m notebook notebooks/resultados.ipynb
```

## Estrutura do Projeto

```
rl-wallet-b3/
├── main.py                    # Orquestração do agente Q-Learning
├── main_td.py                 # Orquestração do agente SARSA
├── experiments_q_learning.py  # Script de varredura (Sweep) para Q-Learning
├── experiments_td.py          # Script de varredura (Sweep) para SARSA
├── requirements.txt           # Dependências do projeto
├── documentacao.txt           # Documentação matemática e arquitetural completa
├── ambiente/
│   └── portfolio_env.py       # Ambiente RL customizado (recompensas, MDP, transições)
├── agentes/
│   ├── Q_learning.py          # Implementação do Agente Q-Learning Tabular
│   └── TD_learning.py         # Implementação do Agente SARSA Tabular
├── src/
│   ├── data_loader.py         # Carregamento e split (treino/teste) dos CSVs
│   └── feature_engineering.py # Discretização e percentis de estado
├── dados/                     # Datasets de preços, CDI e indicadores
├── notebooks/
│   ├── resultados.ipynb       # Análise exploratória dos resultados e plots
│   └── metricas.py            # Helpers para cálculo das métricas financeiras
└── resultados/                # JSONs exportados, histórico de treino e modelos pickle
```

## Recompensa - Markov Decision Process(MDP)

A recompensa (`Reward`) a cada step de tempo (dia útil) foi desenhada para otimizar portfólios no mundo real e inclui:
- **Sharpe Instantâneo**: (Retorno da Carteira - CDI) / Volatilidade.
- **Bônus de Diversificação**: Incentivo para manter alocação distribuída (via índice HHI).
- **Penalidade de Drawdown**: Desconto caso a carteira atinja quedas abruptas desde o topo histórico.
- **Penalidade de Concentração**: Desconto se a carteira alocar muito peso (>60%) em um único ativo.
- **Custos de Transação**: Desconto proporcional à variação de carteira e ao fator de corretagem simulado.

