# 📈 RL-Wallet-B3: Gestão de Portfólio com Aprendizado por Reforço

Este projeto implementa agentes de Aprendizado por Reforço (Reinforcement Learning - RL) para a gestão inteligente e autônoma de um portfólio de investimentos no mercado financeiro brasileiro (B3).

O objetivo principal é maximizar o retorno financeiro ajustado ao risco (Sharpe Ratio), decidindo dinamicamente como distribuir o capital entre diferentes ativos ao longo do tempo.

## 🎯 Principais Características

### Agentes de Reinforcement Learning (Refatorados)

- **Q-Learning Tabular** (Off-policy TD control) — herda de `BaseAgent`
- **TD-Learning (SARSA)** (On-policy TD control) — herda de `BaseAgent`
- **BaseAgent** — classe base compartilhada com funcionalidades comuns a ambos os agentes, incluindo:
  - Cálculo automático de bins de discretização via simulação
  - Política epsilon-greedy com decaimento
  - Rastreamento de recompensas médias e descontadas

### Ambiente Customizado

- **PortfolioEnv**: Ambiente MDP que modela o mercado da B3 com:
  - Custos de transação realistas
  - Penalidades por concentração excessiva
  - Bônus por diversificação
  - Penalidades por drawdown
  - Taxa livre de risco (CDI) integrada

### Discretização Inteligente de Estados

- Conversão automática de 18 dimensões contínuas em espaço discreto
- Bins calculados via simulação monte-carlo no início do treinamento
- Features incluem: retornos diários, médias móveis, excesso sobre CDI, volatilidade

### Benchmarks Integrados

- Comparação automática contra:
  - 100% CDI (sem risco)
  - Estratégia "Buy & Hold" (1/3 em cada ativo)

### Varredura de Hiperparâmetros (Hyperparameter Sweeps)

- Scripts experimentais para testar sensibilidade em relação a:
  - Discretização (`n_bins`)
  - Taxas de decaimento (epsilon, alpha)
  - Custos de transação
  - Rebalanceamento

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

Você pode rodar os arquivos principais para treinar e avaliar um agente específico:

**Para Q-Learning**:
```bash
python main.py
```

**Para TD-Learning**:
```bash
python main_td.py
```

Isso irá:
1. Carregar dados de treino e teste
2. Criar o ambiente `PortfolioEnv`
3. Instanciar o agente (Q-Learning ou TD-Learning)
4. Calcular bins de discretização via simulação monte-carlo (BaseAgent)
5. Treinar por N episódios (default: 500)
6. Avaliar no conjunto de teste
7. Comparar com benchmarks (Buy&Hold, 100% CDI)
8. Salvar agente, logs e métricas em `resultados/`

### Rodar Experimentos (Hyperparameter Sweeps)

O projeto contém scripts dedicados para varredura de hiperparâmetros e análise de sensibilidade:

```bash
# Experimentos com TD-Learning
python experiments_td.py

# Experimentos com Q-Learning
python experiments_q_learning.py
```

Dica: Ajuste `N_EPISODES` nos scripts antes de executar varreduras completas (padrão: 5000 para sweeps).

### Visualização e Análise de Resultados

Os resultados salvos podem ser analisados através do Jupyter Notebook:

```bash
jupyter notebook notebooks/resultados.ipynb
```

O notebook inclui:
- Comparação de desempenho Q-Learning vs TD-Learning
- Gráficos de valor de carteira ao longo do tempo
- Tabelas de métricas (Sharpe Ratio, Retorno Total, Max Drawdown)
- Alocação dinâmica de pesos dos agentes
- Análise de comparativos com benchmarks

## 📁 Estrutura do Projeto (Refatorado)

```
rl-wallet-b3/
├── main.py                       # Ponto de entrada: treino e avaliação Q-Learning
├── main_td.py                    # Ponto de entrada: treino e avaliação TD-Learning
├── experiments_q_learning.py     # Script de varredura (Sweep) para Q-Learning
├── experiments_td.py             # Script de varredura (Sweep) para TD-Learning
├── requirements.txt              # Dependências do projeto
├── documentacao.txt              # Documentação matemática e arquitetural completa
│
├── ambiente/
│   └── portfolio_env.py          # Ambiente RL customizado (MDP, recompensas, transições)
│
├── agentes/                      # Agentes RL refatorados com classe base compartilhada
│   ├── base_agent.py             # BaseAgent: classe base com Q-table, discretização e treinamento
│   ├── Q_learning.py             # AgentQLearning: Q-Learning off-policy
│   └── TD_learning.py            # AgentTD: TD-Learning on-policy (SARSA)
│
├── helpers/                      # Utilitários refatorados
│   ├── data_loader.py            # Carregamento, validação e split (treino/teste) dos CSVs
│   ├── feature_engineering.py    # Discretização automática e cálculo de bins
│   └── benchmarks.py             # Cálculo de benchmarks (Buy&Hold, CDI)
│
├── dados/                        # Datasets de preços, CDI e indicadores técnicos
├── notebooks/                    # Análise e visualização de resultados
│   └── resultados.ipynb          # Jupyter notebook com plots e métricas
├── tests/                        # Testes unitários
└── resultados/                   # Outputs: JSONs, histórico de treino, modelos (pickle)
```

### Refatorações Recentes

✨ **v2.0 - Refatoração com BaseAgent**:
- Criação de `BaseAgent` com funcionalidades compartilhadas entre Q-Learning e TD-Learning
- Simplificação de código reduzindo duplicação entre agentes
- Rastreamento automático de recompensas médias e descontadas
- Migração de lógica de dados para módulo `helpers/`
- Melhor separação de responsabilidades (MDP, agentes, utilitários)

## 🏗️ Arquitetura dos Agentes

### Hierarquia de Classes

```
BaseAgent (agentes/base_agent.py)
├── AgentQLearning (agentes/Q_learning.py)
└── AgentTD (agentes/TD_learning.py)
```

### BaseAgent - Classe Base Comum

A classe `BaseAgent` centraliza funcionalidades compartilhadas:

- **Discretização Inteligente**: Cálcula bins via simulação Monte Carlo (`compute_bins_from_simulation`)
- **Q-Table Management**: Tabela Q persistente com `defaultdict(lambda: np.zeros(n_actions))`
- **Política Epsilon-Greedy**: Exploração vs Exploração com decaimento
- **Persistência**: Métodos `save()` e `load()` usando pickle
- **Rastreamento de Métricas**: 
  - `mean_reward` por episódio
  - `discounted_reward` (com fator gamma)
  - `td_errors` para análise de convergência

### Diferenças entre Q-Learning e TD-Learning

| Aspecto | Q-Learning | TD-Learning (SARSA) |
|---------|-----------|---------------------|
| **Tipo** | Off-policy | On-policy |
| **Target** | `reward + γ * max(Q[s'])` | `reward + γ * Q[s', a']` |
| **Método `update()`** | Recebe `next_state` | Recebe `next_state` e `next_action` |
| **Exploração** | Aprende política ótima mesmo explorando | Aprende a política atual |
| **Convergência** | Mais lenta, mas ótima | Mais rápida, convergência garantida |

## 🔌 Exemplos de Uso (API)

### Treinar um Agente Q-Learning

```python
from ambiente.portfolio_env import PortfolioEnv
from agentes.Q_learning import AgentQLearning
from helpers.data_loader import load_train_data

# Carregar dados
train_df = load_train_data()

# Criar ambiente
env = PortfolioEnv(train_df, initial_balance=100_000.0)

# Instanciar agente Q-Learning
agent = AgentQLearning(
    env=env,
    n_bins=5,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    n_sim_episodes=10,  # para calcular bins
    seed=42
)

# Treinar
history = agent.train(env, n_episodes=500, log_interval=50)

# Salvar agente
agent.save("resultados/q_learning_agent.pkl")
```

### Usar Agente para Inferência (Teste)

```python
from helpers.data_loader import load_test_data
from agentes.Q_learning import AgentQLearning

# Carregar agente treinado
agent = AgentQLearning.load("resultados/q_learning_agent.pkl")

# Carregar dados de teste
test_df = load_test_data()
env_test = PortfolioEnv(test_df, initial_balance=100_000.0)

# Executar episódio de teste (sem exploração)
state, info = env_test.reset()
done = False
total_reward = 0

while not done:
    # Usar política greedy (epsilon=0)
    state_discrete = discretize_state(state, agent.bins)
    action = agent.select_action(state_discrete, epsilon=0.0)
    state, reward, done, truncated, info = env_test.step(action)
    total_reward += reward

print(f"Recompensa total no teste: {total_reward}")
```

## Recompensa - Markov Decision Process(MDP)

A recompensa (`Reward`) a cada step de tempo (dia útil) foi desenhada para otimizar portfólios no mundo real e inclui:

- **Sharpe Instantâneo**: (Retorno da Carteira - CDI) / Volatilidade.
- **Bônus de Diversificação**: Incentivo para manter alocação distribuída (via índice HHI).
- **Penalidade de Drawdown**: Desconto caso a carteira atinja quedas abruptas desde o topo histórico.
- **Penalidade de Concentração**: Desconto se a carteira alocar muito peso (>60%) em um único ativo.
- **Custos de Transação**: Desconto proporcional à variação de carteira e ao fator de corretagem simulado.

## 📚 Documentação Completa

Para uma análise profunda sobre:
- Formulação matemática do MDP
- Equações de Q-Learning e TD-Learning
- Detalhes da discretização de estados
- Explicação completa da função de recompensa
- Benchmarks e métricas de performance

Consulte o arquivo **`documentacao.txt`** na raiz do projeto (referência: seções 1-9).

## 🧪 Testes

O projeto inclui testes unitários para validação de componentes:

```bash
pytest tests/ -v
```

Principais testes:
- `test_q_agent.py` — Validação do AgentQLearning
- `test_td_agent.py` — Validação do AgentTD
- `test_portfolio_env.py` — Validação do PortfolioEnv

## 📊 Resultados Esperados

Após treinar com as configurações padrão (500 episódios):

- **Q-Learning**: Sharpe Ratio ~1.2-1.5 (superior a Buy&Hold)
- **TD-Learning**: Sharpe Ratio ~0.9-1.2 (mais conservador, menos volatilidade)
- **Buy & Hold**: Sharpe Ratio ~0.3-0.8 (baseline)
- **100% CDI**: Sharpe Ratio ~0.0 (sem risco)

Os logs de treinamento incluem gráficos de convergência, evolução de recompensas e alocações dinâmicas.

## 🤝 Contribuindo

Para contribuir com melhorias:

1. Faça fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/melhoria`)
3. Commit suas mudanças (`git commit -am 'Add melhoria'`)
4. Push para a branch (`git push origin feature/melhoria`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob licença MIT. Veja [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

Desenvolvido por **Jean Anderson** como exploração de Reinforcement Learning aplicado a gestão de portfólios.
