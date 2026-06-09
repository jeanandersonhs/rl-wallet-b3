# RL-Wallet-B3: Gestão de Portfólio com Aprendizado por Reforço

Este projeto implementa agentes de Aprendizado por Reforço (Reinforcement Learning - RL) para a gestão inteligente e autônoma de um portfólio de investimentos no mercado financeiro brasileiro (B3).

O objetivo principal é maximizar o retorno financeiro ajustado ao risco (Sharpe Ratio), decidindo dinamicamente como distribuir o capital entre diferentes ativos ao longo do tempo.

## Principais Características

## Ativos Selecionados

Para evitar a explosão combinatória do espaço de estados, o escopo foi restrito a 3 ativos da B3, com comportamentos distintos:

- **ITUB4**: Ação do setor financeiro (Itaú Unibanco).
- **BBAS3**: Ação do setor financeiro (Banco do Brasil).
- **BOVA11**: ETF que replica o Ibovespa (Exposição ampla ao mercado).

## Tecnologias e Bibliotecas

- **Linguagem**: Python 3.12+
- **Gerenciador de Dependências**: `pip` via `requirements.txt`
- **Core ML & Dados**: `numpy`, `pandas`
- **Visualização**: `matplotlib`, `seaborn`, Jupyter Notebooks

##  Como Instalar

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
7. Comparar com benchmarks (100% CDI)
8. Salvar agente, logs e métricas em `resultados/`

### Rodar Experimentos (Hyperparameter Sweeps)

O projeto contém scripts dedicados para varredura de hiperparâmetros e análise de sensibilidade:

```bash
# Experimentos com TD-Learning
python experiments_td.py

# Experimentos com Q-Learning
python experiments_q_learning.py
```


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

## Estrutura do Projeto (Refatorado)

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

### Diferenças entre Q-Learning e TD-Learning

| Aspecto | Q-Learning | TD-Learning  |
|---------|-----------|---------------------|
| **Tipo** | Off-policy | On-policy |
| **Target** | `reward + γ * max(Q[s'])` | `reward + γ * Q[s', a']` |
| **Método `update()`** | Recebe `next_state` | Recebe `next_state` e `next_action` |
| **Exploração** | Aprende política ótima mesmo explorando | Aprende a política atual |
| **Convergência** | Mais lenta, mas ótima | Mais rápida, convergência garantida |


## Contribuindo

Para contribuir com melhorias:

1. Faça fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/melhoria`)
3. Commit suas mudanças (`git commit -am 'Add melhoria'`)
4. Push para a branch (`git push origin feature/melhoria`)
5. Abra um Pull Request

##  Autores

Desenvolvido por **Jean Anderson e Rebeca Oliveira** como exploração de Reinforcement Learning aplicado a gestão de portfólios.
