"""
data_loader.py — Carregamento e validação dos CSVs de mercado.

Responsável por:
  - Carregar treino.csv e teste.csv
  - Converter coluna 'data' para datetime
  - Validar que todas as colunas necessárias para o PortfolioEnv existem
"""

import os
import pandas as pd


# Colunas obrigatórias para o PortfolioEnv
REQUIRED_COLUMNS = [
    "data",
    "ITUB4", "BOVA11", "BBAS3",
    "retorno_ITUB4", "retorno_BOVA11", "retorno_BBAS3",
    "CDI_decimal",
    "retorno_acima_cdi_ITUB4", "retorno_acima_cdi_BOVA11",
    "retorno_acima_cdi_BBAS3",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dados")


def _validate_columns(df: pd.DataFrame, filepath: str) -> None:
    """
    Verifica se o DataFrame contém todas as colunas necessárias.

    Raises
    ------
    ValueError
        Se alguma coluna obrigatória estiver faltando.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas faltando em '{filepath}': {missing}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )


def _load_csv(filename: str) -> pd.DataFrame:
    """
    Carrega um CSV do diretório de dados e prepara o DataFrame.

    Args
    ----------
    filename : str
        Nome do arquivo CSV

    Returns
    -------
    pd.DataFrame
        DataFrame com coluna 'data' convertida para datetime e
        todas as colunas numéricas validadas.
    """
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {filepath}\n"
            f"Verifique se o diretório 'dados/' contém '{filename}'."
        )

    df = pd.read_csv(filepath)
    _validate_columns(df, filepath)

    # Converter coluna de data
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)

    return df


def load_train_data() -> pd.DataFrame:
    """
    Carrega os dados de treinamento (2019-2023).

    Returns
    -------
    pd.DataFrame
        DataFrame com 1.227 dias úteis de dados de mercado.
    """
    df = _load_csv("treino.csv")
    print(f"[data_loader] Treino carregado: {len(df)} registros "
          f"({df['data'].iloc[0].date()} → {df['data'].iloc[-1].date()})")
    return df


def load_test_data() -> pd.DataFrame:
    """
    Carrega os dados de teste (2024).

    Returns
    -------
    pd.DataFrame
        DataFrame com 251 dias úteis de dados de mercado.
    """
    df = _load_csv("teste.csv")
    print(f"[data_loader] Teste carregado:  {len(df)} registros "
          f"({df['data'].iloc[0].date()} → {df['data'].iloc[-1].date()})")
    return df
