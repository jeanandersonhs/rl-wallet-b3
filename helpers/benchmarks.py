"""
Benchmarks simples para comparacao com os agentes.
"""


def compute_benchmarks(test_df, initial_balance: float) -> dict:
    """
    Calcula Buy & Hold igualmente ponderado e 100% CDI no periodo de teste.
    """
    ret_itub4 = test_df["retorno_ITUB4"].values
    ret_bova11 = test_df["retorno_BOVA11"].values
    ret_bbas3 = test_df["retorno_BBAS3"].values

    bh_returns = (ret_itub4 + ret_bova11 + ret_bbas3) / 3.0
    bh_value = initial_balance
    bh_values = [bh_value]
    for daily_return in bh_returns:
        bh_value *= 1.0 + daily_return
        bh_values.append(bh_value)

    cdi_returns = test_df["CDI_decimal"].values
    cdi_value = initial_balance
    cdi_values = [cdi_value]
    for daily_return in cdi_returns:
        cdi_value *= 1.0 + daily_return
        cdi_values.append(cdi_value)

    return {
        "buy_hold": {
            "values": bh_values,
            "total_return": (bh_values[-1] - initial_balance) / initial_balance,
            "final_value": bh_values[-1],
        },
        "cdi": {
            "values": cdi_values,
            "total_return": (cdi_values[-1] - initial_balance) / initial_balance,
            "final_value": cdi_values[-1],
        },
    }
