"""Design and evaluate beta-hedging portfolios."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

from config import DAILY_RETURNS, SECTION2_DIR, TICKERS


# Dates, universe, and model settings.
# U = {FXE, EWJ, GLD, QQQ, SHV, DBA, USO, XBI, ILF, EPP, FEZ}.
APPLE = "AAPL"
MARKET = "SPY"
UNIVERSE = [ticker for ticker in TICKERS if ticker not in {APPLE, MARKET}]
DECISION_DATE = pd.Timestamp("2026-01-02")
REALIZED_START = pd.Timestamp("2026-01-03")
REALIZED_END = pd.Timestamp("2026-02-28")
LOOKBACK_START = DECISION_DATE - pd.DateOffset(years=1)
RETURN_PENALTY = 2.0


def load_returns():
    # Read the prepared daily returns and keep the assets used below.
    returns = pd.read_csv(DAILY_RETURNS, index_col="Date", parse_dates=["Date"])
    return returns[TICKERS].sort_index()


def estimate_betas(returns):
    # Estimate each security's beta relative to SPY from sample covariance.
    market_variance = returns[MARKET].var()
    return returns.cov()[MARKET] / market_variance


def solve_target_beta(mean, covariance, betas, target_beta, return_penalty=0.0):
    # Solve the equality-constrained quadratic program; return_penalty adds
    # the expected-return term to the objective.
    """Solve min w'Σw - λρ'w subject to β'w=target and 1'w=1."""

    covariance = (covariance + covariance.T) / 2
    n_assets = len(mean)
    P = matrix(2 * covariance.to_numpy(), tc="d")
    q = matrix(-return_penalty * mean.to_numpy(), tc="d")
    A = matrix(
        np.vstack([betas.to_numpy(), np.ones(n_assets)]),
        tc="d",
    )
    b = matrix([target_beta, 1.0], tc="d")

    solvers.options["show_progress"] = False
    solution = solvers.qp(P, q, None, None, A, b)
    return pd.Series(np.asarray(solution["x"]).ravel(), index=mean.index)


# Compute ex-ante risk, expected return, beta, and weight sum.
def portfolio_stats(weights, mean, covariance, betas):
    weight_array = weights.to_numpy()
    variance = float(weight_array @ covariance.to_numpy() @ weight_array)
    expected_return = float(weights @ mean)
    return {
        "expected_return_daily": expected_return,
        "variance_daily": variance,
        "volatility_daily": np.sqrt(variance),
        "expected_return_annualized": expected_return * 252,
        "volatility_annualized": np.sqrt(variance) * np.sqrt(252),
        "ex_ante_beta": float(weights @ betas),
        "weight_sum": float(weights.sum()),
    }


# Compute realized performance and tail-risk statistics against the market.
def realized_stats(portfolio_returns, market_returns):
    mean = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    downside_deviation = np.sqrt(np.mean(np.minimum(portfolio_returns.to_numpy(), 0) ** 2))
    lower_five_percent = portfolio_returns.quantile(0.05)
    tail = portfolio_returns[portfolio_returns <= lower_five_percent]

    return {
        "realized_mean_daily": mean,
        "realized_volatility_daily": volatility,
        "realized_cumulative_return": (1 + portfolio_returns).prod() - 1,
        "sharpe_annualized": mean / volatility * np.sqrt(252),
        "sortino_annualized": mean / downside_deviation * np.sqrt(252),
        "var_95_loss": -lower_five_percent,
        "cvar_95_loss": -tail.mean(),
        "skewness": portfolio_returns.skew(),
        "kurtosis_excess": portfolio_returns.kurt(),
        "realized_beta": portfolio_returns.cov(market_returns) / market_returns.var(),
    }


def add_weights(records, portfolio, weights, date):
    for asset, weight in weights.items():
        records.append(
            {
                "portfolio": portfolio,
                "rebalance_date": date.strftime("%Y-%m-%d"),
                "asset": asset,
                "weight": weight,
            }
        )


def estimation_input_table(mean, covariance, betas):
    records = []
    for asset in mean.index:
        records.append(
            {
                "input_type": "expected_return",
                "asset": asset,
                "other_asset": "",
                "value": mean[asset],
            }
        )
        records.append(
            {
                "input_type": "beta",
                "asset": asset,
                "other_asset": MARKET,
                "value": betas[asset],
            }
        )
    for asset in covariance.index:
        for other_asset in covariance.columns:
            records.append(
                {
                    "input_type": "covariance",
                    "asset": asset,
                    "other_asset": other_asset,
                    "value": covariance.loc[asset, other_asset],
                }
            )
    return pd.DataFrame(records)


# Re-estimate the hedge from the previous 60 observations at each new week.
def weekly_dynamic_hedge(returns):
    realized = returns.loc[REALIZED_START:REALIZED_END]
    hedge_returns = []
    weight_records = []
    current_week = None
    current_weights = None

    for date, row in realized.iterrows():
        week = (date.isocalendar().year, date.isocalendar().week)
        if week != current_week:
            history = returns.loc[returns.index < date].tail(60)
            history_betas = estimate_betas(history)
            current_weights = solve_target_beta(
                history[UNIVERSE].mean(),
                history[UNIVERSE].cov(),
                history_betas[UNIVERSE],
                history_betas[APPLE],
            )
            add_weights(
                weight_records,
                "q4b_dynamic_beta_hedge_basket",
                current_weights,
                date,
            )
            current_week = week

        hedge_returns.append(row[APPLE] - row[UNIVERSE].dot(current_weights))

    return pd.Series(hedge_returns, index=realized.index), weight_records


# Plot cumulative returns and the distribution of daily returns.
def save_comparison_plot(realized_returns):
    cumulative = (1 + realized_returns).cumprod()
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    cumulative.plot(ax=axes[0])
    axes[0].set_title("Cumulative Realized Returns")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(alpha=0.3)

    for name in realized_returns.columns:
        axes[1].hist(
            realized_returns[name],
            bins=20,
            density=True,
            alpha=0.45,
            label=name,
        )
    axes[1].set_title("Distribution of Daily Realized Returns")
    axes[1].set_xlabel("Daily return")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.savefig(SECTION2_DIR / "section2_comparison.png", dpi=200)
    plt.close(figure)


# Main analysis and answers.
def run_section2():
    SECTION2_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    estimation = returns.loc[LOOKBACK_START:DECISION_DATE]
    realized = returns.loc[REALIZED_START:REALIZED_END]

    # 1. Sample Expected Returns, Covariance, and individual betas.
    mean = estimation.mean()
    covariance = estimation.cov()
    betas = estimate_betas(estimation)
    universe_mean = mean[UNIVERSE]
    universe_covariance = covariance.loc[UNIVERSE, UNIVERSE]
    universe_betas = betas[UNIVERSE]

    print("======= Data and estimation =======")
    print(f"Estimation window: {estimation.index.min().date()} to {estimation.index.max().date()}")
    print(f"Estimation observations: {len(estimation)}")
    print(f"Realized window: {realized.index.min().date()} to {realized.index.max().date()}")
    print("Sample expected returns, covariance matrix, and betas computed.")
    print("\nSample expected returns:")
    print(mean.to_string())
    print("\nIndividual betas:")
    print(betas.to_string())

    # 1. Two-fund Theorem with Target Beta.
    beta_a, beta_b, beta_target = 0.5, 1.5, 1.0
    alpha = (beta_b - beta_target) / (beta_b - beta_a)
    weights_a = solve_target_beta(universe_mean, universe_covariance, universe_betas, beta_a)
    weights_b = solve_target_beta(universe_mean, universe_covariance, universe_betas, beta_b)
    weights_theorem = alpha * weights_a + (1 - alpha) * weights_b
    weights_direct = solve_target_beta(
        universe_mean,
        universe_covariance,
        universe_betas,
        beta_target,
    )
    theorem_difference = (weights_theorem - weights_direct).abs().max()

    print("\n======= Two-fund theorem with target beta =======")
    print(f"alpha = {alpha:.6f}")
    print(f"Theorem beta = {weights_theorem @ universe_betas:.6f}")
    print(f"Direct beta = {weights_direct @ universe_betas:.6f}")
    print(f"Maximum theorem/direct weight difference = {theorem_difference:.3e}")

    # 2. Hedging a Long position in Apple.
    # 2(a) Compute the betas of the securities with respect to the S&P-500 index.
    beta_apple = betas[APPLE]

    # 2(b) Beta Hedging Portfolio.
    beta_hedge = solve_target_beta(
        universe_mean,
        universe_covariance,
        universe_betas,
        beta_apple,
    )
    enlarged_universe = [APPLE, *UNIVERSE]

    # 2(c) Beta Neutral Portfolio.
    beta_neutral = solve_target_beta(
        mean[enlarged_universe],
        covariance.loc[enlarged_universe, enlarged_universe],
        betas[enlarged_universe],
        0.0,
    )
    aggregated_hedge = pd.Series(0.0, index=enlarged_universe)
    aggregated_hedge[APPLE] = 1.0
    aggregated_hedge[UNIVERSE] = -beta_hedge

    hedged_returns = realized[APPLE] - realized[UNIVERSE].dot(beta_hedge)
    neutral_returns = realized[enlarged_universe].dot(beta_neutral)
    market_realized = realized[MARKET]

    print("\n======= Hedging a long position in Apple =======")
    print(f"Beta_A = {beta_apple:.6f}")
    print("Use the beta-hedge basket by taking a short position in it against the long AAPL position.")
    print(f"Expected daily return of the aggregated hedge = {aggregated_hedge @ mean[enlarged_universe]:.6f}")
    print("The beta-neutral portfolio is a fully invested portfolio in U plus AAPL with target beta 0.")

    q2c_comparison = []
    for name, weights, realized_returns in [
        ("aggregated_hedged_position", aggregated_hedge, hedged_returns),
        ("beta_neutral_portfolio", beta_neutral, neutral_returns),
    ]:
        row = {"section": "2.2c", "portfolio": name}
        row.update(
            portfolio_stats(
                weights,
                mean[enlarged_universe],
                covariance.loc[enlarged_universe, enlarged_universe],
                betas[enlarged_universe],
            )
        )
        row["realized_beta"] = realized_stats(realized_returns, market_realized)["realized_beta"]
        q2c_comparison.append(row)
    print("\nQ2(c) comparison: risk, expected return, and realized beta")
    print(pd.DataFrame(q2c_comparison).set_index("portfolio").to_string())

    # 3(a) Compute the realized daily returns of the two hedging strategies.
    # 3(b) Compare the two hedging strategies.
    comparison_returns = pd.DataFrame(
        {
            "beta_hedged_strategy": hedged_returns,
            "beta_neutral_portfolio": neutral_returns,
        }
    )
    comparison_metrics = []
    for name, series in comparison_returns.items():
        row = {"section": "2.3", "portfolio": f"q3_{name}"}
        row.update(realized_stats(series, market_realized))
        comparison_metrics.append(row)

    print("\n======= Comparison of the two hedging strategies =======")
    print(pd.DataFrame(comparison_metrics).set_index("portfolio").to_string())

    # 4(a) Improve the PnL of the Hedged strategy by adding a linear term.
    beta_hedge_return_term = solve_target_beta(
        universe_mean,
        universe_covariance,
        universe_betas,
        beta_apple,
        return_penalty=RETURN_PENALTY,
    )
    return_term_hedged_returns = realized[APPLE] - realized[UNIVERSE].dot(beta_hedge_return_term)
    return_term_aggregated_hedge = pd.Series(0.0, index=enlarged_universe)
    return_term_aggregated_hedge[APPLE] = 1.0
    return_term_aggregated_hedge[UNIVERSE] = -beta_hedge_return_term

    q4a_ex_ante = pd.DataFrame(
        {
            "Static Hedge": portfolio_stats(
                aggregated_hedge,
                mean[enlarged_universe],
                covariance.loc[enlarged_universe, enlarged_universe],
                betas[enlarged_universe],
            ),
            "Return-Term Hedge": portfolio_stats(
                return_term_aggregated_hedge,
                mean[enlarged_universe],
                covariance.loc[enlarged_universe, enlarged_universe],
                betas[enlarged_universe],
            ),
        }
    ).T[["expected_return_daily", "expected_return_annualized"]]

    # 4(b) Improve the hedge using 60 days and weekly re-hedging.
    dynamic_hedged_returns, dynamic_weight_records = weekly_dynamic_hedge(returns)

    return_term_metrics = {"section": "2.4a", "portfolio": "beta_hedged_lambda_2"}
    return_term_metrics.update(realized_stats(return_term_hedged_returns, market_realized))
    dynamic_metrics = {"section": "2.4b", "portfolio": "dynamic_beta_hedge_60d_weekly"}
    dynamic_metrics.update(realized_stats(dynamic_hedged_returns, market_realized))
    comparison_metrics.extend([return_term_metrics, dynamic_metrics])

    print("\n======= Chasing performance and dynamic hedging =======")
    base_cumulative = (1 + hedged_returns).prod() - 1
    return_term_cumulative = (1 + return_term_hedged_returns).prod() - 1
    print(f"Lambda = {RETURN_PENALTY:.1f}; base cumulative return = {base_cumulative:.6f}")
    print(f"Lambda = {RETURN_PENALTY:.1f}; return-term cumulative return = {return_term_cumulative:.6f}")
    print(f"Improved realized PnL in this sample: {return_term_cumulative > base_cumulative}")
    print("\nQ4(a) ex-ante expected return comparison:")
    print(q4a_ex_ante.to_string())
    print("The dynamic strategy uses a 60-day look-back and re-hedges at the start of each week.")
    print(pd.DataFrame([return_term_metrics, dynamic_metrics]).set_index("portfolio").to_string())

    static_realized_stats = realized_stats(hedged_returns, market_realized)
    dynamic_realized_stats = realized_stats(dynamic_hedged_returns, market_realized)
    q4b_metric_fields = {
        "Cumulative Return": "realized_cumulative_return",
        "Volatility": "realized_volatility_daily",
        "Sharpe Ratio": "sharpe_annualized",
        "Realized Beta": "realized_beta",
        "CVaR 95%": "cvar_95_loss",
    }
    q4b_comparison = pd.DataFrame(
        {
            "Static Hedge": [static_realized_stats[field] for field in q4b_metric_fields.values()],
            "Dynamic Hedge": [dynamic_realized_stats[field] for field in q4b_metric_fields.values()],
        },
        index=q4b_metric_fields,
    )
    q4b_comparison["Difference (Dynamic - Static)"] = (
        q4b_comparison["Dynamic Hedge"] - q4b_comparison["Static Hedge"]
    )
    print("\nQ4(b) static versus dynamic hedge comparison:")
    print(q4b_comparison.to_string())

    # Save results for the report and later sections.
    weights_records = []
    static_weights = {
        "q1_target_beta_0.5": weights_a,
        "q1_target_beta_1.5": weights_b,
        "q1_theorem_target_beta_1": weights_theorem,
        "q1_direct_target_beta_1": weights_direct,
        "q2b_beta_hedge_basket": beta_hedge,
        "q2b_aggregated_hedged_position": aggregated_hedge,
        "q2c_beta_neutral_portfolio": beta_neutral,
        "q2.4a_beta_hedge_lambda_2": beta_hedge_return_term,
        "q2.4a_aggregated_hedged_position": return_term_aggregated_hedge,
    }
    for portfolio, weights in static_weights.items():
        add_weights(weights_records, portfolio, weights, DECISION_DATE)
    weights_records.extend(dynamic_weight_records)

    all_metrics = []
    for portfolio, weights in static_weights.items():
        assets = weights.index
        row = {"section": portfolio.split("_")[0], "portfolio": portfolio}
        row.update(
            portfolio_stats(
                weights,
                mean[assets],
                covariance.loc[assets, assets],
                betas[assets],
            )
        )
        all_metrics.append(row)
    all_metrics.extend(comparison_metrics)
    all_metrics.extend(q2c_comparison)
    all_metrics.append(
        {
            "section": "2.1",
            "portfolio": "two_fund_alpha",
            "alpha": alpha,
            "theorem_direct_max_weight_difference": theorem_difference,
        }
    )

    realized_output = comparison_returns.copy()
    realized_output["beta_hedged_lambda_2"] = return_term_hedged_returns
    realized_output["dynamic_beta_hedge_60d_weekly"] = dynamic_hedged_returns
    realized_output[MARKET] = market_realized

    pd.DataFrame(weights_records).to_csv(SECTION2_DIR / "section2_weights.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(SECTION2_DIR / "section2_metrics.csv", index=False)
    estimation_input_table(mean, covariance, betas).to_csv(
        SECTION2_DIR / "section2_estimation_inputs.csv",
        index=False,
    )
    realized_output.to_csv(SECTION2_DIR / "section2_realized_returns.csv", index_label="Date")
    q4b_comparison.to_csv(SECTION2_DIR / "section2_dynamic_comparison.csv", index_label="Metric")
    save_comparison_plot(comparison_returns)
    print(f"\nOutputs saved to: {SECTION2_DIR}")


if __name__ == "__main__":
    run_section2()
