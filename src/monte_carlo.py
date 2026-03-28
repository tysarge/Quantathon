import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# -------------------------------------------------------
# Core Funding Function
# -------------------------------------------------------
def compute_funding_requirement(outflows, inflows):
    net_cash = np.array(outflows) - np.array(inflows)
    cumulative = np.cumsum(net_cash)
    return np.max(cumulative), cumulative


# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
HOMES        = 72848
YEARS        = np.arange(2026, 2038)
N_YEARS      = len(YEARS)

# Pipe failure
p_base_failure   = 0.02     # annual base probability a pipe fails (~2% per year, ~50yr lifespan)
p_weather_event  = 0.002    # probability of an extreme weather year
weather_multiplier = 2.0    # failure rate multiplier during weather event

# Repair vs replacement
# Repairs excluded from program per problem scope — only replacements are loans
p_replace = 0.40            # 40% of failures → full replacement (loan issued)
                             # 60% of failures → repair (NOT funded by program, ignored)

# Loan parameters
replace_mean = 7386          # mean replacement cost
replace_std  = 2613.92       # standard deviation of replacement cost
loan_cap     = 10000         # max loan amount per LEAP rules

# Repayment triggers
p_sale = 0.0878              # annual probability of home sale
p_refi = 0.0325              # annual probability of refinancing / HELOC

# Inflation
inflation_mean = 0.0321
inflation_std  = 0.0394


# -------------------------------------------------------
# Single Simulation (vectorized over homes for speed)
# -------------------------------------------------------
def single_simulation():
    outflows = np.zeros(N_YEARS)
    inflows  = np.zeros(N_YEARS)

    for t in range(N_YEARS):
        # Inflation for this year
        inflation        = np.random.normal(inflation_mean, inflation_std)
        inflation_factor = (1 + inflation) ** t

        # Weather shock (Bernoulli draw)
        weather_event = np.random.rand() < p_weather_event
        p_failure     = p_base_failure * (weather_multiplier if weather_event else 1.0)

        # --- Vectorized over all homes ---
        # Step 1: which homes fail this year?
        failures = np.random.rand(HOMES) < p_failure

        # Step 2: of those, which get a full replacement (vs repair, which is excluded)?
        replacements = failures & (np.random.rand(HOMES) < p_replace)
        n_replacements = replacements.sum()

        if n_replacements == 0:
            continue

        # Step 3: draw replacement costs, cap at loan_cap, apply inflation
        costs = np.random.normal(replace_mean, replace_std, size=n_replacements)
        costs = np.clip(costs, 0, loan_cap)
        costs *= inflation_factor

        # Step 4: record outflows
        outflows[t] += costs.sum()

        # Step 5: simulate repayment timing for each replacement loan
        for cost in costs:
            for future_t in range(t, N_YEARS):
                if (np.random.rand() < p_sale) or (np.random.rand() < p_refi):
                    inflows[future_t] += cost
                    break
            # If not repaid within program window → long tail, not counted here

    funding, cumulative = compute_funding_requirement(outflows, inflows)
    return funding, cumulative


# -------------------------------------------------------
# Monte Carlo Runner
# -------------------------------------------------------
def monte_carlo(n_sim=1000):
    results     = []
    final_curve = None   # store one representative cumulative curve for plotting

    for sim in range(n_sim):
        funding, cumulative = single_simulation()
        results.append(funding)
        if sim == 0:
            final_curve = cumulative

    results      = np.array(results)
    mean_funding = np.mean(results)

    # 95th percentile: the funding level that covers 95% of simulated scenarios
    # (a distributional worst-case, not an inferential statistic)
    p95_funding  = np.percentile(results, 95)

    # 95% confidence interval: how precisely we've estimated the TRUE mean
    # using a t-distribution to account for finite sample size
    se  = stats.sem(results)                                  # standard error of the mean
    ci  = stats.t.interval(0.95, df=len(results)-1,
                           loc=mean_funding, scale=se)        # (lower, upper)

    return mean_funding, p95_funding, ci, results, final_curve


# -------------------------------------------------------
# Grant Baseline (no repayments — upper bound)
# -------------------------------------------------------
def grant_baseline():
    """
    Under a pure grant model, there are no inflows.
    Funding required = total outflows across all years.
    Run one simulation with inflows zeroed out.
    """
    outflows = np.zeros(N_YEARS)

    for t in range(N_YEARS):
        inflation        = np.random.normal(inflation_mean, inflation_std)
        inflation_factor = (1 + inflation) ** t

        weather_event = np.random.rand() < p_weather_event
        p_failure     = p_base_failure * (weather_multiplier if weather_event else 1.0)

        failures     = np.random.rand(HOMES) < p_failure
        replacements = failures & (np.random.rand(HOMES) < p_replace)
        n_replacements = replacements.sum()

        if n_replacements == 0:
            continue

        costs = np.random.normal(replace_mean, replace_std, size=n_replacements)
        costs = np.clip(costs, 0, loan_cap)
        costs *= inflation_factor
        outflows[t] += costs.sum()

    return outflows.sum()


# -------------------------------------------------------
# Visualizations
# -------------------------------------------------------
def plot_results(results, mean_funding, p95_funding, ci, cumulative_curve):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LEAP Program — Monte Carlo Funding Analysis", fontsize=14)

    # Plot 1: Distribution of funding needs
    ax1 = axes[0]
    ax1.hist(results, bins=40, color="steelblue", edgecolor="white")
    ax1.axvline(mean_funding, color="orange", linestyle="--",
                label=f"Mean: ${mean_funding:,.0f}")
    ax1.axvline(p95_funding,  color="red",    linestyle="--",
                label=f"95th pct: ${p95_funding:,.0f}")
    # Shade the 95% confidence interval around the mean
    ax1.axvspan(ci[0], ci[1], alpha=0.15, color="orange",
                label=f"95% CI: (${ci[0]:,.0f} – ${ci[1]:,.0f})")
    ax1.set_xlabel("Funding Required ($)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Peak Funding Needs")
    ax1.legend(fontsize=8)

    # Plot 2: Sample cumulative cash flow curve
    ax2 = axes[1]
    ax2.plot(YEARS, cumulative_curve, color="steelblue", marker="o")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cumulative Net Cash ($)")
    ax2.set_title("Sample Cumulative Funding Gap (Single Simulation)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    plt.savefig("leap_results.png", dpi=150)
    plt.show()


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":

    print("Running Monte Carlo simulation...")
    mean_funding, p95_funding, ci, results, cumulative_curve = monte_carlo(n_sim=1000)

    print(f"\n--- Loan Model Results ---")
    print(f"Mean Funding Needed  : ${mean_funding:,.0f}")
    print(f"95% Confidence Interval: (${ci[0]:,.0f}  –  ${ci[1]:,.0f})")
    print(f"  → We are 95% confident the true mean funding need falls in this range.")
    print(f"95th Percentile      : ${p95_funding:,.0f}")
    print(f"  → 95% of simulated scenarios required less than this amount.")

    # Grant comparison (single estimate — no repayments)
    grant_runs = [grant_baseline() for _ in range(200)]
    mean_grant = np.mean(grant_runs)
    print(f"\n--- Grant Model Comparison ---")
    print(f"Mean Grant Funding   : ${mean_grant:,.0f}")
    print(f"Loan vs Grant Savings: ${mean_grant - mean_funding:,.0f} "
          f"({(mean_grant - mean_funding) / mean_grant * 100:.1f}% reduction)")

    plot_results(results, mean_funding, p95_funding, ci, cumulative_curve)