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
HOMES   = 16280
YEARS   = np.arange(2026, 2038)   # 2026–2037 inclusive (12 years)
N_YEARS = len(YEARS)

# Replacement schedule decay — rebuilt each simulation
decay_mean = 0.10
decay_std  = 0.025

# Loan parameters
replace_mean = 7386
replace_std  = 2613.92
loan_cap     = 10000

# Repayment triggers
p_sale = 0.0878
p_refi = 0.0325
p_aid  = 0.03821   # 3.821% of homeowners receive aid — loan is forgiven, never repaid

# Inflation
inflation_mean = 0.0321
inflation_std  = 0.0193

# Weather — affects cost slightly, not replacement count
# (pipe is being replaced on schedule regardless)
p_weather_event   = 0.002
weather_cost_mult = 1.15    # costs run ~15% higher in a weather year

# Participation — only ~70% of scheduled homes actually follow through
participate_mean = 0.70
participate_std  = 0.05     # year-to-year variability in uptake


# -------------------------------------------------------
# Build replacement schedule
# -------------------------------------------------------
def build_schedule():
    # Fixed baseline schedule used only for display purposes in main.
    # Each simulation rebuilds its own schedule with stochastic decay.
    weights  = np.array([0.9**t for t in range(N_YEARS)], dtype=float)
    weights /= weights.sum()
    schedule = np.round(weights * HOMES).astype(int)
    diff = HOMES - schedule.sum()
    schedule[0] += diff
    return schedule

SCHEDULE = build_schedule()  # baseline for display only


# -------------------------------------------------------
# Single Simulation
# -------------------------------------------------------
def single_simulation():
    outflows = np.zeros(N_YEARS)
    inflows  = np.zeros(N_YEARS)
    total_loans_accepted = 0  # <-- counter for loans accepted this simulation

    # Rebuild schedule each simulation with stochastic decay
    decays   = np.random.normal(decay_mean, decay_std, size=N_YEARS)
    decays   = np.clip(decays, 0, 1)
    counts   = np.zeros(N_YEARS)
    counts[0] = HOMES
    for t in range(1, N_YEARS):
        counts[t] = counts[t-1] * (1 - decays[t])

    differences = np.zeros(N_YEARS)
    for t in range(1, N_YEARS):
        differences[t] = counts[t-1] - counts[t]
    schedule = np.round(differences).astype(int)

    for t in range(N_YEARS):
        # Draw a participation rate for this year (stochastic, mean 70%)
        p_participate = np.random.normal(participate_mean, participate_std)
        p_participate = np.clip(p_participate, 0, 1)

        # Only a binomial draw of scheduled homes actually follow through
        n_homes = np.random.binomial(schedule[t], p_participate)

        if n_homes == 0:
            continue

        total_loans_accepted += n_homes  # <-- count loans issued this year

        inflation        = np.random.normal(inflation_mean, inflation_std)
        inflation_factor = (1 + inflation) ** t

        # Weather shock — slightly inflates costs
        weather_event   = np.random.rand() < p_weather_event
        cost_multiplier = inflation_factor * (weather_cost_mult if weather_event else 1.0)

        # Draw replacement costs for all homes replaced this year
        costs = np.random.normal(replace_mean, replace_std, size=n_homes)
        costs = np.clip(costs, 0, loan_cap)
        costs *= cost_multiplier

        # Record outflows
        outflows[t] += costs.sum()

        # Simulate repayment timing for each loan
        for cost in costs:
            # Aid recipients have their loan forgiven — no repayment ever
            if np.random.rand() < p_aid:
                continue
            for future_t in range(t, N_YEARS):
                if (np.random.rand() < p_sale) or (np.random.rand() < p_refi):
                    inflows[future_t] += cost
                    break
            # If not repaid within window → long tail, excluded (conservative assumption)

    funding, cumulative = compute_funding_requirement(outflows, inflows)
    return funding, cumulative, total_loans_accepted


# -------------------------------------------------------
# Monte Carlo Runner
# -------------------------------------------------------
def monte_carlo(n_sim=1000):
    results      = []
    loans_counts = []  # <-- collect loan counts across simulations
    final_curve  = None

    for sim in range(n_sim):
        funding, cumulative, total_loans_accepted = single_simulation()
        results.append(funding)
        loans_counts.append(total_loans_accepted)
        if sim == 0:
            final_curve = cumulative

    results      = np.array(results)
    loans_counts = np.array(loans_counts)
    mean_funding = np.mean(results)
    p95_funding  = np.percentile(results, 95)

    se = stats.sem(results)
    ci = stats.t.interval(0.95, df=len(results) - 1, loc=mean_funding, scale=se)

    return mean_funding, p95_funding, ci, results, final_curve, loans_counts


# -------------------------------------------------------
# Grant Baseline (no repayments — upper bound)
# -------------------------------------------------------
def grant_baseline():
    outflows = np.zeros(N_YEARS)

    # Build stochastic schedule for this run
    decays  = np.random.normal(decay_mean, decay_std, size=N_YEARS)
    decays  = np.clip(decays, 0, 1)
    counts  = np.zeros(N_YEARS)
    counts[0] = HOMES
    for t in range(1, N_YEARS):
        counts[t] = counts[t-1] * (1 - decays[t])

    differences = np.zeros(N_YEARS)
    for t in range(1, N_YEARS):
        differences[t] = counts[t-1] - counts[t]
    n_helped = np.round(differences).astype(int)

    for t in range(N_YEARS):
        # Draw a participation rate for this year (stochastic, mean 70%)
        p_participate = np.random.normal(participate_mean, participate_std)
        p_participate = np.clip(p_participate, 0, 1)

        # Only a binomial draw of scheduled homes actually follow through
        n_homes = np.random.binomial(n_helped[t], p_participate)

        if n_homes == 0:
            continue

        inflation        = np.random.normal(inflation_mean, inflation_std)
        inflation_factor = (1 + inflation) ** t

        weather_event   = np.random.rand() < p_weather_event
        cost_multiplier = inflation_factor * (weather_cost_mult if weather_event else 1.0)

        costs = np.random.normal(replace_mean, replace_std, size=n_homes)
        costs = np.clip(costs, 0, loan_cap)
        costs *= cost_multiplier
        outflows[t] += costs.sum()

    return outflows.sum()


# -------------------------------------------------------
# Visualizations
# -------------------------------------------------------
# def plot_results(results, mean_funding, p95_funding, ci, cumulative_curve, loans_counts):
#     fig, axes = plt.subplots(1, 3, figsize=(20, 5))
#     fig.suptitle("LEAP Program — Monte Carlo Funding Analysis", fontsize=14)

#     # Plot 1: Distribution of funding needs
#     ax1 = axes[0]
#     ax1.hist(results, bins=40, color="steelblue", edgecolor="white")
#     ax1.axvline(mean_funding, color="orange", linestyle="--",
#                 label=f"Mean: ${mean_funding:,.0f}")
#     ax1.axvline(p95_funding,  color="red",    linestyle="--",
#                 label=f"95th pct: ${p95_funding:,.0f}")
#     ax1.axvspan(ci[0], ci[1], alpha=0.15, color="orange",
#                 label=f"95% CI: (${ci[0]:,.0f} – ${ci[1]:,.0f})")
#     ax1.set_xlabel("Funding Required ($)")
#     ax1.set_ylabel("Frequency")
#     ax1.set_title("Distribution of Peak Funding Needs")
#     ax1.legend(fontsize=8)

#     # Plot 2: Sample cumulative cash flow curve
#     ax2 = axes[1]
#     ax2.plot(YEARS, cumulative_curve, color="steelblue", marker="o")
#     ax2.axhline(0, color="gray", linestyle="--")
#     ax2.set_xlabel("Year")
#     ax2.set_ylabel("Cumulative Net Cash ($)")
#     ax2.set_title("Sample Cumulative Funding Gap (Single Simulation)")
#     ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

#     # Plot 3: Distribution of total loans accepted  <-- NEW
#     ax3 = axes[2]
#     ax3.hist(loans_counts, bins=40, color="mediumseagreen", edgecolor="white")
#     ax3.axvline(np.mean(loans_counts), color="orange", linestyle="--",
#                 label=f"Mean: {np.mean(loans_counts):,.0f}")
#     ax3.axvline(np.percentile(loans_counts, 95), color="red", linestyle="--",
#                 label=f"95th pct: {np.percentile(loans_counts, 95):,.0f}")
#     ax3.set_xlabel("Total Loans Accepted")
#     ax3.set_ylabel("Frequency")
#     ax3.set_title("Distribution of Total Loans Accepted")
#     ax3.legend(fontsize=8)

#     plt.tight_layout()
#     plt.savefig("leap_results.png", dpi=150)
#     plt.show()

def plot_results(results, mean_funding, p95_funding, ci, cumulative_curve, loans_counts):
    # Plot 1: Distribution of funding needs
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig1.suptitle("LEAP Program — Monte Carlo Funding Analysis", fontsize=14)
    ax1.hist(results, bins=40, color="steelblue", edgecolor="white")
    ax1.axvline(mean_funding, color="orange", linestyle="--",
                label=f"Mean: ${mean_funding:,.0f}")
    ax1.axvline(p95_funding,  color="red",    linestyle="--",
                label=f"95th pct: ${p95_funding:,.0f}")
    ax1.axvspan(ci[0], ci[1], alpha=0.15, color="orange",
                label=f"95% CI: (${ci[0]:,.0f} – ${ci[1]:,.0f})")
    ax1.set_xlabel("Funding Required ($)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Peak Funding Needs")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig("leap_funding_distribution.png", dpi=150)

    # Plot 2: Sample cumulative cash flow curve
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.suptitle("LEAP Program — Monte Carlo Funding Analysis", fontsize=14)
    ax2.plot(YEARS, cumulative_curve, color="steelblue", marker="o")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cumulative Net Cash ($)")
    ax2.set_title("Sample Cumulative Funding Gap (Single Simulation)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig2.tight_layout()
    fig2.savefig("leap_cumulative_cashflow.png", dpi=150)

    # Plot 3: Distribution of total loans accepted
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    fig3.suptitle("LEAP Program — Monte Carlo Funding Analysis", fontsize=14)
    ax3.hist(loans_counts, bins=40, color="mediumseagreen", edgecolor="white")
    ax3.axvline(np.mean(loans_counts), color="orange", linestyle="--",
                label=f"Mean: {np.mean(loans_counts):,.0f}")
    ax3.axvline(np.percentile(loans_counts, 95), color="red", linestyle="--",
                label=f"95th pct: {np.percentile(loans_counts, 95):,.0f}")
    ax3.set_xlabel("Total Loans Accepted")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Total Loans Accepted")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig("leap_loans_distribution.png", dpi=150)

    plt.show()

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":

    print(f"Total homes in pool    : {HOMES:,} (all replaced exactly once)")
    print(f"Schedule               : front-loaded, -10% per year (2026→2037)")
    print(f"Homes per year         : {SCHEDULE[0]:,} (2026) → {SCHEDULE[-1]:,} (2037)")
    print(f"Participation rate     : {participate_mean*100:.0f}% mean (±{participate_std*100:.0f}% std, stochastic per year)")
    print("Running Monte Carlo simulation...")

    mean_funding, p95_funding, ci, results, cumulative_curve, loans_counts = monte_carlo(n_sim=1000)

    print(f"\n--- Loan Model Results ---")
    print(f"Mean Funding Needed     : ${mean_funding:,.0f}")
    print(f"95% Confidence Interval : (${ci[0]:,.0f}  –  ${ci[1]:,.0f})")
    print(f"  → 95% confident the true mean funding need falls in this range.")
    print(f"95th Percentile         : ${p95_funding:,.0f}")
    print(f"  → 95% of simulated scenarios required less than this amount.")

    print(f"\n--- Loans Accepted ---")
    print(f"Mean Loans Accepted     : {np.mean(loans_counts):,.0f}")
    print(f"Std Dev                 : {np.std(loans_counts):,.0f}")
    print(f"Min / Max               : {np.min(loans_counts):,} / {np.max(loans_counts):,}")
    print(f"95th Percentile         : {np.percentile(loans_counts, 95):,.0f}")

    grant_runs = [grant_baseline() for _ in range(200)]
    mean_grant = np.mean(grant_runs)
    print(f"\n--- Grant Model Comparison ---")
    print(f"Mean Grant Funding      : ${mean_grant:,.0f}")
    print(f"Loan vs Grant Savings   : ${mean_grant - mean_funding:,.0f} "
          f"({(mean_grant - mean_funding) / mean_grant * 100:.1f}% reduction)")

    plot_results(results, mean_funding, p95_funding, ci, cumulative_curve, loans_counts)