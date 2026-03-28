import numpy as np

def computing_reqs(outflows, inflows):
    net_cash = np.array(outflows) - np.array(inflows) 
    return np.max(cumulative)

# ----- parameters ----
homes =
years = np.arange(2026, 2037)  
n_years = len(years)            

p_weather_failure =
weather_multiplier =

repair_mean =
replace_mean =
loan_cap =

p_sale =
p_refi =

inflation_mean =
inflation_std =


def single_simulation():
    outflows = np.zeros(n_years)
    inflows = np.zeros(n_years)

    for t in range(n_years):
        inflation = np.random.normal(inflation_mean, inflation_std)
        inflation_factor = (1 + inflation) ** t

        weather_event = np.random.rand() < p_weather_failure  

        for home in range(homes):
            p_failure = p_weather_failure * (weather_multiplier if weather_event else 1)  

            if np.random.rand() < p_failure:
                cost = np.random.normal(replace_mean, 1500)
                cost = min(cost, loan_cap)
                cost *= inflation_factor          
                outflows[t] += cost              

                for future_t in range(t, n_years):
                    if (np.random.rand() < p_sale) or (np.random.rand() < p_refi): 
                        inflows[future_t] += cost
                        break
            else:
                cost = np.random.normal(repair_mean, 500)
                cost *= inflation_factor
                outflows[t] += cost

    funding = computing_reqs(outflows, inflows)  
    return funding                               


def monte_carlo(n_sim=1000):
    results = []
    for sim in range(n_sim):
        funding = single_simulation()
        results.append(funding)
    results = np.array(results)

    mean_funding = np.mean(results)
    p95_funding = np.percentile(results, 95)

    return mean_funding, p95_funding, results