import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sm

def port_creation_and_statistics(df, lags):
    ew_results = []
    vw_results = []

    for lag in lags:
        stk = (
            df.filter(
                (pl.col('caldt') >= dt.date(1965, 1, 1)) &
                (pl.col('caldt') <= dt.date(2002, 12, 31))
            )
            .with_columns(
                me = (pl.when((pl.col('prc') * pl.col('shr')) > 1e-6)
                    .then((pl.col('prc') * pl.col('shr')) / 1000.0)
                    .otherwise(None)),
            )
            .with_columns(
                pl.col('ret').shift(lag).over('permno').alias('retlag'),
                pl.col('prc','me').shift().over('permno').name.suffix('lag')
            )
            .filter(
                pl.col('shrcd').is_between(10,11) &
                pl.col('melag').is_not_null() &
                pl.col('retlag').is_not_null() &
                (pl.col('prclag') > 4.999)
            )
            .with_columns(
                pl.col('retlag')
                .qcut(5, labels=[f"p{x}" for x in range(5)], allow_duplicates=True)
                .over('caldt')
                .alias('port'),
                (pl.col('melag') * pl.col('ret')).alias('meret')
            )
            .filter(pl.col('ret').is_not_null())
        )

        port = (
            stk.group_by(['caldt', 'port'])
            .agg(
                (pl.col('ret') * 100).mean().alias('ewret'),
                (pl.col('meret') * 100).sum().alias('vwret'),
                pl.col('melag').sum().alias('wtotal'),
            )
            .with_columns(
                vwret = pl.col('vwret') / pl.col('wtotal')
            )
            .sort(['caldt', 'port'])
            .collect()
        )

        ew = (
            port.pivot(index='caldt', on='port', values='ewret')
            .to_pandas()
            .set_index('caldt')
        )

        vw = (
            port.pivot(index='caldt', on='port', values='vwret')
            .to_pandas()
            .set_index('caldt')
        )

        # ---- EW row ----
        ew_row = {"lag": lag}
        for col in ew.columns:
            ew_row[col] = ew[col].mean()

        if "p4" in ew.columns and "p0" in ew.columns:
            p4_mean = ew["p4"].mean()
            p0_mean = ew["p0"].mean()
            ew_row["spread"] = p4_mean - p0_mean

        ew_results.append(ew_row)


        # ---- VW row ----
        vw_row = {"lag": lag}
        for col in vw.columns:
            vw_row[col] = vw[col].mean()

        if "p4" in vw.columns and "p0" in vw.columns:
            p4_mean = vw["p4"].mean()
            p0_mean = vw["p0"].mean()
            vw_row["spread"] = p4_mean - p0_mean

        vw_results.append(vw_row)

    # Convert to Polars tables
    ew_table = pl.DataFrame(ew_results).sort("lag")
    vw_table = pl.DataFrame(vw_results).sort("lag")

    print("EW Table")
    print(ew_table)

    print("\nVW Table")
    print(vw_table)

    return ew_table, vw_table

# ----- ATTEMPT TO RECREATE TABLE 1------

def recreate_table_1(df: pl.DataFrame, lags, nw_lags: int = 12):

    def run_single_lag(df, lag):

        # 1. Create lagged return
        df_lag = (
            df.sort(["permno", "caldt"])
            .with_columns(
                pl.col("ret").shift(lag).over("permno").alias("retlag_k"),
                pl.col('prc').shift().over('permno').name.suffix('lag')
            )
            .filter(
                (pl.col("caldt") >= dt.date(1965, 1, 1)) &
                (pl.col("caldt") <= dt.date(2002, 12, 31)) &
                (pl.col("shrcd").is_between(10, 11)) # & (pl.col('prclag') > 4.999)
            )
        )
        
        # 2. Monthly cross-sectional regressions
        gammas = []
        for caldt, sub_df in df_lag.group_by("caldt", maintain_order=True):

            sub_df = sub_df.drop_nulls(["ret", "retlag_k"])

            # if sub_df.height < 10:
            #     continue

            X = sm.add_constant(sub_df.select("retlag_k").to_pandas(),has_constant="add")
            y = sub_df.select("ret").to_pandas()
            model = sm.OLS(y, X).fit()

            gammas.append(model.params["retlag_k"])

        gamma_series = np.array(gammas)
        
        # 3. Fama-MacBeth mean
        mean_gamma = gamma_series.mean() * 100

        # 4. Newey-West t-stat
        T = len(gamma_series)
        X = np.ones((T, 1))

        nw_res = sm.OLS(gamma_series, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": nw_lags}
        )

        t_stat = nw_res.tvalues[0]

        return mean_gamma, t_stat, T

    
    # 5. Loop over all lags (Table 1)
    results = []

    for k in lags:
        mean_gamma, t_stat, T = run_single_lag(df, k)

        results.append({
            "lag": k,
            "mean_gamma": mean_gamma,
            "t_stat": t_stat,
            "n_months": T
        })

    return pl.DataFrame(results)

def plot_table_1_results(table_1, lags_table_1):

    x_vals = np.array(lags_table_1)

    table_y_vals = np.array([
        -5.03, -0.07, 1.36, 0.58, 0.96, 0.98, 1.06, 0.58, 1.31, 0.85, 1.39, 2.61,
        1.30, 1.27, 1.29, 0.62, 1.08, 1.03, 0.93, 1.41, 1.34, 1.68, 1.19, 0.70,
        0.78, 1.29, 1.43, 1.21, 1.14, 0.00, 1.14
    ])

    our_y_vals = np.array(table_1["mean_gamma"])

    diff = our_y_vals - table_y_vals
    rmse = np.sqrt(np.mean(diff ** 2))

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, table_y_vals, label="Paper")
    plt.plot(x_vals, our_y_vals, label="Replication")

    plt.legend()
    plt.xlabel("Lag")
    plt.ylabel(r'$\gamma_{k,t}$ (Coefficient for lagged return $r_{i,t-k}$)')
    plt.title(f"Table 1 Replication WITHOUT price filter (RMSE = {rmse:.4f})")

    plt.show()
    plt.savefig('graph2.png')

    return rmse
    
def plot_portfolio_returns(ew_table, vw_table, df_name):
    port_cols = ['p0', 'p1', 'p2', 'p3', 'p4']
    x = list(range(5))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f'{df_name} — average return by portfolio and lag')

    for ax, (table, label) in zip(axes, [(ew_table, 'EW'), (vw_table, 'VW')]):
        for row in table.iter_rows(named=True):
            y = [row[col] for col in port_cols]
            ax.plot(x, y, marker='o', label=f'lag={row["lag"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(port_cols)
        ax.set_title(f'{label} average return (%)')
        ax.set_xlabel('Portfolio')
        ax.set_ylabel('Mean monthly return (%)')
        ax.axhline(0.0, color='k', lw=0.5)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{df_name}_by_port.png')
    # plt.show()

def our_extension():
    dstk = (
        pl.scan_ipc('crsp_daily.ftr', memory_map=False)
        .with_columns(
            prodret = (1 + pl.col('ret')).cum_prod().over('permno'),
            mdt = pl.col('caldt').dt.truncate('1mo')
        )
        .filter(
            pl.col('caldt').dt.day() <= 15
        )
        .group_by(['permno','mdt'],maintain_order=True).last()
        .with_columns(
            ret = pl.col('prodret')/pl.col('prodret').shift().over('permno') - 1
        )
        .select(['permno','caldt','prc','ret','shr','excd','shrcd'])
    )

    return dstk

def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)

    mstk = pl.scan_ipc('crsp_monthly.ftr', memory_map=False)
    dstk = our_extension()

    # lags_table_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
    #     12, 24, 36, 48, 60, 72, 84, 96, 108,
    #     120, 132, 144, 156, 168, 180, 192, 204,
    #     216, 228, 240]
    # table_1 = recreate_table_1(mstk.collect(), lags=lags_table_1)
    # print('TABLE 1')
    # print(table_1)
    # plot_table_1_results(table_1, lags_table_1)

    lags_table_2 = [12 * y for y in range(1, 21)]
    # print('REPLICATION')
    # port_creation_and_statistics(mstk, lags_table_2)
    # print('EXTENSION')
    # port_creation_and_statistics(dstk, lags_table_2)

    print('REPLICATION')
    ew_table, vw_table = port_creation_and_statistics(mstk, lags_table_2)
    plot_portfolio_returns(ew_table, vw_table, 'Replication_Monotone')

    print('EXTENSION')
    ew_table, vw_table = port_creation_and_statistics(dstk, lags_table_2)
    plot_portfolio_returns(ew_table, vw_table, 'Extension_Monotone')

if __name__ == "__main__":
    main()

# What specific results from table are we recreating with the portfolios?
# Is it annual in Table 2?
# Should we not drop the price for Table 1 to get a closer replication?
