import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sf_quant.data as sfd
import pandas as pd
from finance_byu.regtables import Regtable

def port_creation_and_statistics(df, lags, start, end):
    ew_results = []
    vw_results = []

    for lag in lags:
        stk = (
            df.filter(
                (pl.col('caldt') >= dt.date(start[0], start[1], start[2])) &
                (pl.col('caldt') <= dt.date(end[0], end[1], end[2]))
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
        .with_columns(
            (pl.col("p4") - pl.col("p0")).alias("spread")
        )
        .to_pandas()
        .set_index("caldt")
        )
        
        vw = (
        port.pivot(index='caldt', on='port', values='vwret')
        .with_columns(
            (pl.col("p4") - pl.col("p0")).alias("spread")
        )
        .to_pandas()
        .set_index("caldt")
        )

        # ---- EW row ----
        ew_row = {"lag": lag}
        for col in ew.columns:
            ew_row[col] = ew[col].mean()

        ew_results.append(ew_row)

        # ---- VW row ----
        vw_row = {"lag": lag}
        for col in vw.columns:
            vw_row[col] = vw[col].mean()

        vw_results.append(vw_row)

    # Convert to Polars tables
    ew_table = pl.DataFrame(ew_results).sort("lag")
    vw_table = pl.DataFrame(vw_results).sort("lag")

    print("EW Table")
    print(ew_table)

    print("\nVW Table")
    print(vw_table)

    return ew, vw, ew_table, vw_table

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

def ff4(ew, vw):
    from tabulate import tabulate

    fac = pd.read_csv('14-factors.csv')
    fac = fac.set_index('caldt')
    ew.index = pd.to_datetime(ew.index).to_period('M').to_timestamp('M')  # or 'MS' for month-start
    fac.index = pd.to_datetime(fac.index).to_period('M').to_timestamp('M')  
    vw.index = pd.to_datetime(vw.index).to_period('M').to_timestamp('M') 

    ew = ew.join(fac,how='left')

    print('EW Regression')
    ew['spread_no_rf'] = ew['spread'] - ew['rf']
    reg_ew = [smf.ols('spread_no_rf' + ' ~ 1 + exmkt + smb + hml + umd',data=ew).fit()]
    r_ew = Regtable(reg_ew,sig='coeff').render()
    print(tabulate(r_ew,tablefmt='github',headers=r_ew.columns))

    vw = vw.join(fac,how='left')

    print('\nVW Regression')
    vw['spread_no_rf'] = vw['spread'] - vw['rf']
    reg_vw = [smf.ols('spread_no_rf' + ' ~ 1 + exmkt + smb + hml + umd',data=vw).fit()]
    r_vw = Regtable(reg_vw,sig='coeff').render()
    print(tabulate(r_vw,tablefmt='github',headers=r_vw.columns))

def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)

    mstk = pl.scan_ipc('crsp_monthly.ftr', memory_map=False)
    dstk = our_extension()
    in_sample_start = [1965, 1, 1]
    in_sample_end = [2002, 12, 31]
    out_sample_start = [2003, 1, 1]
    out_sample_end = [2023, 12, 31]
    lags_table_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
        12, 24, 36, 48, 60, 72, 84, 96, 108,
        120, 132, 144, 156, 168, 180, 192, 204,
        216, 228, 240]
    lags_table_2 = [12 * y for y in range(1, 21)]

    # ------------ IN-SAMPLE ----------------

    # CREATE TABLE 1 with return lags and regression coefficient to current returns
    table_1 = recreate_table_1(mstk.collect(), lags=lags_table_1)
    print('TABLE 1')
    print(table_1)
    plot_table_1_results(table_1, lags_table_1)

    print('IN SAMPLE\n')
    # CREATE for each lag an EW and VW (p0 through p4 and spread)
    print('IN SAMPLE REPLICATION')
    ew_m_in, vw_m_in, ew_m_in_table, vw_m_in_table = port_creation_and_statistics(mstk, lags_table_2, in_sample_start, in_sample_end)
    print('IN SAMPLE EXTENSION')
    ew_d_in, vw_d_in, ew_d_in_table, vw_d_in_table = port_creation_and_statistics(dstk, lags_table_2, in_sample_start, in_sample_end)

    # CREATE colorful charts to see montonicity pattern
    print('REPLICATION')
    plot_portfolio_returns(ew_m_in_table, vw_m_in_table, 'Replication_Monotone')

    print('EXTENSION')
    plot_portfolio_returns(ew_d_in_table, vw_d_in_table, 'Extension_Monotone')

    # Run the FF4 Regression and print table
    print('FF4 Replication')
    ff4(ew_m_in, vw_m_in)
    print('FF4 Extension')
    ff4(ew_d_in, vw_d_in)

    # -------- OUT-OF-SAMPLE ------------
    print('OUT OF SAMPLE\n')
    # CREATE for each lag an EW and VW (p0 through p4 and spread)
    print('OUT OF SAMPLE REPLICATION')
    ew_m_out, vw_m_out, ew_m_out_table, vw_m_out_table = port_creation_and_statistics(mstk, lags_table_2, out_sample_start, out_sample_end)
    print('OUT OF SAMPLE EXTENSION')
    ew_d_out, vw_d_out, ew_d_out_table, vw_d_out_table = port_creation_and_statistics(dstk, lags_table_2, out_sample_start, out_sample_end)

    # Run the FF4 Regression and print table
    print('\nFF4 Replication')
    ff4(ew_m_out, vw_m_out)
    print('\nFF4 Extension')
    ff4(ew_d_out, vw_d_out)


if __name__ == "__main__":
    main()

# What specific results from table are we recreating with the portfolios?
# Is it annual in Table 2?
# Should we not drop the price for Table 1 to get a closer replication?
