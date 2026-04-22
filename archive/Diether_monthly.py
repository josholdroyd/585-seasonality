def main():
    stk = (
        pl.scan_ipc('../data/mstk_polars.ftr',memory_map=False)
        .with_columns(
            me = (pl.when((pl.col('prc') * pl.col('shr')) > 1e-6)
                  .then((pl.col('prc') * pl.col('shr')) / 1000.0)
                  .otherwise(None)),
        ).with_columns(
            pl.col('ret').shift(12).over('permno').alias('retlag'),
            pl.col('prc','me').shift().over('permno').name.suffix('lag')
        ).filter(
            pl.col('shrcd').is_between(10,11) &
            pl.col('melag').is_not_null() &
            pl.col('retlag').is_not_null() &
            (pl.col('prclag') > 4.999) 
        ).with_columns(
            pl.col('retlag')
            .qcut(5,labels=[f"p{x}" for x in range(5)]).over('caldt')
            .alias('port'),
            (pl.col('melag')*pl.col('ret')).alias('meret')
        ).filter(
            pl.col('ret').is_not_null()
        )
    )

    port = (
        stk.group_by(
            ['caldt','port'])
        .agg(
            (pl.col('ret')*100).mean().alias('ewret'),
            (pl.col('meret')*100).sum().alias('vwret'),
            pl.col('melag').sum().alias('wtotal'),
        )
        .with_columns(
            vwret = pl.col('vwret')/pl.col('wtotal')
        ).sort(['caldt','port'])
    ).collect()

    ew = (port.pivot(index='caldt',on='port',values='ewret')
          .to_pandas().set_index('caldt'))
    vw = (port.pivot(index='caldt',on='port',values='vwret')
          .to_pandas().set_index('caldt'))
