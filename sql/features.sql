-- FleetCast feature engineering — DuckDB window functions (leakage-safe).
-- AUTO-RENDERED from src/fleetcast/features/sql.py for the DEFAULT config.
-- Lags are backward-only; rolling windows end at '1 PRECEDING' (current row
-- EXCLUDED). Regenerate with: uv run python -m fleetcast.features.sql

WITH top_zones AS (   -- N highest-volume pickup zones over the window
    SELECT PULocationID AS zone_id
    FROM read_parquet('data/raw/yellow_tripdata_*.parquet')
    WHERE PULocationID BETWEEN 1 AND 263
      AND tpep_pickup_datetime >= TIMESTAMP '2025-01-01 00:00:00'
      AND tpep_pickup_datetime <  TIMESTAMP '2025-04-01 00:00:00'
    GROUP BY zone_id
    ORDER BY count(*) DESC, zone_id
    LIMIT 20
),
hour_spine AS (   -- every hour in [start, end_excl): half-open, gap-free
    SELECT hour
    FROM range(TIMESTAMP '2025-01-01 00:00:00', TIMESTAMP '2025-04-01 00:00:00', INTERVAL 1 HOUR) t(hour)
),
grid AS (   -- dense zone x hour lattice
    SELECT z.zone_id, s.hour
    FROM top_zones z CROSS JOIN hour_spine s
),
hourly AS (   -- observed pickup counts per zone-hour
    SELECT
        PULocationID               AS zone_id,
        date_trunc('hour', tpep_pickup_datetime) AS hour,
        count(*)                   AS demand
    FROM read_parquet('data/raw/yellow_tripdata_*.parquet')
    WHERE PULocationID IN (SELECT zone_id FROM top_zones)
      AND tpep_pickup_datetime >= TIMESTAMP '2025-01-01 00:00:00'
      AND tpep_pickup_datetime <  TIMESTAMP '2025-04-01 00:00:00'
    GROUP BY zone_id, hour
),
panel AS (   -- DENSE, zero-filled target: the prerequisite for gap-free lags
    SELECT g.zone_id, g.hour, COALESCE(h.demand, 0) AS demand
    FROM grid g
    LEFT JOIN hourly h USING (zone_id, hour)
),
featured AS (
    SELECT
        zone_id,
        hour,
        demand,
        -- calendar (row-local; uses only the row's own timestamp)
        hour(hour)          AS hour_of_day,
        isodow(hour)        AS day_of_week,
        (isodow(hour) >= 6) AS is_weekend,
        month(hour)         AS month,
        -- lags: dense gap-free panel => LAG(n rows) == lag(n hours), backward-only
        LAG(demand, 1) OVER w AS lag_1h,
        LAG(demand, 24) OVER w AS lag_24h,
        LAG(demand, 168) OVER w AS lag_168h,
        -- rolling stats over the PRIOR window only; '... AND 1 PRECEDING' EXCLUDES
        -- the current row, so demand[t] can never enter its own features (no leakage)
        AVG(demand)         OVER w24 AS roll_mean_24h,
        STDDEV_SAMP(demand) OVER w24 AS roll_std_24h,
        AVG(demand)         OVER w168 AS roll_mean_168h,
        STDDEV_SAMP(demand) OVER w168 AS roll_std_168h
    FROM panel
    WINDOW
        w AS (PARTITION BY zone_id ORDER BY hour),
        w24 AS (PARTITION BY zone_id ORDER BY hour ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING),
        w168 AS (PARTITION BY zone_id ORDER BY hour ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING)
),
weather AS (   -- daily Central Park weather, broadcast across each day's 24 hours
    SELECT
        "DATE"::DATE              AS d,
        TRY_CAST("PRCP" AS DOUBLE) AS prcp_mm,
        TRY_CAST("SNOW" AS DOUBLE) AS snow_mm,
        TRY_CAST("TMAX" AS DOUBLE) AS tmax_c,
        TRY_CAST("TMIN" AS DOUBLE) AS tmin_c
    FROM read_csv_auto('data/raw/weather_daily.csv', header=true)
)
SELECT
    f.*,
    (f.hour::DATE IN (DATE '2025-01-01', DATE '2025-01-20', DATE '2025-02-17')) AS is_holiday,
    w.prcp_mm AS prcp_mm_d1, w.snow_mm AS snow_mm_d1,
    w.tmax_c AS tmax_c_d1, w.tmin_c AS tmin_c_d1
FROM featured f
LEFT JOIN weather w ON f.hour::DATE = w.d + INTERVAL 1 DAY
ORDER BY f.zone_id, f.hour
