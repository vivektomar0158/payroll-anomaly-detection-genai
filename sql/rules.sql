-- Rule-based anomaly flags using window functions.

DROP VIEW IF EXISTS v_salary_jumps;
CREATE VIEW v_salary_jumps AS
WITH ordered AS (
  SELECT employee_id, pay_period, gross_pay,
         LAG(gross_pay) OVER (PARTITION BY employee_id ORDER BY pay_period) AS prev_gross
  FROM pay_periods
)
SELECT employee_id, pay_period, gross_pay, prev_gross,
       ROUND((gross_pay - prev_gross) / prev_gross * 100, 1) AS pct_change
FROM ordered
WHERE prev_gross IS NOT NULL AND (gross_pay - prev_gross) / prev_gross > 0.25;

DROP VIEW IF EXISTS v_duplicate_payments;
CREATE VIEW v_duplicate_payments AS
SELECT employee_id, pay_period, COUNT(*) AS payment_count, SUM(net_pay) AS total_paid
FROM pay_periods
GROUP BY employee_id, pay_period
HAVING COUNT(*) > 1;

DROP VIEW IF EXISTS v_overtime_outliers;
CREATE VIEW v_overtime_outliers AS
WITH stats AS (
  SELECT employee_id, pay_period, overtime_hours,
         AVG(overtime_hours) OVER (PARTITION BY employee_id) AS emp_avg,
         (SELECT AVG(overtime_hours) FROM pay_periods) AS global_avg,
         (SELECT
            SQRT(AVG((overtime_hours - (SELECT AVG(overtime_hours) FROM pay_periods)) *
                     (overtime_hours - (SELECT AVG(overtime_hours) FROM pay_periods))))
          FROM pay_periods) AS global_stddev
  FROM pay_periods
)
SELECT employee_id, pay_period, overtime_hours,
       ROUND((overtime_hours - global_avg) / global_stddev, 2) AS z_score
FROM stats
WHERE (overtime_hours - global_avg) / global_stddev > 3;
