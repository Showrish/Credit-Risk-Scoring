SELECT 
    debt_to_income_ratio AS dti_range,
    COUNT(*) AS total,
    SUM(CASE WHEN action_taken IN ('3','7') THEN 1 ELSE 0 END) AS denied,
    ROUND(100.0 * SUM(CASE WHEN action_taken IN ('3','7') THEN 1 ELSE 0 END) / COUNT(*), 2) AS denial_rate
FROM hmda
WHERE action_taken IN ('1','2','3','7')
AND debt_to_income_ratio NOT IN ('Exempt', 'NA', '')
GROUP BY debt_to_income_ratio
ORDER BY denial_rate DESC;