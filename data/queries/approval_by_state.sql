SELECT 
    state_code,
    COUNT(*) AS total,
    ROUND(100.0 * SUM(CASE WHEN action_taken IN ('1','2') THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate
FROM hmda
WHERE action_taken IN ('1','2','3','7')
GROUP BY state_code
ORDER BY approval_rate DESC;