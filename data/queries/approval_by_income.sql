SELECT 
    CASE 
        WHEN TRY_CAST(income AS DOUBLE) < 50 THEN 'Under 50K'
        WHEN TRY_CAST(income AS DOUBLE) < 100 THEN '50K-100K'
        WHEN TRY_CAST(income AS DOUBLE) < 200 THEN '100K-200K'
        ELSE 'Over 200K'
    END AS income_segment,
    COUNT(*) AS total,
    SUM(CASE WHEN action_taken IN ('1','2') THEN 1 ELSE 0 END) AS approved,
    ROUND(100.0 * SUM(CASE WHEN action_taken IN ('1','2') THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate
FROM hmda
WHERE action_taken IN ('1','2','3','7')
AND TRY_CAST(income AS DOUBLE) IS NOT NULL
GROUP BY income_segment
ORDER BY approval_rate DESC;