SELECT 
    CASE 
        WHEN TRY_CAST(loan_amount AS DOUBLE) < 150000 THEN 'Under 150K'
        WHEN TRY_CAST(loan_amount AS DOUBLE) < 300000 THEN '150K-300K'
        WHEN TRY_CAST(loan_amount AS DOUBLE) < 500000 THEN '300K-500K'
        WHEN TRY_CAST(loan_amount AS DOUBLE) < 1000000 THEN '500K-1M'
        ELSE 'Over 1M'
    END AS loan_bucket,
    COUNT(*) AS total,
    ROUND(100.0 * SUM(CASE WHEN action_taken IN ('1','2') THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate
FROM hmda
WHERE action_taken IN ('1','2','3','7')
GROUP BY loan_bucket
ORDER BY approval_rate DESC;