SELECT 
    derived_race,
    COUNT(*) AS total,
    SUM(CASE WHEN action_taken IN ('3','7') THEN 1 ELSE 0 END) AS denied,
    ROUND(100.0 * SUM(CASE WHEN action_taken IN ('3','7') THEN 1 ELSE 0 END) / COUNT(*), 2) AS denial_rate
FROM hmda
WHERE action_taken IN ('1','2','3','7')
AND derived_race NOT IN ('Race Not Available')
GROUP BY derived_race
ORDER BY denial_rate DESC;