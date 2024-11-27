-- Retrieve origin and number of fans for metal bands
SELECT origin, SUM(fans) nb_fans FROM metal_bands GROUP BY origin ORDER BY nb_fans DESC;
