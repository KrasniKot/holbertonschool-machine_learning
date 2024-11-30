-- Calculate and list Glam rock bands by longevity until 2020
SELECT 
    band_name,
    CASE
        WHEN split IS NOT NULL THEN split - formed
        ELSE 2020 - formed
    END AS lifespan
FROM 
    metal_bands
WHERE 
    style LIKE '%Glam rock%'
ORDER BY 
    lifespan DESC;
