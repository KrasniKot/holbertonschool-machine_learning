-- Retrieve the rating for each show
SELECT ts.title, SUM(tsr.rate) rating FROM tv_shows ts, tv_show_ratings tsr WHERE ts.id = tsr.show_id GROUP BY ts.title ORDER BY rating DESC;;