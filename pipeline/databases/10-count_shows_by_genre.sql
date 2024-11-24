-- Count shows by genre
SELECT tg.name, COUNT(tsg.genre_id) AS number_of_shows FROM tv_genres tg, tv_show_genres tsg WHERE tg.id = tsg.genre_id GROUP BY tg.name ORDER BY number_of_shows DESC;
