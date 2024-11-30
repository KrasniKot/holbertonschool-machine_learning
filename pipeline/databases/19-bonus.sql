-- Add a new correction for a student
DELIMITER #

CREATE PROCEDURE AddBonus(IN user_id INT, IN project_name VARCHAR(100), IN score INT)
BEGIN
    DECLARE pid INT;

    -- 1. Check for the existence of the project
    SELECT id INTO pid FROM projects WHERE name = project_name;

    -- 2. If not found, create it
    IF pid IS NULL THEN
        INSERT INTO projects (name) VALUES (project_name);
        SET project_id = LAST_INSERT_ID();
    END IF;

    -- 3. Add the correction
    INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, project_id, score);

END #;

DELIMITER;
