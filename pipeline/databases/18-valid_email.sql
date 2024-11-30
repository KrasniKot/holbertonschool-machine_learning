--  Reset valid_email to 0 if the new email is different from the old one
DELIMITER #

CREATE TRIGGER UpdateMail
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NEW.email != OLD.email THEN
        SET NEW.valid_email = 0;
    END IF;
END #

DELIMITER;
