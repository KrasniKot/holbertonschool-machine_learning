-- Create a table users
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,                               -- id is an auto-incrementing integer and primary key
    email VARCHAR(255) NOT NULL UNIQUE,                  -- email is a string, cannot be null, and must be unique
    name VARCHAR(255),                                   -- name is a string, optional
    country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US' -- country is restricted to 'US', 'CO', 'TN' with a default of 'US'
);
