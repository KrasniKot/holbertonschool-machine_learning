-- Create a trigger that decreases the quantity of an item after adding a new order
CREATE TRIGGER decrease_item_quantity AFTER INSERT ON orders
FOR EACH ROW
    -- Decrease the quantity of the item based on the new order
    UPDATE items
    SET items.quantity = items.quantity - NEW.number
    WHERE item.name = NEW.item_name;
