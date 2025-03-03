import json


class MenuItem:
    def __init__(self, item_name, quantity, size, price, special_instructions=""):
        self.item_name = item_name
        self.quantity = quantity
        self.size = size  # e.g., Small, Medium, Large
        self.price = price
        self.special_instructions = special_instructions

    def calculate_total(self):
        """Calculates the total price for this menu item."""
        return self.price * self.quantity

    def __str__(self):
        """Returns a formatted string representation of the menu item."""
        return (f"{self.quantity} x {self.size} {self.item_name} @ ${self.price:.2f} each"
                + (f" | Special instructions: {self.special_instructions}" if self.special_instructions else ""))


class RestaurantOrder:
    def __init__(self):
        self.order_items = []
        self.delivery = False
        self.delivery_address = None
        self.price = 0
        self.custom_instructions = None
        self.json_order = None

    def add_item(self, item_name, quantity, size, price, special_instructions=""):
        """Adds a menu item to the order."""
        item = MenuItem(item_name, quantity, size, price, special_instructions)
        self.order_items.append(item)
        print(f"Added {quantity} x {size} {item_name} to the order.")

    # Function to extract JSON with nested curly brackets
    def extract_json(self, text):
        bracket_count = 0
        json_start = None
        json_end = None

        for i, char in enumerate(text):
            if char == '{':
                if bracket_count == 0:
                    json_start = i
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    json_end = i
                    break

        if json_start is not None and json_end is not None:
            return text[json_start:json_end + 1]
        else:
            return None

    def add_items_from_json(self, json_message):
        response_str = str(json_message).replace('\n', '')

        # check if the reponse is proper json
        # if yes process
        # else return None

        json_string = self.extract_json(response_str)

        if json_string:
            # Parse the JSON message
            try:
                order_data = json.loads(json_string)
                print("Extracted JSON data:")
                print(json.dumps(order_data, indent=2))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            # print("No JSON data found in the text.")
            raise Exception

        # Better option is to receive a json when customer order finalized....
        # Initialize RestaurantOrder object here and send to POS/email. etc.

        # Extract and process the menu items ordered
        items_ordered = order_data["menu_items_ordered"]
        total_price = 0

        print("menu_items_ordered")
        for item in items_ordered:
            item_name = item["item"]
            try:
                item_size = item["size"]
            except:
                pass
            item_quantity = item["quantity"]
            custom_instructions = None
            item_price = float(item["price"].replace('$', ''))  # Convert price to float
            total_price += item_price * item_quantity
            print(f"{item_quantity} {item_name} ({item_size}) - ${item_price:.2f}")
            self.add_item(item_name, item_quantity, item_size, item_price, custom_instructions)

        # Extract total price from the JSON and compare with calculated total price
        json_total_price = float(order_data["total_price"].replace('$', ''))
        self.price = json_total_price
        self.json_order = json_string
        self.delivery = True if order_data["pickup_or_delivery"] != "pickup" else False

    def remove_item(self, item_name):
        """Removes a menu item from the order by name."""
        for item in self.order_items:
            if item.item_name == item_name:
                self.order_items.remove(item)
                print(f"Removed {item_name} from the order.")
                break
        else:
            print(f"{item_name} is not in the order.")

    def calculate_total(self):
        """Calculates the total price of the entire order."""
        return sum(item.calculate_total() for item in self.order_items)

    def show_order(self):
        """Displays the current items in the order."""
        if self.order_items:
            print("Your order contains:")
            for item in self.order_items:
                print(item)
        else:
            print("Your order is empty.")

    def order_summary(self):
        """Returns a formatted string summarizing the order."""
        if not self.order_items:
            return "Your order is empty."

        summary_lines = ["Your order summary:"]
        for item in self.order_items:
            summary_lines.append(str(item))  # Each item string will be added

        total = self.calculate_total()
        summary_lines.append(f"\nTotal: ${total:.2f}")

        # Join the list of lines into a single formatted string
        return "\n".join(summary_lines)


    def apply_discount(self, discount_percentage):
        """Applies a discount to the total order."""
        total = self.calculate_total()
        discount = total * (discount_percentage / 100)
        return total - discount

# Example usage:
# order = RestaurantOrder()
# order.add_item('Burger', 2, 'Large', 8.50, 'No onions')
# order.add_item('Fries', 1, 'Medium', 3.00)
# order.add_item('Soda', 2, 'Small', 1.50, 'Extra ice')
# order.show_order()
#
# total = order.calculate_total()
# print(f"Total: ${total:.2f}")
#
# discounted_total = order.apply_discount(10)
# print(f"Total after 10% discount: ${discounted_total:.2f}")