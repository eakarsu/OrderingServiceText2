from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import re
import json

class MenuParser:
    def __init__(self):
        self.categories = []
        self.rules = []
        self.current_category = None
        self.current_rule = None
        self.patterns = {
            'category': re.compile(r'^\[Begin Category\]\s*(.*)'),
            'category_end': re.compile(r'^\[End Category\]'),
            'rule_begin': re.compile(r'^\[Begin Rule\]\s*(.*)'),
            'rule_end': re.compile(r'^\[End Rule\]'),
            'base_price': re.compile(r'Base Price:\s*\$([\d.]+)'),
            'select_rules': re.compile(r'select rules\s*(.+)'),
            'category_rule': re.compile(r'^-\s*Select rules\s+(.+?)\s+applies to all'),
            #food regex
            'rule_option': re.compile(r'^\s*(.+?)\s*\(Rule:\s*(.+?)\)(:)?'),
            #rule item with description
            'rule_item' : re.compile(r'^(\s*)-\s*(.+?)\s*-\s*\$([\d.]+)\s*-?\s*(.*)'),
            #'rule_item': re.compile(r'^(\s*)-\s*(.+?)\s*-\s*\$([\d.]+)'),
            'standard_item': re.compile(r'^-\s*(.*?):\s*\$(\d+\.\d{2})')
        }

    def parse_menu_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        item_rules = {}  # New dictionary to store item-specific rules
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Category detection
            cat_match = self.patterns['category'].match(line)
            if cat_match:
                self.current_category = {
                    "name": cat_match.group(1).strip(),
                    "items": []
                }
                self.categories.append(self.current_category)
                i += 1
                continue

            # Category end detection
            if self.patterns['category_end'].match(line):
                self.current_category = None
                i += 1
                continue

            if self.current_category:
                # Check for category-wide rule directive
                category_rule_match = self.patterns['category_rule'].match(line)
                if category_rule_match:
                    rule_names = [rule.strip() for rule in category_rule_match.group(1).split(',')]
                    for item in self.current_category['items']:
                        if item['name'] not in item_rules:
                            item_rules[item['name']] = []
                        item_rules[item['name']].extend(rule_names)
                    i += 1
                    continue

                # Standard item detection
                item_match = self.patterns['standard_item'].match(line)
                if item_match:
                    item_name = item_match.group(1).strip()
                    price = float(item_match.group(2))
                    description = ""
                    rules = []
                    
                    # Check for rules in the item line
                    select_rules_match = self.patterns['select_rules'].search(line)
                    if select_rules_match:
                        rules = [rule.strip() for rule in select_rules_match.group(1).split(',')]
                    
                    # Get description from next line if it exists
                    if i + 1 < len(lines) and lines[i+1].startswith(" "):
                        description = lines[i+1].strip()
                        i += 2
                    else:
                        i += 1

                    self.current_category['items'].append({
                        'name': item_name,
                        'price': price,
                        'description': description,
                        'selected_rules': rules
                    })

        # After parsing, add the rules to the items
        for category in self.categories:
            for item in category['items']:
                if item['name'] in item_rules:
                    item['selected_rules'] = item_rules[item['name']]



    def parse_menu_file2(self, file_path):
        """Parse the menu file to extract categories and items."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                # Category detection
                cat_match = self.patterns['category'].match(line)
                if cat_match:
                    self.current_category = {
                        "name": cat_match.group(1).strip(),
                        "items": [],
                        "selected_rules": []
                    }
                    self.categories.append(self.current_category)
                    i += 1
                    continue
                # Category end detection
                if self.patterns['category_end'].match(line):
                    self.current_category = None
                    i += 1
                    continue
                if self.current_category:
                    # Check for "select rules" directive and store them
                    """ select_rules_match = self.patterns['select_rules'].search(line)
                    if select_rules_match:
                        rules_list = select_rules_match.group(1).split(',')
                        self.current_category['selected_rules'] = [rule.strip() for rule in rules_list]
                    """  # Base price detection

                    # Insert the new code here
                    # Check for category-wide rule directive
                    category_rule_match = self.patterns['category_rule'].match(line)
                    if category_rule_match:
                        rule_names = [rule.strip() for rule in category_rule_match.group(1).split(',')]
                        if 'selected_rules' not in self.current_category:
                            self.current_category['selected_rules'] = []
                        for rule_name in rule_names:
                            if rule_name not in self.current_category['selected_rules']:
                                self.current_category['selected_rules'].append(rule_name)
                                print(f"[DEBUG] Added category-wide rule '{rule_name}' to {self.current_category['name']}")
                        i += 1
                        continue

                    # Change this section in parse_menu_file:
                    select_rules_match = self.patterns['select_rules'].search(line)
                    if select_rules_match:
                        rules_list = select_rules_match.group(1).split(',')
                        # Add this check to append rather than replace
                        if 'selected_rules' not in self.current_category:
                            self.current_category['selected_rules'] = []
                        # Append new rules instead of replacing
                        for rule in rules_list:
                            if rule.strip() not in self.current_category['selected_rules']:
                                self.current_category['selected_rules'].append(rule.strip())
                                        
                    base_price_match = self.patterns['base_price'].search(line)
                    if base_price_match:
                        self.current_category['base_price'] = float(base_price_match.group(1))
                    # Standard item detection
                    item_match = self.patterns['standard_item'].match(line)
                    if item_match:
                        item_name = item_match.group(1).strip()
                        price = float(item_match.group(2))
                        description = ""
                        # If the next line is indented, it is treated as a description/ingredients line.
                        if i + 1 < len(lines) and lines[i+1].startswith("  "):
                            description = lines[i+1].strip()
                            i += 2
                        else:
                            i += 1
                        self.current_category['items'].append({
                            'name': item_name,
                            'price': price,
                            'description': description
                        })
                        continue
                i += 1



    def parse_rules_file(self, file_path):
        """Parse the rules file to extract rule definitions, options, and rule items."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                # Rule begin detection
                rule_begin_match = self.patterns['rule_begin'].match(line)
                if rule_begin_match:
                    rule_name = rule_begin_match.group(1).strip()
                    self.current_rule = {
                        "name": rule_name,
                        "options": []
                    }
                    self.rules.append(self.current_rule)
                    i += 1
                    continue
                # Rule end detection
                if self.patterns['rule_end'].match(line):
                    self.current_rule = None
                    i += 1
                    continue
                # Option detection within a rule
                rule_option_match = self.patterns['rule_option'].match(line)
                if rule_option_match and self.current_rule:
                    option_name = rule_option_match.group(1).strip()
                    constraint_text = rule_option_match.group(2).strip()
                    constraints = self._parse_constraints(constraint_text)
                    current_option = {
                        "name": option_name,
                        "constraints": constraints,
                        "items": []
                    }
                    self.current_rule["options"].append(current_option)
                    j = i + 1
                    while j < len(lines):
                        item_line = lines[j].strip()
                        if not item_line or self.patterns['rule_option'].match(item_line) or self.patterns['rule_end'].match(item_line):
                            break
                        rule_item_match = self.patterns['rule_item'].match(item_line)
                        if rule_item_match:
                            item_name = rule_item_match.group(2).strip()
                            price = float(rule_item_match.group(3))
                            description = rule_item_match.group(4)
                            if not description:
                                description = ""
                            current_option["items"].append({
                                "name": item_name,
                                "price": price,
                                "description": description
                            })
                        j += 1
                    i = j
                    continue
                i += 1

    def _parse_constraints(self, text):
        """Parse rule constraints from text (e.g., 'Select 1', 'select up to 3 items', or 'select 1 to 6')."""
        constraints = {'min': 0, 'max': None}
        text = text.lower()
        if 'select 1' in text and 'to' not in text:
            constraints.update({'min': 1, 'max': 1})
        elif 'up to' in text:
            match = re.search(r'up to (\d+)', text)
            if match:
                constraints['max'] = int(match.group(1))
        elif 'select' in text:
            match = re.search(r'select (\d+) to (\d+)', text)
            if match:
                constraints['min'] = int(match.group(1))
                constraints['max'] = int(match.group(2))
            else:
                match = re.search(r'select up to (\d+)', text)
                if match:
                    constraints['max'] = int(match.group(1))
        return constraints

class MenuIndexer:
    def __init__(self):
        self.client = PersistentClient(path="chroma_database")
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
        self._initialize_collections()

    def _initialize_collections(self):
        self.categories_col = self._get_or_create_collection("categories")
        self.items_col = self._get_or_create_collection("items")
        self.rules_col = self._get_or_create_collection("rules")
        #self.rule_options_col = self._get_or_create_collection("rule_service")
        self.rule_options_col = self._get_or_create_collection("rule_options")
        self.rule_items_col = self._get_or_create_collection("rule_items")

    def _get_or_create_collection(self, name):
        try:
            return self.client.get_collection(name, self.embedder)
        except Exception as e:
            return self.client.create_collection(name, embedding_function=self.embedder)


    def validate_option_metadata(self,metadata):
        """Ensure all metadata values are valid types for ChromaDB."""
        for key, value in metadata.items():
            if value is None:
                metadata[key] = -1 if key == "max" else ""  # Convert None to appropriate default
            elif not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)  # Convert to string if not a valid type
                
    def index_menu_and_rules(self, menu_parser):
        # Remove old collections if they exist
        try:
            self.client.delete_collection("categories")
            self.client.delete_collection("items")
            self.client.delete_collection("rules")
            self.client.delete_collection("rule_options")
            self.client.delete_collection("rule_items")
            print("[DEBUG] Deleted old collections successfully")
        except Exception as e:
            print(f"[DEBUG] Error deleting collections: {str(e)}")
        
        self._initialize_collections()
        print("[DEBUG] Initialized new collections")
        
        # Debug: Print all categories and their items
        print("\n[DEBUG] INDEXING CATEGORIES AND ITEMS:")
        for category in menu_parser.categories:
            print(f"[DEBUG] Category: {category['name']}")
            if 'base_price' in category:
                print(f"[DEBUG] Base Price: ${category['base_price']}")
            if 'selected_rules' in category:
                print(f"[DEBUG] Selected Rules: {category['selected_rules']}")
            print(f"[DEBUG] Items:")
            for item in category.get('items', []):
                print(f"[DEBUG] - {item['name']}: ${item['price']} (Description: {item.get('description', 'None')})")
        
        # First, index all rules and their options/items
        print("\n[DEBUG] INDEXING RULES AND OPTIONS:")
        for rule in menu_parser.rules:
            print(f"[DEBUG] Processing rule: {rule['name']}")
            rule_id = f"rule_{rule['name']}"
            self.rules_col.add(
                documents=[rule['name']],
                metadatas=[{"name": rule['name']}],
                ids=[rule_id]
            )
            print(f"[DEBUG] Added rule {rule['name']} to rules collection")
            
            # Index options for this rule
            print(f"[DEBUG] Processing {len(rule.get('options', []))} options for rule {rule['name']}")
            for option in rule.get('options', []):
                option_id = f"option_{rule['name']}_{option['name']}"
                option_metadata = {
                    "rule": rule['name'],
                    "name": option['name'],
                    "min": 1,
                    #"max": option['constraints'].get('max', None)
                    "max": option['constraints'].get('max', -1)  # Use -1 to represent 'unlimited' if 'max' is None
                }
                 
                self.validate_option_metadata(option_metadata)
                self.rule_options_col.add(
                    documents=[option['name']],
                    metadatas=[option_metadata],
                    ids=[option_id]
                )
                print(f"[DEBUG] Added option {option['name']} to rule_options collection with constraints: min={option_metadata['min']}, max={option_metadata['max']}")
                
                # Index items for this option
                print(f"[DEBUG] Processing {len(option.get('items', []))} items for option {option['name']}")
                for item in option.get('items', []):
                    item_id = f"rule_item_{rule['name']}_{option['name']}_{item['name']}"
                    item_metadata = {
                        "rule": rule['name'],
                        "option": option['name'],
                        "price": item['price'],
                        "item": item['name'],
                        "description": item["description"]
                    }
                    self.rule_items_col.add(
                        documents=[item['name']],
                        metadatas=[item_metadata],
                        ids=[item_id]
                    )
                    print(f"[DEBUG] Added rule item {item['name']} with price ${item['price']} to rule_items collection")
        
        # Build a mapping of item names to their specific rules from the menu file
        item_to_rules_map = {}
        print("[DEBUG] Building item-to-rules mapping from menu file")
        """ for category in menu_parser.categories:
            for item in category.get('items', []):
                item_name = item['name'].split(',')[0].strip()
                # Extract rules from item description if available
                if 'select rules' in item.get('description', '').lower():
                    rules_text = item.get('description', '').split('select rules')[1].strip()
                    rules = [r.strip() for r in rules_text.split(',')]
                    item_to_rules_map[item_name] = rules
                    print(f"[DEBUG] Found rules for {item_name} in description: {rules}") """
        
        for category in menu_parser.categories:
            for item in category.get('items', []):
                item_name = item['name'].split(',')[0].strip()
                # Extract rules from item name if available
                if 'selected_rules' in item:
                    item_to_rules_map[item_name] = item['selected_rules']
                    print(f"[DEBUG] Found pre-defined rules for {item_name}: {item['selected_rules']}")
                            
        # Debug: Print the item-to-rules mapping
        print("[DEBUG] Item-to-rules mapping:")
        for item_name, rules in item_to_rules_map.items():
            print(f"[DEBUG] {item_name}: {rules}")
        
        # Now index categories and items
        print(f"[DEBUG] Processing {len(menu_parser.categories)} categories")
        for category in menu_parser.categories:
            print(f"[DEBUG] Processing category: {category['name']}")
            category_id = f"cat_{category['name']}"
            category_metadata = {
                'name': category['name']
            }
            
            if 'base_price' in category:
                category_metadata['base_price'] = category['base_price']
                print(f"[DEBUG] Category {category['name']} has base_price: ${category['base_price']}")
            
            if 'selected_rules' in category and category['selected_rules']:
                category_metadata['selected_rules'] = json.dumps(category['selected_rules'])
                print(f"[DEBUG] Category {category['name']} has rules: {category['selected_rules']}")
            
            self.categories_col.add(
                documents=[category['name']],
                metadatas=[category_metadata],
                ids=[category_id]
            )
            print(f"[DEBUG] Added category {category['name']} to categories collection")
            
            # Index items in this category
            print(f"[DEBUG] Processing {len(category.get('items', []))} items in category {category['name']}")
            for item in category.get('items', []):
                self._index_item(item, category, item_to_rules_map, menu_parser)
        
        print("[DEBUG] Completed index_menu_and_rules function")

    def _index_item(self, item, category, item_to_rules_map, menu_parser):
        print(f"[DEBUG] Processing item: {item['name']} in category {category['name']}")
        item_id = f"item_{category['name']}_{item['name']}"
        document_text = f"{item['name']} "
        metadata = {
            'name': item['name'],
            'category': category['name'],
            'price': item['price'],
            'base_price': item['price'],
            'ingredients': item.get('description', ''),
            "description": item.get('description', '')
        }
        if 'selected_rules' in item and item['selected_rules']:
            metadata['selected_rules'] = json.dumps(item['selected_rules'])
            print(f"[DEBUG] Using item-specific rules for {item['name']}: {item['selected_rules']}")
         
        print(f"[DEBUG] Item {item['name']} has price: ${item['price']}")
            
        # Extract item name without any suffix for rule matching
        item_name = item['name'].split(',')[0].strip()
        print(f"[DEBUG] Extracted item name for rule matching: {item_name}")

        # Check if this item has specific rules defined in the item-to-rules map
        if item_name in item_to_rules_map:
            item_rules = item_to_rules_map[item_name]
            metadata['selected_rules'] = json.dumps(item_rules)
            print(f"[DEBUG] Using mapped rules for {item_name}: {item_rules}")
        else:
            # Check if there's a rule with the same name as the item
            item_specific_rules = []
            for rule in menu_parser.rules:
                if item_name.lower() in rule['name'].lower():
                    rule_options = [option['name'] for option in rule.get('options', [])]
                    if rule_options:
                        item_specific_rules = rule_options
                        print(f"[DEBUG] Found matching rule '{rule['name']}' for item {item_name} with options: {rule_options}")
                        break
            
            # If item-specific rules found, use those
            if item_specific_rules:
                metadata['selected_rules'] = json.dumps(item_specific_rules)
                print(f"[DEBUG] Using item-specific rules for {item_name}: {item_specific_rules}")

        self.items_col.add(
            documents=[f"{document_text},description:{metadata['description']}"],
            metadatas=[metadata],
            ids=[item_id]
        )

        print(f"[DEBUG] Added item {item['name']} to items collection")



    def _index_item2(self, item, category, item_to_rules_map, menu_parser):
        print(f"[DEBUG] Processing item: {item['name']} in category {category['name']}")
        item_id = f"item_{category['name']}_{item['name']}"
        #document_text = f"{item['name']} {category['name']} {item.get('description', '')}"
        document_text = f"{item['name']} "

        metadata = {
            'name': item['name'],
            'category': category['name'],
            'price': item['price'],
            'base_price': item['price'],
            'ingredients': item.get('description', ''),
            "description": item.get('description', '')
        }
        
        print(f"[DEBUG] Item {item['name']} has price: ${item['price']}")
        print(f"[DEBUG] Using item's own price as base_price: ${item['price']}")
        
        # Extract item name without any suffix for rule matching
        item_name = item['name'].split(',')[0].strip()
        print(f"[DEBUG] Extracted item name for rule matching: {item_name}")
        
        # Check if this item has specific rules defined in the item-to-rules map
        if item_name in item_to_rules_map:
            item_rules = item_to_rules_map[item_name]
            metadata['selected_rules'] = json.dumps(item_rules)
            print(f"[DEBUG] Using mapped rules for {item_name}: {item_rules}")
        else:
            # Check if there's a rule with the same name as the item
            item_specific_rules = []
            for rule in menu_parser.rules:
                if item_name.lower() in rule['name'].lower():
                    # Found a rule specifically for this item
                    rule_options = [option['name'] for option in rule.get('options', [])]
                    if rule_options:
                        item_specific_rules = rule_options
                        print(f"[DEBUG] Found matching rule '{rule['name']}' for item {item_name} with options: {rule_options}")
                        break
                   
            # If item-specific rules found, use those
            if item_specific_rules:
                metadata['selected_rules'] = json.dumps(item_specific_rules)
                print(f"[DEBUG] Using item-specific rules for {item_name}: {item_specific_rules}")
            # Otherwise use category rules if available
            #elif 'selected_rules' in category and category['selected_rules']:
            #    metadata['selected_rules'] = json.dumps(category['selected_rules'])
            #    print(f"[DEBUG] Using category rules for {item_name}: {category['selected_rules']}")
            # Use this more generic approach:
            elif 'selected_rules' in category and category['selected_rules']:
                # Check if this item appears as an add-on in any rule_items
                is_addon = False
                for rule in menu_parser.rules:
                    for option in rule.get('options', []):
                        for addon in option.get('items', []):
                            if addon['name'] == item_name:
                                is_addon = True
                                print(f"[DEBUG] {item_name} is an add-on for {option['name']}, skipping category rules")
                                break
                        if is_addon:
                            break
                    if is_addon:
                        break
                
                # Only assign category rules to main services, not to add-ons
                if not is_addon:
                    metadata['selected_rules'] = json.dumps(category['selected_rules'])
                    print(f"[DEBUG] Using category rules for main service {item_name}: {category['selected_rules']}")

     
        self.items_col.add(
            documents=[document_text],
            metadatas=[metadata],
            ids=[item_id]
        )
        print(f"[DEBUG] Added item {item['name']} to items collection")

    

    def _index_item_old(self, item, category, item_to_rules_map, menu_parser):
        print(f"[DEBUG] Processing item: {item['name']} in category {category['name']}")
        item_id = f"item_{category['name']}_{item['name']}"
        document_text = f"{item['name']} {category['name']} {item.get('description', '')}"
        
        metadata = {
            'name': item['name'],
            'category': category['name'],
            'price': item['price'],
            'base_price': item['price'],
            'ingredients': item.get('description', '')
        }
        
        print(f"[DEBUG] Item {item['name']} has price: ${item['price']}")
        print(f"[DEBUG] Using item's own price as base_price: ${item['price']}")
        
        # Extract item name without any suffix for rule matching
        item_name = item['name'].split(',')[0].strip()
        print(f"[DEBUG] Extracted item name for rule matching: {item_name}")
        
        # Check if this item has specific rules defined in the item-to-rules map
        if item_name in item_to_rules_map:
            item_rules = item_to_rules_map[item_name]
            metadata['selected_rules'] = json.dumps(item_rules)
            print(f"[DEBUG] Using mapped rules for {item_name}: {item_rules}")
        else:
            # Check if there's a rule with the same name as the item
            item_specific_rules = []
            for rule in menu_parser.rules:
                if item_name.lower() in rule['name'].lower():
                    # Found a rule specifically for this item
                    rule_options = [option['name'] for option in rule.get('options', [])]
                    if rule_options:
                        item_specific_rules = rule_options
                        print(f"[DEBUG] Found matching rule '{rule['name']}' for item {item_name} with options: {rule_options}")
                        break
                        
            # Special case handling for known items
            if item_name == "Bagel":
                item_specific_rules = ['Bagel Options', 'Bagel Spreads']
                print(f"[DEBUG] Using special case rules for {item_name}: {item_specific_rules}")
                
            # If item-specific rules found, use those
            if item_specific_rules:
                metadata['selected_rules'] = json.dumps(item_specific_rules)
                print(f"[DEBUG] Using item-specific rules for {item_name}: {item_specific_rules}")
            # Otherwise use category rules if available
            elif 'selected_rules' in category and category['selected_rules']:
                metadata['selected_rules'] = json.dumps(category['selected_rules'])
                print(f"[DEBUG] Using category rules for {item_name}: {category['selected_rules']}")
                
        self.items_col.add(
            documents=[document_text],
            metadatas=[metadata],
            ids=[item_id]
        )
        print(f"[DEBUG] Added item {item['name']} to items collection")

def test_local_honey_menu():
    print("Testing Local Honey BK Menu Parser...\n")
    
    # Create a MenuParser instance and parse files
    menu_parser = MenuParser()
    """ print("Parsing menu file: local_honey_menu_formatted.txt")
    menu_parser.parse_menu_file("local_honey_menu_formatted.txt")
    print("Parsing rules file: local_honey_rules.txt")
    menu_parser.parse_rules_file("local_honey_rules.txt")
 """
    print("Parsing menu file: local_honey_menu_formatted.txt")
    menu_parser.parse_menu_file("salon/salon_prompt2.txt")
    print("Parsing rules file: local_honey_rules.txt")
    menu_parser.parse_rules_file("salon/salon_rules.txt")
    
    # Test 1: Verify Mani category has correct services
    print("\nTest 1: Verify Mani category has correct services")
    mani_category = None
    for category in menu_parser.categories:
        if category['name'] == "Mani":
            mani_category = category
            break
    
    if mani_category:
        print(f"Found Mani category with {len(mani_category.get('items', []))} services")
        
        # Check for specific services
        service_names = [item['name'] for item in mani_category.get('items', [])]
        assert "Classic Honey Mani" in service_names, "Missing Classic Honey Mani service"
        assert "Crystal Energy Mani" in service_names, "Missing Crystal Energy Mani service"
        assert "Gel Removal (hands)" in service_names, "Missing Gel Removal service"
        print("✓ All expected services found")
    else:
        print("❌ Mani category not found")
    
    # Test 2: Verify Classic Honey Mani rule parsing
    print("\nTest 2: Verify Classic Honey Mani rule parsing")
    # Test 2: Verify Classic Honey Mani rule parsing

    mani_rule = None
    for rule in menu_parser.rules:
        if rule['name'] == "Mani":
            mani_rule = rule
            break

    if mani_rule:
        print(f"Found Mani rule with {len(mani_rule.get('options', []))} options")
        
        # Check for Classic Honey Mani option
        classic_honey_option = None
        for option in mani_rule.get('options', []):
            if option['name'] == "Classic Honey Mani":
                classic_honey_option = option
                break
        
        if classic_honey_option:
            print(f"Found Classic Honey Mani option with {len(classic_honey_option.get('items', []))} items")
            
            # Check for specific add-ons
            addon_names = [item['name'] for item in classic_honey_option.get('items', [])]
            assert "Gel Removal (hands)" in addon_names, "Missing Gel Removal add-on"
            assert "Paraffin Wax Dip" in addon_names, "Missing Paraffin Wax Dip add-on"
            assert "Crystal Energy Hand Massage" in addon_names, "Missing Crystal Energy Hand Massage add-on"
            
            print("✓ Add-ons option correctly parsed")
        else:
            print("❌ Classic Honey Mani option not found")
    else:
        print("❌ Mani rule not found")

    # Test 3: Verify indexing
    print("\nTest 3: Verify indexing")
    indexer = MenuIndexer()
    indexer.index_menu_and_rules(menu_parser)
    
    # Check if Classic Honey Mani was indexed with add-ons
    results = indexer.items_col.query(
        query_texts=["Classic Honey Mani"],
        include=["metadatas"],
        n_results=1
    )
    
    if results and results["metadatas"] and results["metadatas"][0]:
        metadata = results["metadatas"][0][0]
        print(f"Classic Honey Mani metadata: {metadata}")
        assert 'selected_rules' in metadata, "Classic Honey Mani should have selected_rules in metadata"
        print("✓ Classic Honey Mani correctly indexed with rules")
    else:
        print("❌ Classic Honey Mani not found in indexed data")
    
    print("\nAll tests completed!")
    


def main():
    print("Testing MenuParser and MenuIndexer...\n")
    
    # Create a MenuParser instance and parse files
    menu_parser = MenuParser()
    print("Parsing menu file: misc2/prompt2.txt")
    menu_parser.parse_menu_file("misc2/prompt2.txt")
    print("Parsing rules file: misc2/rules.txt")
    menu_parser.parse_rules_file("misc2/rules.txt")
    
    # Test 1: Verify BYO Salad has correct rules
    print("\nTest 1: Verify BYO Salad has correct rules")
    salad_category = None
    for category in menu_parser.categories:
        if category['name'] == "Chopped Salad":
            salad_category = category
            break
    
    if salad_category:
        print(f"Found Chopped Salad category with base price: ${salad_category.get('base_price', 0)}")
        print(f"Selected rules: {salad_category["items"][0].get('selected_rules', [])}")
        assert "Salad Add-ons" in salad_category["items"][0].get('selected_rules', []), "Missing Salad Add-ons rule"
        assert "Salad Base" in salad_category["items"][0].get('selected_rules', []), "Missing Salad Base rule"
        assert "Salad Dressing" in salad_category["items"][0].get('selected_rules', []), "Missing Salad Dressing rule"
        print("✓ All expected rules found")
    else:
        print("❌ Chopped Salad category not found")
    
    # Test 2: Verify Chopped Salad rule parsing
    print("\nTest 2: Verify Chopped Salad rule parsing")
    chopped_salad_rule = None
    for rule in menu_parser.rules:
        if rule['name'] == "Chopped Salad":
            chopped_salad_rule = rule
            break
    
    if chopped_salad_rule:
        print(f"Found Chopped Salad rule with {len(chopped_salad_rule.get('options', []))} options")
        
        # Check Salad Base option
        salad_base = None
        for option in chopped_salad_rule.get('options', []):
            if option['name'] == "Salad Base":
                salad_base = option
                break
        
        if salad_base:
            print(f"Salad Base has {len(salad_base.get('items', []))} items")
            print(f"Constraints: min={salad_base['constraints'].get('min')}, max={salad_base['constraints'].get('max')}")
            assert salad_base['constraints'].get('min') == 1, "Salad Base should have min=1"
            assert salad_base['constraints'].get('max') == 1, "Salad Base should have max=1"
            assert len(salad_base.get('items', [])) == 3, "Salad Base should have 3 items"
            print("✓ Salad Base option correctly parsed")
        else:  
            print("❌ Salad Base option not found")
        
        # Check Salad Dressing option
        salad_dressing = None
        for option in chopped_salad_rule.get('options', []):
            if option['name'] == "Salad Dressing":
                salad_dressing = option
                break
        
        if salad_dressing:
            print(f"Salad Dressing has {len(salad_dressing.get('items', []))} items")
            print(f"Constraints: min={salad_dressing['constraints'].get('min')}, max={salad_dressing['constraints'].get('max')}")
            assert salad_dressing['constraints'].get('min') == 0, "Salad Dressing should have min=0"
            assert salad_dressing['constraints'].get('max') == 3, "Salad Dressing should have max=3"
            assert len(salad_dressing.get('items', [])) == 11, "Salad Dressing should have 11 items"
            print("✓ Salad Dressing option correctly parsed")
        else:
            print("❌ Salad Dressing option not found")
    else:
        print("❌ Chopped Salad rule not found")
    
    # Test 3: Verify indexing
    print("\nTest 3: Verify indexing")
    indexer = MenuIndexer()
    indexer.index_menu_and_rules(menu_parser)
    
    # Check if BYO Salad was indexed with rules
    results = indexer.items_col.query(
        query_texts=["BYO Salad"],
        include=["metadatas"],
        n_results=1
    )
    
    if results and results["metadatas"] and results["metadatas"][0]:
        metadata = results["metadatas"][0][0]
        print(f"BYO Salad metadata: {metadata}")
        assert 'selected_rules' in metadata, "BYO Salad should have selected_rules in metadata"
        rules = json.loads(metadata.get('selected_rules', '[]'))
        assert "Salad Add-ons" in rules, "Missing Salad Add-ons rule in indexed data"
        assert "Salad Base" in rules, "Missing Salad Base rule in indexed data"
        assert "Salad Dressing" in rules, "Missing Salad Dressing rule in indexed data"
        print("✓ BYO Salad correctly indexed with rules")
    else:
        print("❌ BYO Salad not found in indexed data")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
    #test_local_honey_menu()

