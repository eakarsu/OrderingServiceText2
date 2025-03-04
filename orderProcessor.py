from typing import Dict, List, Union, Optional
import json
import re
from menuIndexer import MenuIndexer, MenuParser
import itertools
from fuzzywuzzy import fuzz
from collections import Counter

class OrderProcessor:
    def __init__(self, indexer):
        self.indexer = indexer
        print("[OrderProcessor] Initialized.")

    def process_order(self, query):
        print(f"[DEBUG] Processing order query: {query}")
        
        # First check if query matches a category name exactly
        category_results = self.indexer.categories_col.query(
            query_texts=[query],
            n_results=1  # Only get the top match
        )
        
        if category_results and len(category_results["ids"]) > 0:
            # Check if the similarity score is high enough to consider it a match
            similarity_score = category_results["distances"][0]
            if similarity_score[0] < 0.3:  # Adjust threshold as needed
                category_name = category_results["documents"][0][0]  # Extract the string from the list
                print(f"[DEBUG] Found potential category match: {category_name} with score {similarity_score}")
                
                # Get all items in this category
                items_in_category = self.indexer.items_col.get(
                    where={"category": category_name}
                )
                
                if items_in_category and len(items_in_category["ids"]) > 1:
                    print(f"[DEBUG] Category {category_name} has multiple items: {len(items_in_category['ids'])}")
                    # Return all items in this category for user selection
                    results = []
                    # Assuming items_in_category["documents"] and items_in_category["metadatas"] are lists
                    for doc, meta in zip(items_in_category["documents"], items_in_category["metadatas"]):
                        item = {    
                            "item": doc,
                            "category": category_name,
                            "price": meta.get("price", 0),
                            "ingredients": meta.get("ingredients", "")
                        }
                        
                        if "base_price" in meta:
                            item["base_price"] = meta["base_price"]
                        
                        if "selected_rules" in meta:
                            item["selected_rules"] = meta["selected_rules"]
                                                       
                        results.append(item)
                    
                    print(f"[DEBUG] Returning {len(results)} items for category selection")
                    if len(results) == 2:
                        updated_selected_rules = json.loads(results[1]['selected_rules'])  
                        del updated_selected_rules[0]
                        del updated_selected_rules[0]   
                        results[1]["selected_rules"]  = json.dumps(updated_selected_rules)      

                    return {
                        "status": "need_category_selection",
                        "category": category_name,
                        "results": results
                    }
         
        # If not a category match or only one item in category, proceed with regular search
        results = self.indexer.items_col.query(
            query_texts=[query],
            n_results=5
        )
        
        if not results or len(results["ids"]) == 0:
            print(f"[DEBUG] No results found for query: {query}")
            return {"status": "no_results"}
        
        # Check if there's an exact match
        exact_match = None
        # If results["documents"] is a list containing another list
        if results["documents"] and isinstance(results["documents"][0], list):
            documents = results["documents"][0]  # Get the inner list
            for i, doc in enumerate(documents):
                # Use partial matching - check if query is part of the document name
                exact_match = self.find_exact_match(i,query.lower(), doc.lower())#i
                if exact_match is not None:
                    break
        elif results["documents"]:
            # Handle case where documents is a flat list
            for i, doc in enumerate(results["documents"]):
                exact_match = self.find_exact_match(i,query.lower(), doc.lower()) #i
                if exact_match is not None:
                    break
                

        if exact_match is not None:
            print(f"[DEBUG] Found exact match: {results['documents'][exact_match]}")
            meta_list = results["metadatas"][exact_match]
            
            # Extract the dictionary from the list
            if isinstance(meta_list, list):
                meta = meta_list[0]
            else:
                meta = meta_list
            
            # Now proceed with the dictionary
            if "selected_rules" in meta:
                print(f"[DEBUG] Item has rules: {meta['selected_rules']}")
                return self._process_rule_based_item(
                    query,
                    meta
                )
        # No exact match, return similar items
        similar_items = []

        for doc, meta_list in zip(results["documents"], results["metadatas"]):
            # Handle case where doc is a list
            if isinstance(doc, list):
                doc = doc[0]
            
            # Handle case where meta is a list
            if isinstance(meta_list, list):
                meta = meta_list[0]
            else:
                meta = meta_list
            
            item = {
                "item": doc,
                "category": meta.get("category", ""),
                "price": meta.get("price", 0),
                "ingredients": meta.get("ingredients", "")
            }
            
            if "base_price" in meta:
                item["base_price"] = meta["base_price"]
            
            if "selected_rules" in meta:
                item["selected_rules"] = meta["selected_rules"]
            
            similar_items.append(item)


        print(f"[DEBUG] Returning {len(similar_items)} similar items")
        return {
                    "status": "need_input",
                    "results": similar_items
                }


    def compare_strings(self,str1, str2):
        words1 = str1.lower().split()
        words2 = str2.lower().split()
        common_words = set(words1) & set(words2)
        common_count = len(common_words)
        
        if common_count == 0:
            return 0
        
        fuzzy_score = fuzz.ratio(str1.lower(), str2.lower()) / 100
        word_score = 2 * common_count / (len(words1) + len(words2))
        combined_score = (fuzzy_score + word_score) / 2
        
        return combined_score  # Return a single float value

    def find_exact_match(self, index,query, document):
        score = self.compare_strings(query, document)
        return index if score > 0.4 else None


    def _process_rule_based_item(self, item_name, meta):
        print(f"[DEBUG] Processing rule-based item: {item_name} with base price ${meta.get('base_price', 0)}")
        
        # Extract rules from metadata
        rules = []
        if "selected_rules" in meta:
            try:
                rules = json.loads(meta["selected_rules"])
                print(f"[DEBUG] Successfully parsed rules: {rules}")
            except Exception as e:
                print(f"[DEBUG] Error parsing rules: {meta['selected_rules']} - {str(e)}")
                if isinstance(meta["selected_rules"], list):
                    rules = meta["selected_rules"]
        
        # Get all rule items for this category at once
        category = meta.get("category")
        all_rule_items = self.indexer.rule_items_col.get(
            where={"rule": category}
        )
        
        print(f"[DEBUG] Found {len(all_rule_items['ids'])} total rule items for category {category}")
        
        # Group items by their option
        items_by_option = {}
        for i, item_id in enumerate(all_rule_items["ids"]):
            item_doc = all_rule_items["documents"][i]
            item_meta = all_rule_items["metadatas"][i]
            option_name = item_meta["option"]
            if item_name in option_name and not item_name in option_name:
                continue 
            if option_name not in items_by_option :
                items_by_option [option_name] = []
            
            items_by_option[option_name].append({
                "name": item_doc,
                "price": float(item_meta["price"])
            })
        
        # Get rule options for this category
        rule_options = self.indexer.rule_options_col.get(
            where={"rule": category}
        )
        
        # Build available options for each rule
        available_options = {}
        for rule_name in rules:
            print(f"[DEBUG] Processing rule: {rule_name}")
            
            # Get rule constraints from rule options
            rule_option = next((opt for opt in rule_options["metadatas"] if opt["name"] == rule_name), None)
            if rule_option:
                min_val = rule_option["min"]
                max_val = rule_option["max"]
                print(f"[DEBUG] Found constraints for {rule_name}: min={min_val}, max={max_val}")
            else:
                min_val = 1
                max_val = 1
                print(f"[DEBUG] No constraints found for {rule_name}, using defaults: min={min_val}, max={max_val}")
            
            if rule_name in items_by_option:
                available_options[rule_name] = [{
                    "name": rule_name,
                    "min": min_val,
                    "max": max_val,
                    "items": items_by_option[rule_name]
                }]
                print(f"[DEBUG] Found {len(items_by_option[rule_name])} items for rule {rule_name}")
        
        # Make sure the base price is stored as a float for calculations
        base_price = float(meta.get("base_price", 0))
        
        return {
            "status": "need_rule_selections",
            "item": item_name,
            "base_price": base_price,
            "rules": rules,
            "available_options": available_options
        }

 

    def _process_rule_based_item_old2(self, item_name, meta):
        """
        Process a rule-based item like Breakfast or Bagel that requires selections.
        """
        print(f"[DEBUG] Processing rule-based item: {item_name} with base price ${meta.get('base_price', 0)}")
        
        # Extract rules from metadata
        rules = []
        if "selected_rules" in meta:
            try:
                rules = json.loads(meta["selected_rules"])
                print(f"[DEBUG] Successfully parsed rules: {rules}")
            except Exception as e:
                print(f"[DEBUG] Error parsing rules: {meta['selected_rules']} - {str(e)}")
                if isinstance(meta["selected_rules"], list):
                    rules = meta["selected_rules"]
        
        # Get all rule items for this rule at once
        all_rule_items = self.indexer.rule_items_col.get(
            where={"rule": "BYO Breakfast"}  # Use the actual rule name from the database
        )
        
        print(f"[DEBUG] Found {len(all_rule_items['ids'])} total rule items")
        
        # Group items by their option
        items_by_option = {}
        for i, item_id in enumerate(all_rule_items["ids"]):
            item_doc = all_rule_items["documents"][i]
            item_meta = all_rule_items["metadatas"][i]
            option_name = item_meta["option"]
            
            if option_name not in items_by_option:
                items_by_option[option_name] = []
            
            items_by_option[option_name].append({
                "name": item_doc,
                "price": float(item_meta["price"])
            })
        
        # Build available options for each rule
        available_options = {}
        for rule_name in rules:
            print(f"[DEBUG] Processing rule: {rule_name}")
            
            if rule_name in items_by_option:
                # This rule name matches an option name directly
                available_options[rule_name] = [{
                    "name": rule_name,
                    "min": 1,  # Default constraints
                    "max": 1,
                    "items": items_by_option[rule_name]
                }]
                print(f"[DEBUG] Found {len(items_by_option[rule_name])} items for rule {rule_name}")
            else:
                # Check if any option contains this rule name
                matching_options = []
                for option_name in items_by_option:
                    if rule_name in option_name or option_name in rule_name:
                        matching_options.append({
                            "name": option_name,
                            "min": 1,  # Default constraints
                            "max": None if "Spreads" in option_name else 1,  # Special handling for spreads
                            "items": items_by_option[option_name]
                        })
                        print(f"[DEBUG] Found matching option {option_name} with {len(items_by_option[option_name])} items")
                
                if matching_options:
                    available_options[rule_name] = matching_options
        
        # Make sure the base price is stored as a float for calculations
        base_price = float(meta.get("base_price", 0))
        
        return {
            "status": "need_rule_selections",
            "item": item_name,
            "base_price": base_price,
            "rules": rules,
            "available_options": available_options
        }



    def _process_rule_based_item_old(self, item_name, meta):
        """
        Process a rule-based item like Breakfast or Bagel that requires selections.
        """
        print(f"[DEBUG] Processing rule-based item: {item_name} with base price ${meta.get('base_price', 0)}")
        
        # Extract rules from metadata
        rules = []
        if "selected_rules" in meta:
            try:
                rules = json.loads(meta["selected_rules"])
                print(f"[DEBUG] Successfully parsed rules: {rules}")
            except Exception as e:
                print(f"[DEBUG] Error parsing rules: {meta['selected_rules']} - {str(e)}")
                # If parsing fails, try to use the value directly if it's already a list
                if isinstance(meta["selected_rules"], list):
                    rules = meta["selected_rules"]
                    print(f"[DEBUG] Using rules directly: {rules}")
        
        # Get available options for each rule
        available_options = {}
        for rule_name in rules:
            print(f"[DEBUG] Processing rule: {rule_name}")
            
            # Get rule options - use exact match on rule name
            rule_options_results = self.indexer.rule_items_col.get(
                where={"rule": meta["category"]} #rule_name}
            )
            
            print(f"[DEBUG] Query results for rule options: {rule_options_results}")
            
            if not rule_options_results or len(rule_options_results["ids"]) == 0:
                print(f"[DEBUG] No options found for rule: {rule_name}")
                continue
            
            options = []
            for i, option_id in enumerate(rule_options_results["ids"]):
                option_meta = rule_options_results["metadatas"][i]
                option_name = rule_options_results["documents"][i]
                
                print(f"[DEBUG] Found option: {option_name} for rule: {rule_name}")
                
                # Get items for this option - use exact match on rule name and option name
                rule_items_results = self.indexer.rule_items_col.get(
                    where={"$and": [
                        {"rule": meta["category"]},
                        {"item": option_name}
                    ]}
                    #where={"rule": rule_name, "option": option_name}
                )
                
                print(f"[DEBUG] Query results for rule items: {rule_items_results}")
                
                if not rule_items_results or len(rule_items_results["ids"]) == 0:
                    print(f"[DEBUG] No items found for option: {option_name}")
                    continue
                
                # Format items
                items = []
                for j, item_id in enumerate(rule_items_results["ids"]):
                    item_doc = rule_items_results["documents"][j]
                    item_meta = rule_items_results["metadatas"][j]
                    
                    # Ensure price is properly extracted and stored as a float
                    item_price = float(item_meta.get("price", 0))
                    print(f"[DEBUG] Adding rule item: {item_doc} with price: ${item_price}")
                    items.append({
                        "name": item_doc,
                        "price": item_price
                    })
                
                # Extract min/max constraints
                min_val = option_meta.get("min", 0)
                max_val = option_meta.get("max", None)
                
                options.append({
                    "name": option_name,
                    "min": min_val,
                    "max": max_val,
                    "items": items
                })
            
            if options:
                available_options[rule_name] = options
                print(f"[DEBUG] Found {len(options)} options with {len(options[0]['items'])} items for rule {rule_name}")
        
        # Make sure the base price is stored as a float for calculations
        base_price = float(meta.get("base_price", 0))
        print(f"[DEBUG] Final base price: ${base_price}")
        
        return {
            "status": "need_rule_selections",
            "item": item_name,
            "base_price": base_price,
            "rules": rules,
            "available_options": available_options
        }





    def _process_rule_based_item2(self, item_name, meta):
        """
        Process a rule-based item like Breakfast or Bagel that requires selections.
        
        Args:
            item_name: The name of the item
            meta: The metadata containing rules and other item info
        
        Returns:
            A dictionary with status, item, base_price, rules, and available_options
        """
        print(f"[DEBUG] Processing rule-based item: {item_name} with base price ${meta.get('base_price', 0)}")
        
        # Extract rules from metadata
        rules = []
        if "selected_rules" in meta:
            try:
                rules = json.loads(meta["selected_rules"])
            except:
                print(f"[DEBUG] Error parsing rules: {meta['selected_rules']}")
                rules = []
        
        # Get available options for each rule
        available_options = {}
        for rule_name in rules:
            print(f"[DEBUG] Processing rule: {rule_name}")
            
            # Get rule options
            rule_options = self.indexer.rule_options_col.get(
                where={"rule": rule_name}
            )
            
            if not rule_options or len(rule_options["ids"]) == 0:
                print(f"[DEBUG] No options found for rule: {rule_name}")
                continue
            
            options = []
            for option_name, option_meta in zip(rule_options["documents"], rule_options["metadatas"]):
                # Get items for this option
                rule_items = self.indexer.rule_items_col.get(
                    where={"rule": rule_name, "option": option_name}
                )
                
                if not rule_items or len(rule_items["ids"]) == 0:
                    print(f"[DEBUG] No items found for option: {option_name}")
                    continue
                
                # Format items
                items = []
                for item_doc, item_meta in zip(rule_items["documents"], rule_items["metadatas"]):
                    items.append({
                        "name": item_doc,
                        "price": item_meta.get("price", 0)
                    })
                
                options.append({
                    "name": option_name,
                    "min": option_meta.get("min", 0),
                    "max": option_meta.get("max", None),
                    "items": items
                })
            
            if options:
                available_options[rule_name] = options
                print(f"[DEBUG] Found {len(options[0]['items'])} items for rule {rule_name}")
        
        return {
            "status": "need_rule_selections",
            "item": item_name,
            "base_price": meta.get("base_price", 0),
            "rules": rules,
            "available_options": available_options
        }



    def process_rule_based_order(self, menu_item: str, selected_options: Dict[str, List[str]]) -> Dict:
        """
        Process an order for a menu item that uses rules
        
        Args:
            menu_item (str): The name of the menu item
            selected_options (dict): Dictionary mapping rule names to selected options
            
        Returns:
            Dict with order status and details
        """
        # Find the menu item
        item_result = self._exact_match_item(menu_item)
        if not item_result:
            return {"status": "error", "message": f"Menu item '{menu_item}' not found"}
        
        # Get the selected rules for this item
        selected_rules = []
        if 'selected_rules' in item_result:
            selected_rules = json.loads(item_result['selected_rules'])
        
        if not selected_rules:
            return {"status": "error", "message": f"Menu item '{menu_item}' does not have any rules"}
        
        # Validate that all required rules have selections
        missing_rules = []
        for rule_name in selected_rules:
            if rule_name not in selected_options:
                missing_rules.append(rule_name)
        
        if missing_rules:
            # Return the rules that need selections
            return {
                "status": "need_rule_selections",
                "message": f"Please select options for: {', '.join(missing_rules)}",
                "missing_rules": missing_rules,
                "available_options": {rule: self._get_rule_options(rule) for rule in missing_rules}
            }
        
        # Validate selections against rule constraints
        for rule_name, selections in selected_options.items():
            validation_result = self._validate_rule_selections(rule_name, selections)
            if validation_result["status"] != "success":
                return validation_result
        
        # Calculate total price
        base_price = item_result.get('base_price', 0)
        if not base_price and 'category' in item_result:
            # Try to get base price from category
            cat_result = self.indexer.categories_col.get(
                ids=[f"cat_{item_result['category']}"],
                include=["metadatas"]
            )
            if cat_result and cat_result['metadatas']:
                base_price = cat_result['metadatas'][0].get('base_price', 0)
        
        total_price = base_price
        
        # Add prices for selected options
        for rule_name, selections in selected_options.items():
            for selection in selections:
                option_price = self._get_option_price(rule_name, selection)
                total_price += option_price
        
        return {
            "status": "success",
            "item": menu_item,
            "base_price": base_price,
            "total_price": total_price,
            "selections": selected_options
        }

    def _handle_rule_based_item(self, item_data):
        """Handle items that have rules associated with them"""
        selected_rules = json.loads(item_data.get('selected_rules', '[]'))
        
        if not selected_rules:
            return {"status": "success", **item_data}
        
        # Return information about the rules that need to be selected
        return {
            "status": "need_rule_selections",
            "item": item_data.get('item'),
            "category": item_data.get('category'),
            "base_price": item_data.get('base_price', 0),
            "rules": selected_rules,
            "available_options": {rule: self._get_rule_options(rule) for rule in selected_rules}
        }

    
    def _get_rule_options(self, rule_name):
        """Get all available options for a specific rule"""
        # Get all rule options and filter them manually
        all_options = self.indexer.rule_options_col.get()
        
        options = []
        if all_options and all_options["documents"]:
            for doc, meta in zip(all_options["documents"], all_options["metadatas"]):
                # Check if this option belongs to the requested rule
                if doc == rule_name:
                    # Get the items for this option
                    items = self._get_rule_option_items(doc)
                    
                    options.append({
                        "name": doc,
                        "min": meta.get("min", 0),
                        "max": meta.get("max"),
                        "items": items
                    })
        
        return options


    def _get_rule_option_items(self, option_name):
        """Get all items available under a specific rule option"""
        # Get all rule items and filter them manually
        all_items = self.indexer.rule_items_col.get()
        
        items = []
        if all_items and all_items["documents"]:
            for doc, meta in zip(all_items["documents"], all_items["metadatas"]):
                # Check if this item belongs to the requested rule and option
                if meta.get('option') == option_name:
                    items.append({
                        "name": doc,
                        "price": meta.get("price", 0)
                    })
        
        return items



    def _validate_rule_selections(self, rule_name, selections):
        """Validate that selections comply with rule constraints"""
        # Get the rule options
        options = self._get_rule_options(rule_name)
        
        if not options:
            return {"status": "error", "message": f"No options found for rule {rule_name}"}
        
        # Get all valid items for this rule
        all_valid_items = []
        min_required = 0
        max_allowed = None
        
        for option in options:
            if option.get("min", 0) > min_required:
                min_required = option.get("min", 0)
            
            if option.get("max") is not None:
                if max_allowed is None or option.get("max") < max_allowed:
                    max_allowed = option.get("max")
            
            for item in option.get("items", []):
                all_valid_items.append(item["name"])
        
        # Check if all selections are valid items
        invalid_selections = [s for s in selections if s not in all_valid_items]
        if invalid_selections:
            return {
                "status": "error",
                "message": f"Invalid selections for {rule_name}: {', '.join(invalid_selections)}"
            }
        
        # Check minimum constraint
        if len(selections) < min_required:
            return {
                "status": "error",
                "message": f"You must select at least {min_required} items for {rule_name}"
            }
        
        # Check maximum constraint
        if max_allowed is not None and len(selections) > max_allowed:
            return {
                "status": "error",
                "message": f"You can select at most {max_allowed} items for {rule_name}"
            }
        
        return {"status": "success"}

    def _get_option_price(self, rule_name, selection):
        """Get the price of a specific option under a rule"""
        # Query the rule items collection to find the price
        results = self.indexer.rule_items_col.query(
            query_texts=[selection],
            include=["metadatas"],
            n_results=10
        )
        
        if results and results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                if meta.get('rule') == rule_name and selection in meta.get('name', ''):
                    return meta.get("price", 0)
        
        return 0.0

    def _exact_match_item(self, query: str) -> Union[Dict, None]:
        """Look for an exact match for the query in the items collection"""
        # Retrieve all items from the vector database
        items = self.indexer.items_col.get()
        
        # Iterate over all items to look for an exact match (ignoring case)
        for doc, meta in zip(items['documents'], items['metadatas']):
            if query.strip().lower() == doc.strip().lower():
                result = {
                    "item": doc,
                    "price": meta.get("price", 0),
                    "ingredients": meta.get("ingredients", ""),
                    "category": meta.get("category", "Unknown")
                }
                
                # Add base price if present
                if 'base_price' in meta:
                    result['base_price'] = meta['base_price']
                
                # Add selected rules if present
                if 'selected_rules' in meta:
                    result['selected_rules'] = meta['selected_rules']
                
                return result
        
        # If nothing matched exactly, return None
        return None

    def _vector_search_items(self, query: str, n: int = 3) -> List[Dict]:
        """Use vector search to find the most relevant items for the query"""
        # Compute the embedding for the query
        query_embedding = self.indexer.embedder([query])
        
        # Query the items collection using vector search
        result = self.indexer.items_col.query(
            query_embeddings=query_embedding,
            n_results=n,
            include=["documents", "metadatas"]
        )
        
        results_list = []
        if result and result.get("documents") and result["documents"][0]:
            for i in range(min(n, len(result["documents"][0]))):
                doc = result["documents"][0][i]
                meta = result["metadatas"][0][i]
                
                item_result = {
                    "item": doc,
                    "price": meta.get("price", 0),
                    "ingredients": meta.get("ingredients", ""),
                    "category": meta.get("category", "Unknown")
                }
                
                # Add base price if present
                if 'base_price' in meta:
                    item_result['base_price'] = meta['base_price']
                
                # Add selected rules if present
                if 'selected_rules' in meta:
                    item_result['selected_rules'] = meta['selected_rules']
                
                results_list.append(item_result)
        
        return results_list


    def get_categories(self):
        """Get all available category names in the system"""
        categories = self.indexer.categories_col.get()
        if categories and categories["documents"]:
            return categories["documents"]
        return []

    def get_items_in_category(self, category_name):
        """Get all items in a specific category"""
        items = self.indexer.items_col.get(
            where={"category": category_name}
        )
        
        if items and items["documents"]:
            result = []
            for doc, meta in zip(items["documents"], items["metadatas"]):
                item = {
                    "name": doc,
                    "price": meta.get("price", 0),
                    "ingredients": meta.get("ingredients", "")
                }
                
                if 'base_price' in meta:
                    item['base_price'] = meta['base_price']
                
                if 'selected_rules' in meta:
                    item['rules'] = json.loads(meta['selected_rules'])
                
                result.append(item)
            return result
        return []


if __name__ == "__main__":
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
        print(f"Selected rules: {salad_category.get('selected_rules', [])}")
        assert "Salad Add-ons" in salad_category.get('selected_rules', []), "Missing Salad Add-ons rule"
        assert "Salad Base" in salad_category.get('selected_rules', []), "Missing Salad Base rule"
        assert "Salad Dressing" in salad_category.get('selected_rules', []), "Missing Salad Dressing rule"
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
    
    # Test 3: Verify BYO Breakfast rules
    print("\nTest 3: Verify BYO Breakfast rules")
    breakfast_category = None
    for category in menu_parser.categories:
        if category['name'] == "BYO Breakfast":
            breakfast_category = category
            break
    
    if breakfast_category:
        print(f"Found BYO Breakfast category with items:")
        for item in breakfast_category.get('items', []):
            print(f"  - {item.get('name')} (${item.get('price'):.2f})")
            if 'selected_rules' in item:
                print(f"    Selected rules: {item.get('selected_rules')}")
        
        # Check for Breakfast item with rules
        breakfast_item = None
        for item in breakfast_category.get('items', []):
            if item.get('name') == "Breakfast,Base Price":
                breakfast_item = item
                break
        
        if breakfast_item:
            print(f"Found Breakfast item with base price: ${breakfast_item.get('base_price', 0)}")
            assert breakfast_item.get('price') == 2.60, "Breakfast should have base price $2.60"
            print("✓ Breakfast item correctly parsed")
        else:
            print("❌ Breakfast item not found")
    else:
        print("❌ BYO Breakfast category not found")
    
    # Test 4: Verify BYO Breakfast rule parsing
    print("\nTest 4: Verify BYO Breakfast rule parsing")
    breakfast_rule = None
    for rule in menu_parser.rules:
        if rule['name'] == "BYO Breakfast":
            breakfast_rule = rule
            break
    
    if breakfast_rule:
        print(f"Found BYO Breakfast rule with {len(breakfast_rule.get('options', []))} options")
        
        # Check Breakfast Egg Quantity option
        egg_quantity = None
        for option in breakfast_rule.get('options', []):
            if option['name'] == "Breakfast Egg Quantity":
                egg_quantity = option
                break
        
        if egg_quantity:
            print(f"Breakfast Egg Quantity has {len(egg_quantity.get('items', []))} items")
            print(f"Constraints: min={egg_quantity['constraints'].get('min')}, max={egg_quantity['constraints'].get('max')}")
            assert egg_quantity['constraints'].get('min') == 1, "Breakfast Egg Quantity should have min=1"
            assert egg_quantity['constraints'].get('max') == 1, "Breakfast Egg Quantity should have max=1"
            print("✓ Breakfast Egg Quantity option correctly parsed")
        else:
            print("❌ Breakfast Egg Quantity option not found")
    else:
        print("❌ BYO Breakfast rule not found")

    # Test 5: Verify indexing
    print("\nTest 5: Verify indexing")
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
    
    # Test 6: Verify rule options indexing
    print("\nTest 6: Verify rule options indexing")
    rule_options = indexer.rule_options_col.get()
    
    if rule_options and rule_options["documents"]:
        print(f"Found {len(rule_options['documents'])} rule options in the database")
        
        # Check for Salad Base option
        salad_base_found = False
        for doc, meta in zip(rule_options["documents"], rule_options["metadatas"]):
            if doc == "Salad Base" and meta.get("rule") == "Chopped Salad":
                salad_base_found = True
                print(f"Found Salad Base option with constraints: min={meta.get('min')}, max={meta.get('max')}")
                assert meta.get('min') == 1, "Salad Base should have min=1"
                assert meta.get('max') == 1, "Salad Base should have max=1"
                break
        
        if salad_base_found:
            print("✓ Salad Base option correctly indexed")
        else:
            print("❌ Salad Base option not found in indexed data")
    else:
        print("❌ No rule options found in indexed data")
    
    # Test 7: Verify rule items indexing
    print("\nTest 7: Verify rule items indexing")
    rule_items = indexer.rule_items_col.get()
    
    if rule_items and rule_items["documents"]:
        print(f"Found {len(rule_items['documents'])} rule items in the database")
        
        # Check for Romaine item
        romaine_found = False
        for doc, meta in zip(rule_items["documents"], rule_items["metadatas"]):
            if doc == "Romaine (Small)" and meta.get("rule") == "Chopped Salad" and meta.get("option") == "Salad Base":
                romaine_found = True
                print(f"Found Romaine item with price: ${meta.get('price'):.2f}")
                assert meta.get('price') == 0.0, "Romaine should have price $0.00"
                break
        
        if romaine_found:
            print("✓ Romaine item correctly indexed")
        else:
            print("❌ Romaine item not found in indexed data")
    else:
        print("❌ No rule items found in indexed data")
    
    # Test cases
    test_queries = [
        "I want a salad",
        "BYO Salad",
        "Cool Ranch Doritos",
        "French Toast",
        "a salad with romaine, mixed greens, grilled chicken, tomatoes, olives and balsalmic vinegarette",
        "breakfast sandwich on croissant, 2 eggs, mushrooms, grilled chicken and tomatoes",
        "omelet"
    ]
    processor = OrderProcessor(indexer)
    for query in test_queries:
        print(f"\n{'='*50}\nTesting query: {query}")
        result = processor.process_order(query)
        print("Result:", result)

    print("\nAll tests completed!")
