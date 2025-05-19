#!/usr/bin/env python3
import os
import json
import re
import psycopg2
from datetime import datetime
from typing import TypedDict, List, Union, Annotated

from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI

# Use updated Chroma client API
from chromadb import Client
from chromadb.config import Settings
from collections import defaultdict
from menuIndexer import MenuIndexer,MenuParser
from orderProcessor import OrderProcessor

##########################
# Pydantic models for order validation
##########################
class MenuItem(BaseModel):
    item: str
    size: str
    quantity: int
    custom: str = ""
    price: str

class OrderSchema(BaseModel):
    message_type: str = "order"
    phone_number: str
    menu_items_ordered: List[MenuItem]
    pickup_or_delivery: str
    payment_type: str
    address: str
    total_price: str

##########################
# Workflow state definitions
##########################
class WorkflowState:
    COLLECT_ITEMS = "collect_items"
    DELIVERY_INFO = "delivery_info"
    PAYMENT = "payment"
    FINALIZED = "finalized"
    MODIFY_ORDER = "modify_order"  # New state for modifications

##########################
# Define the AgentState type for our state machine
##########################
class AgentState(TypedDict):
    user_input: Annotated[str, "single_value"]
    chat_history: List[Union[HumanMessage, AIMessage]]
    order_status: str
    current_order: dict
    input: str         # For call_agent: raw user input
    tools: list        # List of tools if applicable
    error_count: int

##########################
# Main OrderSystem class using state-based workflow, OpenRouter API, and Chroma-based RAG
##########################
class orderChat:
    def __init__(self, caller_id: str):
        self.caller_id = caller_id

        # Set up API key and URL for OpenRouter AI API.
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1"
        self.llm_local = ChatOpenAI(
            model="openai/chatgpt-4o-latest",
            #model="google/gemma-3-27b-it",
            openai_api_key=self.api_key,
            openai_api_base=self.api_url,
            default_headers={
                "Referer": "norshin.com",
                "X-Title": "Your App Name",
                "Content-Type": "application/json"
            },
            max_retries=3
        )

        # Initial order state.
        self.state = {
            "order_status": WorkflowState.COLLECT_ITEMS,
            "current_order": {
                "phone_number": caller_id,
                "menu_items_ordered": [],
                "pickup_or_delivery": "",
                "payment_type": "",
                "address": "",
                "total_price": "0.00"
            },
            "chat_history": [],
            "error_count": 0
        }

        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "database": os.getenv("DB_DATABASE"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD")
        }
        self.test_database_connection()

        self.load_prompts("misc2/prompt.txt","misc2/prompt3.txt")
        # Initialize Chroma vector store with new client API.
        self.processor = self.getProcessor("misc2/prompt2.txt","misc2/rules.txt")

        """  self.load_prompts("salon/prompt.txt","salon/prompt3.txt")
        # Initialize Chroma vector store with new client API.
        self.processor = self.getProcessor("salon/salon_prompt2.txt","salon/salon_rules.txt") """

        # Create retriever by using the collection query.
        self.workflow = self.create_state_machine()


    def get_state(self):
        # Return only serializable state (dict, list, str, int, etc.)
        return self.state

    def set_state(self, new_state):
        # Return only serializable state (dict, list, str, int, etc.)
         self.state = new_state
         
    def getProcessor(self, prompt2File, rulesFile):
        """Get or create an OrderProcessor with indexed menu and rules data"""
        try:
            # Try to connect to existing collections first
            indexer = MenuIndexer()
            
            # Check if required collections exist
            collections_exist = self._check_collections_exist(indexer)
            
            if collections_exist:
                print("Using existing menu and rule collections")
            else:
                # Create a MenuParser instance and parse files
                menu_parser = MenuParser()
                print("Parsing menu file: salon/salon_prompt2.txt")
                menu_parser.parse_menu_file(prompt2File)
                print("Parsing rules file: salon/salon_rules.txt")
                menu_parser.parse_rules_file(rulesFile)
                
                # Index the parsed menu and rules
                indexer.index_menu_and_rules(menu_parser)
                
            # Create and return the processor with the indexer
            processor = OrderProcessor(indexer)
            return processor
            
        except Exception as e:
            print(f"Error in getProcessor: {str(e)}")
            # Fallback to complete reindexing if any errors occur
            menu_parser = MenuParser()
            print("Parsing menu file: misc2/prompt2.txt")
            menu_parser.parse_menu_file(prompt2File)
            print("Parsing rules file: misc2/rules.txt")
            menu_parser.parse_rules_file(rulesFile)
            indexer = MenuIndexer()
            indexer.index_menu_and_rules(menu_parser)
            processor = OrderProcessor(indexer)
            return processor

    def _check_collections_exist(self, indexer):
        """Check if all required collections exist and have data"""
        required_collections = [
            "categories", "items", "rules", 
            "rule_options", "rule_items"
        ]
        
        try:
            # Initialize collections
            indexer._initialize_collections()
            
            # Check if each collection exists and has data
            for collection_name in required_collections:
                collection = getattr(indexer, f"{collection_name}_col", None)
                if collection is None:
                    return False
                    
                # Check if collection has data
                count = collection.count()
                if count == 0:
                    return False
                    
            return True
        except Exception as e:
            print(f"Error checking collections: {str(e)}")
            return False


    def keepChromeDB(self):
        client = Client(Settings(persist_directory="chroma_database"))
        try:
            self.collection = client.get_collection("category_prompts")
            print("Vector DB found. Using stored prompts.")
        except Exception as e:
            print("No existing collection found. Creating a new one using prompt2 file...")
            self.collection = client.create_collection("category_prompts")
            try:
                with open("misc/prompt2", "r") as f:
                    prompt2_text = f.read().strip()
            except Exception as e:
                prompt2_text = ""
                print(f"Error reading prompt2 file: {e}")
            self.collection.add(
                documents=[prompt2_text],
                metadatas=[{"id": "1"}],
                ids=["1"]
            )

    def test_database_connection(self):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            print("Database connection successful")
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {str(e)}")

    def get_user_home_address(self, phone_number: str) -> str:
        """
        Retrieve the customer's home (delivery) address from the most recent order.
        Assumes that the orders table stores order_data as JSON (a string) that
        includes the key "address".
        """
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Limit to the most recent order.
                    cur.execute("""
                        SELECT order_data 
                        FROM orders 
                        WHERE phone_number = %s 
                        AND order_data::json->>'pickup_or_delivery' = 'delivery'
                        ORDER BY orderdate DESC 
                        LIMIT 1
                    """, (phone_number,))
                    row = cur.fetchone()
                    if row:
                        order_data_address = row[0]["address"]
                        return order_data_address
            return ""
        except Exception as e:
            print(f"Error retrieving user's home address: {e}")
            return ""


    def get_top_ordered_items_by_phone(self, phone_number: str) -> list:
        """
        Retrieve the top 3 most ordered items for the given phone number by aggregating
        quantities from the JSON order_data stored in the database.
        Assumes order_data is stored as a JSONB column and that the key "menu_items_ordered"
        is an array of objects with each containing "item" and "quantity".
        """
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT item_elem->>'item' as item_name,
                            SUM((item_elem->>'quantity')::int) as total_quantity
                        FROM orders, jsonb_array_elements(order_data->'menu_items_ordered') as item_elem
                        WHERE phone_number = %s
                        GROUP BY item_elem->>'item'
                        ORDER BY total_quantity DESC
                        LIMIT 3;
                    """
                    cur.execute(query, (phone_number,))
                    results = cur.fetchall()
                    return results  # Each row is (item_name, total_quantity)
        except Exception as e:
            print(f"Database error in get_top_ordered_items: {str(e)}")
            return []


    def load_prompts(self, promptFile, prompt3File):
        """Load prompt text from files (prompt, prompt2, prompt3) and append selective context."""
        try:
            with open(promptFile, "r") as f:
                self.prompt_text = f.read().strip()
            with open(prompt3File, "r") as f:
                self.prompt_text += " " + f.read().strip()
            # Instead of dumping dates for all orders, get only the top 3 items (most ordered).
            top_items = self.get_top_ordered_items_by_phone(self.caller_id)
            if top_items:
                # Format a summary string: item_name (total_quantity)
                summary = " Top 3 most ordered items: " + ", ".join([f"{row[0]} ({row[1]})" for row in top_items])
                self.prompt_text += summary
            
            home_address = self.get_user_home_address(self.caller_id)
            if home_address:
                self.prompt_text += f" Customer home address: {home_address}"
            print (f" home address :{home_address}")

            self.prompt_text += f" Phone number: {self.caller_id}"
            print("Loaded prompt text.")
        except FileNotFoundError as e:
            print(f"Prompt file error: {str(e)}")
            self.prompt_text = "Welcome to our ordering system."
  

    def get_order_data_by_phonenumber(self, phone_number: str) -> list:
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT order_data FROM orders 
                        WHERE phone_number = %s 
                        ORDER BY orderdate DESC
                    """, (phone_number,))
                    return [row for row in cur.fetchall()]
        except Exception as e:
            print(f"Database error: {str(e)}")
            return []
    
    def get_prompt_for_selection(self, user_input: str) -> str:
        """
        Query the Chroma vector store for a relevant prompt based on the user's input.
        If the input is too short (less than 3 characters) or is a common greeting,
        do not perform a search and return an empty string.
        """
        common_greetings = {"hi", "hello", "hey"}
        trimmed = user_input.strip()
        if not trimmed or len(trimmed) < 3 or trimmed.lower() in common_greetings:
            print("Input too short or generic; skipping vector DB search.")
            return ""
        result = self.processor.process_order(trimmed)
        
        if result and "documents" in result and len(result["documents"]) > 0 and len(result["documents"][0]) > 0:
            doc = result["documents"][0][0]
            # Further fuzzy check: ensure the document has a significant overlap with the query
            if trimmed.lower() in doc.lower():
                print("Retrieved prompt from vector DB for selection.")
                return doc
        return ""


    def get_prompt_for_selection2(self, user_input: str) -> str:
        """
        Query the Chroma vector store for a relevant prompt based on the user's input.
        If the input is too short (less than 3 characters) or is a common greeting,
        do not perform a search and return an empty string.
        """
        common_greetings = {"hi", "hello", "hey"}
        trimmed = user_input.strip()
        if not trimmed or len(trimmed) < 3 or trimmed.lower() in common_greetings:
            print("Input too short or generic; skipping vector DB search.")
            return ""
        result = self.collection.query(query_texts=[trimmed], n_results=1)
        if result and "documents" in result and len(result["documents"]) > 0 and len(result["documents"][0]) > 0:
            doc = result["documents"][0][0]
            # Further fuzzy check: ensure the document has a significant overlap with the query
            if trimmed.lower() in doc.lower():
                print("Retrieved prompt from vector DB for selection.")
                return doc
        return ""
    
    def create_state_machine(self):
        workflow = StateGraph(AgentState)
        workflow.add_node(WorkflowState.COLLECT_ITEMS, self.handle_collect_items)
        workflow.add_node(WorkflowState.DELIVERY_INFO, self.handle_delivery_info)
        workflow.add_node(WorkflowState.PAYMENT, self.handle_payment)
        workflow.add_node(WorkflowState.FINALIZED, self.handle_finalized)
        workflow.add_node(WorkflowState.MODIFY_ORDER, self.handle_modify_order)
        workflow.set_entry_point(WorkflowState.COLLECT_ITEMS)
        workflow.add_conditional_edges(
            WorkflowState.COLLECT_ITEMS,
            lambda s: WorkflowState.MODIFY_ORDER if "modify" in s["user_input"].lower() else WorkflowState.DELIVERY_INFO,
            {WorkflowState.MODIFY_ORDER: WorkflowState.MODIFY_ORDER, WorkflowState.DELIVERY_INFO: WorkflowState.DELIVERY_INFO}
        )
        workflow.add_conditional_edges(
            WorkflowState.DELIVERY_INFO,
            lambda s: WorkflowState.MODIFY_ORDER if "modify" in s["user_input"].lower() else WorkflowState.PAYMENT,
            {WorkflowState.MODIFY_ORDER: WorkflowState.MODIFY_ORDER, WorkflowState.PAYMENT: WorkflowState.PAYMENT}
        )
        workflow.add_conditional_edges(
            WorkflowState.PAYMENT,
            lambda s: WorkflowState.MODIFY_ORDER if "modify" in s["user_input"].lower() else WorkflowState.FINALIZED,
            {WorkflowState.MODIFY_ORDER: WorkflowState.MODIFY_ORDER, WorkflowState.FINALIZED: WorkflowState.FINALIZED}
        )
        workflow.add_edge(WorkflowState.FINALIZED, END)
        return workflow.compile()

    def parse_order_data(self, content: str) -> dict:
        print(f"\n=== PARSING ORDER DATA ===\nInput: {content}")
        if "{" in content and "}" in content:
            try:
                order_data = json.loads(content)
                print("Direct JSON parse successful:", order_data)
                validated = OrderSchema(**order_data)
                print("Schema validation passed")
                return validated.dict()
            except Exception as e:
                print(f"Direct JSON parsing failed: {str(e)}")
        order_data = {
            "message_type": "order",
            "phone_number": self.caller_id,
            "menu_items_ordered": [],
            "pickup_or_delivery": "",
            "payment_type": "",
            "address": "",
            "total_price": "0.00"
        }
        item_matches = re.findall(r'(\d+)\s+([^,$]+)\s+(\$[\d\.]+)', content, re.IGNORECASE)
        if item_matches:
            for qty, name, price in item_matches:
                order_data["menu_items_ordered"].append({
                    "item": name.strip(),
                    "size": "Regular",
                    "quantity": int(qty),
                    "custom": "",
                    "price": price
                })
        else:
            print("Fallback extraction: No item matches found.")
        delivery_match = re.search(r'\b(delivery|pickup)\b', content, re.IGNORECASE)
        if delivery_match:
            order_data["pickup_or_delivery"] = delivery_match.group(1).lower()
        payment_match = re.search(r'\b(cash|credit|card|venmo|paypal)\b', content, re.IGNORECASE)
        if payment_match:
            order_data["payment_type"] = payment_match.group(1).lower()
        address_match = re.search(r'(?:address:|to)\s*([\w\s,.-]+)', content, re.IGNORECASE)
        if address_match:
            order_data["address"] = address_match.group(1).strip()
        if order_data["menu_items_ordered"]:
            total = sum(float(item["price"].strip("$")) * item["quantity"] for item in order_data["menu_items_ordered"])
            order_data["total_price"] = f"${total:.2f}"
        print("Fallback extraction result:", order_data)
        return order_data if order_data["menu_items_ordered"] else None

    def handle_collect_items(self, state: AgentState):
        print(f"\n=== COLLECTING ITEMS ===\nUser input: {state['user_input']}")
        try:
            order_data = self.parse_order_data(state["user_input"])
            if not order_data or not order_data.get("menu_items_ordered"):
                raise ValueError("Could not detect order items")
            print("Validated order data:", order_data)
            state["error_count"] = 0
            return {
                **state,
                "current_order": order_data,
                "order_status": WorkflowState.DELIVERY_INFO,
                "chat_history": state["chat_history"] + [
                    AIMessage(content="Items detected. Please confirm delivery/pickup information.")
                ]
            }
        except Exception as e:
            return self.handle_error(state, f"Item error: {str(e)}")

    def handle_delivery_info(self, state: AgentState):
        print(f"\n=== PROCESSING DELIVERY INFO ===\nInput: {state['user_input']}")
        try:
            print("Current order state:", state["current_order"])
            delivery_match = re.search(r'(delivery|pickup)(?:.*?address:)?\s*([^\n]*)', state["user_input"], re.IGNORECASE)
            if not delivery_match:
                raise ValueError("Missing delivery/pickup specification")
            print("Delivery match groups:", delivery_match.groups())
            return {
                **state,
                "current_order": {
                    **state["current_order"],
                    "pickup_or_delivery": delivery_match.group(1).lower(),
                    "address": delivery_match.group(2).strip() if delivery_match.group(2) else ""
                },
                "order_status": WorkflowState.PAYMENT,
                "chat_history": state["chat_history"] + [
                    AIMessage(content="Delivery information saved.")
                ]
            }
        except Exception as e:
            return self.handle_error(state, f"Delivery error: {str(e)}")

    def handle_payment(self, state: AgentState):
        print(f"\n=== PROCESSING PAYMENT ===\nInput: {state['user_input']}")
        try:
            print("Current order state:", state["current_order"])
            payment_match = re.search(r'(cash|credit|card|venmo|paypal)\b', state["user_input"], re.IGNORECASE)
            if not payment_match:
                raise ValueError("Unsupported payment method")
            print("Payment method matched:", payment_match.group(1))
            return {
                **state,
                "current_order": {
                    **state["current_order"],
                    "payment_type": payment_match.group(1).lower()
                },
                "order_status": WorkflowState.FINALIZED,
                "chat_history": state["chat_history"] + [
                    AIMessage(content="Payment information saved.")
                ]
            }
        except Exception as e:
            return self.handle_error(state, f"Payment error: {str(e)}")

    def handle_finalized(self, state: AgentState):
        print("\n=== FINALIZING ORDER ===")
        try:
            print("Final order data:", state["current_order"])
            required_fields = ["phone_number", "menu_items_ordered", "payment_type"]
            if state["current_order"].get("pickup_or_delivery", "").lower() == "delivery":
                required_fields.append("address")
            print("Required fields:", required_fields)
            if not all(state["current_order"].get(field) for field in required_fields):
                raise ValueError("Missing required order fields")
            print("Saving to database...")
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO orders (phone_number, order_data)
                           VALUES (%s, %s)""",
                        (self.caller_id, json.dumps(state["current_order"]))
                    )
                    conn.commit()
            print("Database insert successful")
            return {
                **state,
                "order_status": WorkflowState.FINALIZED,
                "chat_history": state["chat_history"] + [
                    AIMessage(content="Order finalized successfully!")
                ]
            }
        except Exception as e:
            return self.handle_error(state, f"Finalization failed: {str(e)}")

    def handle_modify_order(self, state: AgentState):
        print(f"\n=== MODIFYING ORDER ===\nUser input: {state['user_input']}")
        try:
            modification_request = self.parse_modification_request(state["user_input"])
            if not modification_request:
                raise ValueError("Could not understand modification request.")
            updated_order = self.apply_modifications(state["current_order"], modification_request)
            print("Updated order:", updated_order)
            return {
                **state,
                "current_order": updated_order,
                "order_status": WorkflowState.COLLECT_ITEMS,
                "chat_history": state["chat_history"] + [
                    AIMessage(content="Your order has been updated. Do you want to make any other changes?")
                ]
            }
        except Exception as e:
            return self.handle_error(state, f"Modification error: {str(e)}")

    def parse_modification_request(self, user_input: str) -> dict:
        print(f"Parsing modification request: {user_input}")
        if "add" in user_input.lower():
            item_match = re.search(r'add\s+(\d+)\s+(.+?)\s+\$(\d+\.\d+)', user_input, re.IGNORECASE)
            if item_match:
                return {
                    "action": "add",
                    "item": {
                        "quantity": int(item_match.group(1)),
                        "item": item_match.group(2).strip(),
                        "price": f"${item_match.group(3)}"
                    }
                }
        if "remove" in user_input.lower():
            item_match = re.search(r'remove\s+(.+)', user_input, re.IGNORECASE)
            if item_match:
                return {
                    "action": "remove",
                    "item_name": item_match.group(1).strip()
                }
        return None

    def apply_modifications(self, current_order: dict, modification_request: dict) -> dict:
        print(f"Applying modifications: {modification_request}")
        updated_order = current_order.copy()
        if modification_request["action"] == "add":
            updated_order["menu_items_ordered"].append(modification_request["item"])
        elif modification_request["action"] == "remove":
            updated_order["menu_items_ordered"] = [
                item for item in updated_order["menu_items_ordered"]
                if item["item"].lower() != modification_request["item_name"].lower()
            ]
        total_price = sum(float(item["price"].strip("$")) * item["quantity"] for item in updated_order["menu_items_ordered"])
        updated_order["total_price"] = f"${total_price:.2f}"
        return updated_order

    def sanitize_json(self, response_content: str) -> str:
        try:
            sanitized = response_content.replace("'", "\"")
            json.loads(sanitized)
            return sanitized
        except json.JSONDecodeError as e:
            print(f"JSON sanitization failed: {str(e)}")
            return None

    def parse_agent_response(self, response, user_input: str):
        try:
            content = response.content.strip()
            if content.startswith("{") and content.endswith("}"):
                sanitized_content = self.sanitize_json(content)
                if sanitized_content is None:
                    raise ValueError("LLM response appears to be JSON but could not be sanitized.")
                action_data = json.loads(sanitized_content)
                if "action" in action_data and action_data["action"] in ["add", "remove", "cancel"]:
                    return AgentAction(
                        tool="order_processing",
                        tool_input=action_data,
                        log=response.content
                    )
                raise ValueError("Invalid modification structure in LLM response.")
            return AgentFinish(
                return_values={"output": content},
                log=response.content
            )
        except Exception as e:
            print(f"Error parsing agent response: {str(e)}")
            return AgentFinish(
                return_values={"output": f"Error: {str(e)}"},
                log=str(e)
            )
    
    def update_prompt (self, item_name, prompt_update):
        rule_options = self.processor.indexer.rule_options_col.get(where={"name": item_name})
                    
        if rule_options and "metadatas" in rule_options and rule_options["metadatas"]:
            # Get min/max values from the first option's metadata
            option_meta = rule_options["metadatas"][0]
            min_val = option_meta.get("min", 1)
            max_val = option_meta.get("max", 1)
            # Add selection requirements to the prompt
            prompt_update += f" {item_name}:\n (Select from {min_val} to {max_val})\n"
            return prompt_update
        else:
            return ""
                     

    def getIngredients(self, state: AgentState, origPrompt):
        print(f"[DEBUG] Processing input: {state['input']}")
        processor_output = self.processor.process_order(state["input"])
        
        if not processor_output or not processor_output.get("results"):
            return origPrompt
        
        prompt_update = ""
        results = processor_output.get("results", [])
        category_name = processor_output.get("category", "")
        
        if category_name:
            prompt_update = f"I see you're interested in {category_name}. We have these options:\n\n"
        else:
            prompt_update = "Here are all the matching items:\n\n"
        
        categories = {}
        for result in results:
            result["item"] = result["item"].strip()
            if result.get("type") == "category":
                category_name = result.get("item", "")
                if category_name not in categories:
                    categories[category_name] = []
                
                # Get all items for this category
                category_items = self.processor.indexer.items_col.get(
                    where={"category": category_name}
                )
                
                if category_items and "metadatas" in category_items:
                    for item in category_items["metadatas"]:
                        item["type"] = "item"
                        item["item"] = item["name"]
                        del item["description"]
                        del item["name"]
                        if not item in categories[category_name]:
                            categories[category_name].append(item)
            else:
                category = result.get("category", "Other")
                if category not in categories:
                    categories[category] = []
                if not result in categories[category]:
                    categories[category].append(result)
                
        for category, items in categories.items():
            if category and len(categories) > 1:
                prompt_update += f"[Begin Category] {category}\n"
            
            for item in items:
                item_type = item.get("type", "")
                item_name = item.get("item") or item.get("name", "")
                price = item.get("metadata", {}).get("price", 0)
                base_price = None
                if  item_type == "category" or item_type == "rule_option":
                    continue
                if item_type == "item":
                    offerings = item.get("ingredients", "") or item.get("description", "")
                    base_price = item.get("base_price", price)
                    prompt_update += f"- {item_name}: base price ${base_price:.2f}\n"
                    if offerings:
                        prompt_update += f", offerings: {offerings}\n" 
                    update_prompt = self.update_prompt(item_name,prompt_update)
                    if update_prompt:
                        prompt_update = update_prompt
                elif item_type == "rule_option":
                    update_prompt = self.update_prompt(item_name,prompt_update)
                    if update_prompt:
                        prompt_update = update_prompt
                       
                if item_type == "rule_item":
                    prompt_update += f"- {item_name}: ${price:.2f}\n"
                else:
                    #item_display = f"- {item_name}: ${price:.2f}\n"
                    item_display = ""
                    if 'selected_rules' in item:
                        try:
                            rules = json.loads(item.get('selected_rules', '[]')) if isinstance(item.get('selected_rules'), str) else item.get('selected_rules', [])
                            
                            for rule in rules:
                                update_prompt = self.update_prompt(rule,item_display)
                                if update_prompt:
                                    item_display = update_prompt
                                rule_items = self.processor.indexer.rule_items_col.get(where={"option": rule})
                                if rule_items and "documents" in rule_items:
                                    options_text = []
                                    for i, opt_doc in enumerate(rule_items["documents"]):
                                        if i < len(rule_items["metadatas"]):
                                            opt_price = rule_items["metadatas"][i].get("price", 0)
                                            options_text.append(f"- {opt_doc} (${opt_price:.2f})")
                                    if options_text:
                                        if base_price is not None:
                                            item_display += f",base price ${base_price:.2f}\n Requires selections:\n" +f"\n".join(options_text) + "\n"
                                        else:
                                            item_display += f"Requires selections:\n" +f"\n".join(options_text) + "\n"
                        except json.JSONDecodeError:
                            item_display += "Requires additional selections\n"
                    else:
                        ingredients = item.get("ingredients", "")
                        item_display += f"{ingredients}\n"
                    prompt_update += item_display
                                
            if category and len(categories) > 1:
                prompt_update += f"[End Category]\n"
        
        prompt_update += "\nWhich option would you like to choose?"
        prompt_to_send = origPrompt + "\n" + prompt_update
        print(f" prompt_update: {prompt_update}")
        
        if "calculate_sum(" in prompt_to_send:
            pattern = r'calculate_sum\(\[([\d\., ]+)\]\)'
            matches = re.findall(pattern, prompt_to_send)
            for match in matches:
                numbers = [float(num.strip()) for num in match.split(',')]
                sum_result = sum(numbers)
                addition_str = " + ".join([f"{num:.2f}" for num in numbers])
                replacement = f"{addition_str} = {sum_result:.2f}"
                prompt_to_send = prompt_to_send.replace(f"calculate_sum([{match}])", replacement)
        
        return prompt_to_send


    def getIngredients2(self, state: AgentState, origPrompt):
        print(f"[DEBUG] Processing input: {state['input']}")
        processor_output = self.processor.process_order(state["input"])
        print(f"[DEBUG] Processor output status: {processor_output.get('status')}")
        
        # Handle greeting without database search
        if processor_output.get("status") == "greeting":
            return origPrompt
        
        # Early return if no meaningful data
        if not processor_output or (not processor_output.get("results") and not processor_output.get("item")):
            print("[DEBUG] No relevant information found")
            return origPrompt
        
        # Collect and format all results
        prompt_update = ""
        
        # Get all results and category information
        results = processor_output.get("results", [])
        category_name = processor_output.get("category", "")
        
        # Set appropriate header based on context
        if category_name:
            prompt_update = f"I see you're interested in {category_name}. We have these options:\n\n"
        else:
            prompt_update = "Here are all the matching items:\n\n"
        
        # Group items by category for better organization
        categories = {}
        for item in results:
            category = item.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(item)

        #For type:"item" insert it as it is. For type:"rule_option" bring all items for that option
        #take 'item': 'Crystal Energy Mani', rule_items_col.get(where={"option": item value})
        #[{'item': 'Mani', 'type': 'category', 'ingredients': '', 'price': 0, 'category': 'Mani'}, {'item': 'Gel Removal (hands) ', 'type': 'item', 'ingredients': 'Please include removal service if you currently have gel. Removal is included in our ...lishment, you will be charged for removal.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Gel Removal (hands)"]'}, {'item': 'SNS Removal ', 'type': 'item', 'ingredients': 'Please include removal service if you currently have SNS. Removal is included in our ...lishment, you will be charged for removal.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["SNS Removal"]'}, {'item': 'Acrylic Extensions Removal ', 'type': 'item', 'ingredients': 'Please include removal service if you currently have extensions that need to be removed before any service.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Acrylic Extensions Removal"]'}, {'item': 'Apres Gel-X Extensions Removal ', 'type': 'item', 'ingredients': 'Please include removal service if you currently have extensions that need to be removed before any service.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Apres Gel-X Extensions Removal"]'}, {'item': 'Hard Gel Extensions Removal ', 'type': 'item', 'ingredients': 'Please include removal service if you currently have extensions that need to be removed before any service.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Hard Gel Extensions Removal"]'}, {'item': 'Japanese Gel Removal ', 'type': 'item', 'ingredients': 'Removal of Japanese gel or extension *Removal will be comped if service was originally done by Local Honey', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Japanese Gel Removal"]'}, {'item': 'Chrome Nails ', 'type': 'item', 'ingredients': '', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Chrome Nails"]'}, {'item': 'SNS Removal (free returning client) ', 'type': 'item', 'ingredients': '', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["SNS Removal (free returning client)"]'}, {'item': 'Mani/Pedi', 'type': 'category', 'ingredients': '', 'price': 0, 'category': 'Mani/Pedi'}, {'item': 'Classic Honey Mani ', 'type': 'item', 'ingredients': 'The Local Honey standard mani, that goes above and beyond your average service. Nails...ervice was previously done by Local Honey.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Classic Honey Mani"]'}, {'item': 'Classic Honey Mani', 'type': 'rule_option', 'ingredients': '', 'price': 0, 'category': ''}, {'item': 'Classic Honey Gel Mani ', 'type': 'item', 'ingredients': 'The Local Honey standard mani, that goes above and beyond your average service. Nails...ervice was previously done by Local Honey.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Classic Honey Gel Mani"]'}, {'item': 'Classic Honey Gel Mani', 'type': 'rule_option', 'ingredients': '', 'price': 0, 'category': ''}, {'item': 'Crystal Energy Mani ', 'type': 'item', 'ingredients': 'Treat yourself to an enhanced crystal energy mani. Service includes our classic honey...ervice was previously done by Local Honey.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Crystal Energy Mani"]'}, {'item': 'Crystal Energy Mani', 'type': 'rule_option', 'ingredients': '', 'price': 0, 'category': ''}, {'item': 'Crystal Energy Gel Mani ', 'type': 'item', 'ingredients': 'Treat yourself to an enhanced crystal energy mani. Service includes our classic honey...ervice was previously done by Local Honey.', 'price': 0.0, 'category': 'Mani', 'base_price': 0.0, 'selected_rules': '["Crystal Energy Gel Mani"]'}, {'item': 'Crystal Energy Gel Mani', 'type': 'rule_option', 'ingredients': '', 'price': 0, 'category': ''}]

        # Loop through all categories and include all items
        for category, items in categories.items():
            if category and len(categories) > 1:
                prompt_update += f"[Begin Category] {category}\n"

            ##Insert here
            for item in items:
                item_name = item.get("item", "")
                base_price = item.get("base_price", item.get("price", 0))
                
                # Start building the item display with consistent pricing format
                item_display = f"- {item_name}: ${base_price:.2f}\n"
                
                # Check if this item has rules
                if 'selected_rules' in item:
                    try:
                        # Parse selected rules (handling both string and list formats)
                        rules = json.loads(item.get('selected_rules', '[]')) if isinstance(item.get('selected_rules'), str) else item.get('selected_rules', [])
                        
                        # Add "Requires selections:" label
                        item_display += "Requires selections:\n"
                        
                        # Get options for this item
                        item_name = item_name.strip()
                        rule_items = self.processor.indexer.rule_items_col.get(where={"item": item_name})

                        if rule_items and "documents" in rule_items:
                            options_text = []
                            for i, opt_doc in enumerate(rule_items["documents"]):
                                opt_price = rule_items["metadatas"][i].get("price", 0) if i < len(rule_items["metadatas"]) else 0
                                option_meta = rule_items["metadatas"][i]
                                min_val = option_meta.get("min", 1)
                                max_val = option_meta.get("max", -1) #"unlimited")
                                
                                if max_val != -1:
                                    if min_val == max_val:
                                        req_text = f"exactly {min_val}"
                                    elif min_val == 0:
                                        req_text = f"up to {max_val}"
                                    else:
                                        req_text = f"{min_val} to {max_val}"
                                    options_text.append(f"- {opt_doc} (${opt_price:.2f}) (Select {req_text})")
                                else:
                                    options_text.append(f"- {opt_doc} (${opt_price:.2f})")
                            
                            if options_text:
                                item_display += "\n".join(options_text) + "\n"
                    except json.JSONDecodeError:
                        item_display += "Requires additional selections\n"
                else:
                    # Regular item - just show ingredients
                    ingredients = item.get("ingredients", "")
                    item_display += f"{ingredients}\n"
                
                # Add the formatted item to the prompt
                prompt_update += item_display


            # Insert  
            if category and len(categories) > 1:
                prompt_update += f"[End Category]\n"
        
        # Add call-to-action
        prompt_update += "\nWhich option would you like to choose?"
        
        # Process any calculate_sum function calls in the prompt
        prompt_to_send = origPrompt + "\n" + prompt_update
        
        if "calculate_sum(" in prompt_to_send:
            pattern = r'calculate_sum\(\[([\d\., ]+)\]\)'
            matches = re.findall(pattern, prompt_to_send)
            for match in matches:
                numbers = [float(num.strip()) for num in match.split(',')]
                sum_result = sum(numbers)
                
                # Create a formatted string showing the addition
                addition_str = " + ".join([f"{num:.2f}" for num in numbers])
                replacement = f"{addition_str} = {sum_result:.2f}"
                
                prompt_to_send = prompt_to_send.replace(f"calculate_sum([{match}])", replacement)
        
        return prompt_to_send


    
    def getIngredients2(self, state: AgentState, origPrompt):
        print(f"[DEBUG] Processing input: {state['input']}")
        processor_output = self.processor.process_order(state["input"])
        print(f"[DEBUG] Processor output status: {processor_output.get('status')}")
        
        # Handle greeting without database search
        if processor_output.get("status") == "greeting":
            return origPrompt  # Return original prompt which contains "What would you like to order?"
    
        # Case 1: Check if input exactly matches a category name
        if processor_output.get("status") == "need_input" and "results" in processor_output:
            similar_items = processor_output.get("results", [])
            print(f"[DEBUG] Found {len(similar_items)} similar items")
            
            # Group items by category
            categories = {}
            for item in similar_items:
                category = item.get("category", "")
                if category not in categories:
                    categories[category] = []
                categories[category].append(item)
            
            print(f"[DEBUG] Grouped items into {len(categories)} categories")
            
            # Create a prompt that includes all items from all categories
            prompt_update = "Here are all the matching items:\n\n"
            
            # Loop through all categories and include all items
            for category, items in categories.items():
                if category:  # Only add category header for non-empty categories
                    prompt_update += f"[Begin Category] {category}\n"
                
                for item in items:
                    item_name = item.get("item", "")
                    base_price = item.get("base_price", item.get("price", 0))
                    print(f"[DEBUG] Adding item {item_name} with base price ${base_price}")
                    
                    if 'selected_rules' in item:
                        try:
                            rules = json.loads(item.get('selected_rules', '[]')) if isinstance(item.get('selected_rules'), str) else item.get('selected_rules', [])
                            prompt_update += f"- {item_name}: ${base_price:.2f}\n  Requires selections for: {', '.join(rules)}\n"
                        except json.JSONDecodeError:
                            prompt_update += f"- {item_name}: ${base_price:.2f}\n  Requires additional selections\n"
                    else:
                        ingredients = item.get("ingredients", "")
                        prompt_update += f"- {item_name}: ${base_price:.2f}\n  {ingredients}\n"
                
                if category:
                    prompt_update += f"[End Category]\n"
            
            prompt_update += "\nWhich option would you like to choose?"
            print(f"[DEBUG] Created all items prompt")
            return origPrompt + "\n" + prompt_update


        # Case 2: Rule-based item requiring selections
        if processor_output.get("status") == "need_rule_selections":
            # Check if we have multiple rule-based items or a single item with options
            if "results" in processor_output and processor_output.get("results"):
                # Multiple items with rules
                rule_items = [item for item in processor_output.get("results", []) 
                              if "selected_rules" in item]
                
                if rule_items:
                    prompt_update = "I found these items that require additional selections:\n\n"
                    
                    for item in rule_items[:5]:  # Show up to 5 matches
                        item_name = item.get("item", "")
                        base_price = item.get("base_price", item.get("price", 0))
                        
                        # Parse rules
                        selected_rules = item.get('selected_rules', [])
                        rules = []
                        
                        if isinstance(selected_rules, str):
                            try:
                                rules = json.loads(selected_rules)
                            except json.JSONDecodeError:
                                rules = []
                        else:
                            rules = selected_rules
                        
                        prompt_update += f"- {item_name} (${base_price:.2f}) - Requires selections for: {', '.join(rules)}\n"
                    
                    prompt_update += "\nWhich item would you like to select?"
                    return origPrompt + "\n" + prompt_update
            
            # Single rule-based item with detailed options
            if "item" in processor_output:
                item_name = processor_output.get("item", "")
                base_price = processor_output.get("base_price", 0)
                rules = processor_output.get("rules", [])
                
                prompt_update = f"I see you're interested in our {item_name}! It starts at ${base_price:.2f} and you'll need to select:\n"
                
                # Format available options for each rule
                for rule_name, options in processor_output.get("available_options", {}).items():
                    if not options:
                        continue
                    
                    min_val = options[0].get("min", 0)
                    max_val = options[0].get("max", "unlimited")
                    
                    prompt_update += f"\n{rule_name} (Select "
                    if min_val == max_val:
                        prompt_update += f"exactly {min_val}"
                    elif min_val == 0:
                        prompt_update += f"up to {max_val}"
                    else:
                        prompt_update += f"{min_val} to {max_val}"
                    prompt_update += f"):\n"
                    
                    # Get all items for this rule option
                    all_items = []
                    for option in options:
                        for item in option.get("items", []):
                            all_items.append(f"{item.get('name', '').split(' (')[0]} (${item.get('price', 0):.2f})")
                    
                    if all_items:
                        prompt_update += f" Options include: {', '.join(all_items)}\n"
                
                prompt_update += f"\nWhat would you like to select for your {item_name}?"
                return origPrompt + "\n" + prompt_update
        
        # Case 3: Exact match found (standard item)
        elif processor_output.get("status") == "success":
            found_item = processor_output.get("item", "")
            ingredients = processor_output.get("ingredients", "No ingredients available")
            print(f"[DEBUG] Found exact match: {found_item}")
            prompt_update = f"Found item: {found_item}\nIngredients: {ingredients}\n"
            prompt_to_send = origPrompt + "\n" + prompt_update
        
        # Add this case at the beginning of getIngredients function
        elif processor_output.get("status") == "need_category_selection":
            category_name = processor_output.get("category", "")
            items = processor_output.get("results", [])
            print(f"[DEBUG] Handling category selection for {category_name} with {len(items)} items")
            
            prompt_update = f"I see you're interested in {category_name}. We have these options:\n\n"
            
            for item in items:
                item_name = item.get("item", "")
                base_price = item.get("base_price", item.get("price", 0))
                print(f"[DEBUG] Adding item {item_name} with base price ${base_price}")
                
                if 'selected_rules' in item:
                    #rules = json.loads(item.get('selected_rules', '[]'))
                    #prompt_update += f"- {item_name} (${base_price:.2f}) - Requires selections for: {', '.join(rules)}\n"
                    selected_rules = item.get('selected_rules', [])
                    if isinstance(selected_rules, str):
                        try:
                            rules = json.loads(selected_rules)
                        except json.JSONDecodeError:
                            print(f"[DEBUG] Error parsing selected_rules for {item_name}")
                            rules = []
                    else:
                            rules = selected_rules
                    prompt_update += f"- {item_name} (${base_price:.2f}) - Requires selections for: {', '.join(rules)}\n"
                        
                else:
                    ingredients = item.get("ingredients", "")
                    prompt_update += f"- {item_name} (${base_price:.2f}): {ingredients}\n"
            
            prompt_update += "\nWhich option would you like to choose?"
            print(f"[DEBUG] Created category options prompt")
            return origPrompt + "\n" + prompt_update



        # Default case: No relevant information found
        else:
            print("[DEBUG] No relevant information found")
            prompt_to_send = origPrompt
        
        # Process any calculate_sum function calls in the prompt
        if "calculate_sum(" in prompt_to_send:
            pattern = r'calculate_sum\(\[([\d\., ]+)\]\)'
            matches = re.findall(pattern, prompt_to_send)
            for match in matches:
                numbers = [float(num.strip()) for num in match.split(',')]
                sum_result = sum(numbers)
                
                # Create a formatted string showing the addition
                addition_str = " + ".join([f"{num:.2f}" for num in numbers])
                replacement = f"{addition_str} = {sum_result:.2f}"
                
                print(f"[DEBUG] Replaced calculate_sum with formatted addition: {replacement}")
                prompt_to_send = prompt_to_send.replace(f"calculate_sum([{match}])", replacement)
        
        return prompt_to_send


    def getIngredients2 (self,state:AgentState,origPrompt):
        processor_output = self.processor.process_order(state["input"])
        # Check if the order processor was able to extract an item successfully.
        if processor_output.get("status") == "success":
            found_item = processor_output.get("item", "")
            ingredients = processor_output.get("ingredients", "No ingredients available")
            # Append the found info to the prompt. You can customize the text as needed.
            prompt_update = f"Found item: {found_item}\nIngredients: {ingredients}\n"
            print(f"[orderChat] Updating prompt with:\n{prompt_update}")
            prompt_to_send = origPrompt + "\n" + prompt_update
        else:
            # In case processor did not return a successful result, leave prompt unchanged.
            print("[orderChat] Order processor did not extract item successfully; using default prompt.")
            prompt_to_send = origPrompt
        return prompt_to_send

    def call_agent(self, state: AgentState):
        print("\n--- AGENT PROCESSING ---")
        #selection_prompt = self.get_prompt_for_selection(state["input"])
        #if selection_prompt:
        #    prompt_to_send = selection_prompt
        #else:
        #    prompt_to_send = self.prompt_text
        prompt_to_send = self.getIngredients(state,self.prompt_text)
        messages = [
            SystemMessage(content=prompt_to_send),
            *state["chat_history"],
            HumanMessage(content=state["input"])
        ]
        try:
            print(f"Sending prompt to LLM:\n{prompt_to_send}")
            response = self.llm_local.invoke(messages)
            print(f"LLM raw response: {response.content}")
            action = self.parse_agent_response(response, state["input"])
            if isinstance(action, AgentFinish):
                print("Returning AgentFinish immediately")
                return {
                    "input": state["input"],
                    "chat_history": state["chat_history"],
                    "intermediate_steps": [action],
                    "tools": state["tools"]
                }
            print(f"Returning AgentAction: {action.tool}")
            return {
                "input": state["input"],
                "chat_history": state["chat_history"],
                "intermediate_steps": [action],
                "tools": state["tools"]
            }
        except Exception as e:
            print(f"Error in call_agent: {str(e)}")
            return {
                "input": state["input"],
                "chat_history": state["chat_history"],
                "intermediate_steps": [
                    AgentFinish(
                        return_values={"output": f"Error processing request: {str(e)}"},
                        log=str(e)
                    )
                ],
                "tools": state["tools"]
            }

    def truncateMessage (self, message):
        # Use a regex pattern to match words along with following spaces.
        pattern = re.compile(r'\S+\s*')
        truncated = ""
        max_chars = 160
        for match in pattern.finditer(message):
            token = match.group()
            # Check if adding the next token would exceed the limit.
            if len(truncated) + len(token) > max_chars:
                break
            truncated += token

        # Remove any trailing whitespace
        return truncated.strip()


    def chatAway(self, user_input: str) -> str:
        print(f"\n=== NEW CHAT REQUEST ===\nInput: {user_input}")
        try:
            state = {
                "input": user_input,
                "user_input": user_input,
                "chat_history": self.state["chat_history"].copy(),
                "order_status": self.state["order_status"],
                "current_order": self.state["current_order"].copy(),
                "error_count": self.state.get("error_count", 0),
                "tools": []
            }
            print("Initial state:", state)
            agent_response = self.call_agent(state)
            if isinstance(agent_response, dict) and "intermediate_steps" in agent_response:
                for action in agent_response["intermediate_steps"]:
                    if isinstance(action, AgentFinish):
                        self.state.update({
                            "chat_history": state["chat_history"] + [
                                HumanMessage(content=user_input),
                                AIMessage(content=action.return_values["output"])
                            ],
                            "order_status": agent_response.get("order_status", self.state["order_status"]),
                            "current_order": agent_response.get("current_order", self.state["current_order"]),
                            "error_count": state.get("error_count", 0)
                        })
                        return action.return_values["output"]
                        #self.truncateMessage(action.return_values["output"])
            return "Order processing failed to complete"
        except Exception as e:
            print(f"Workflow error: {str(e)}")
            return "An error occurred. Please start over."

    def process_message(self, user_input: str) -> str:
        return self.chatAway(user_input)

    def calculate_order_price(self,order_text):
        # 1. Find relevant category
        categories = self.indexer.categories_col.query(
            query_texts=[order_text],
            n_results=1
        )
        category = categories['metadatas'][0][0]
        
        # 2. Extract ingredients
        ingredients = self.indexer.ingredients_col.query(
            query_texts=[order_text],
            where={"category": category['name']},
            n_results=20
        )
        
        # 3. Apply rules and calculate price
        total = category['base_price']
        selections = defaultdict(list)
        
        for ing in ingredients['metadatas'][0]:
            rule_type = ing['rule_type']
            if len(selections[rule_type]) < ing['max_select']:
                total += ing['price']
                selections[rule_type].append(ing['price'])
        
        return round(total, 2)


if __name__ == "__main__":
    system = orderChat("+19175587915")
    print("Ordering system ready. Type 'exit' to quit.")
    while True:
        user_input = input("Customer: ")
        if user_input.lower() == "exit":
            break
        response = system.chatAway(user_input)
        print("System:", response)
