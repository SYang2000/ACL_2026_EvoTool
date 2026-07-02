"""Self-contained vendored tau-bench (retail) tool invoke-logic.

Vendored verbatim (invoke logic only) from sierra-research/tau-bench
`tau_bench/envs/retail/tools/*.py`. Each upstream tool is a class with a
``staticmethod invoke(data, **kwargs) -> str``; here we keep ONLY the pure-dict
invoke logic, stripped of the ``tau_bench.envs.tool.Tool`` import.

Exposes ``TOOLS = {tool_name: invoke_callable}`` where every callable has the
signature ``f(data: dict, **kwargs) -> str`` and either mutates ``data`` in
place (write tools) or returns a string without mutating (read tools).

Read-only tools  (no DB mutation):
    calculate, find_user_id_by_email, find_user_id_by_name_zip,
    get_order_details, get_product_details, get_user_details,
    list_all_product_types, think, transfer_to_human_agents
Write tools (mutate db["orders"] and/or db["users"]):
    cancel_pending_order, modify_pending_order_address,
    modify_pending_order_items, modify_pending_order_payment,
    modify_user_address, return_delivered_order_items,
    exchange_delivered_order_items
"""

import json
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# Read-only tools
# --------------------------------------------------------------------------- #

def calculate(data: Dict[str, Any], expression: str) -> str:
    if not all(char in "0123456789+-*/(). " for char in expression):
        return "Error: invalid characters in expression"
    try:
        # Evaluate the mathematical expression safely
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
    except Exception as e:
        return f"Error: {e}"


def find_user_id_by_email(data: Dict[str, Any], email: str) -> str:
    users = data["users"]
    for user_id, profile in users.items():
        if profile["email"].lower() == email.lower():
            return user_id
    return "Error: user not found"


def find_user_id_by_name_zip(
    data: Dict[str, Any], first_name: str, last_name: str, zip: str
) -> str:
    users = data["users"]
    for user_id, profile in users.items():
        if (
            profile["name"]["first_name"].lower() == first_name.lower()
            and profile["name"]["last_name"].lower() == last_name.lower()
            and profile["address"]["zip"] == zip
        ):
            return user_id
    return "Error: user not found"


def get_order_details(data: Dict[str, Any], order_id: str) -> str:
    orders = data["orders"]
    if order_id in orders:
        return json.dumps(orders[order_id])
    return "Error: order not found"


def get_product_details(data: Dict[str, Any], product_id: str) -> str:
    products = data["products"]
    if product_id in products:
        return json.dumps(products[product_id])
    return "Error: product not found"


def get_user_details(data: Dict[str, Any], user_id: str) -> str:
    users = data["users"]
    if user_id in users:
        return json.dumps(users[user_id])
    return "Error: user not found"


def list_all_product_types(data: Dict[str, Any]) -> str:
    products = data["products"]
    product_dict = {
        product["name"]: product["product_id"] for product in products.values()
    }
    product_dict = dict(sorted(product_dict.items()))
    return json.dumps(product_dict)


def think(data: Dict[str, Any], thought: str) -> str:
    # Does not change the state of the data; simply returns an empty string.
    return ""


def transfer_to_human_agents(data: Dict[str, Any], summary: str) -> str:
    # Simulates the transfer to a human agent.
    return "Transfer successful"


# --------------------------------------------------------------------------- #
# Write tools (mutate the DB in place)
# --------------------------------------------------------------------------- #

def cancel_pending_order(data: Dict[str, Any], order_id: str, reason: str) -> str:
    # check order exists and is pending
    orders = data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be cancelled"

    # check reason
    if reason not in ["no longer needed", "ordered by mistake"]:
        return "Error: invalid reason"

    # handle refund
    refunds = []
    for payment in order["payment_history"]:
        payment_id = payment["payment_method_id"]
        refund = {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment_id,
        }
        refunds.append(refund)
        if "gift_card" in payment_id:  # refund to gift card immediately
            payment_method = data["users"][order["user_id"]]["payment_methods"][
                payment_id
            ]
            payment_method["balance"] += payment["amount"]
            payment_method["balance"] = round(payment_method["balance"], 2)

    # update order status
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["payment_history"].extend(refunds)

    return json.dumps(order)


def exchange_delivered_order_items(
    data: Dict[str, Any],
    order_id: str,
    item_ids: List[str],
    new_item_ids: List[str],
    payment_method_id: str,
) -> str:
    products, orders, users = data["products"], data["orders"], data["users"]

    # check order exists and is delivered
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "delivered":
        return "Error: non-delivered order cannot be exchanged"

    # check the items to be exchanged exist
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return f"Error: {item_id} not found"

    # check new items exist and match old items and are available
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"

    diff_price = 0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        product_id = item["product_id"]
        if not (
            new_item_id in products[product_id]["variants"]
            and products[product_id]["variants"][new_item_id]["available"]
        ):
            return f"Error: new item {new_item_id} not found or available"

        old_price = item["price"]
        new_price = products[product_id]["variants"][new_item_id]["price"]
        diff_price += new_price - old_price

    diff_price = round(diff_price, 2)

    # check payment method exists and can cover the price difference if gift card
    if payment_method_id not in users[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < diff_price
    ):
        return "Error: insufficient gift card balance to pay for the price difference"

    # modify the order
    order["status"] = "exchange requested"
    order["exchange_items"] = sorted(item_ids)
    order["exchange_new_items"] = sorted(new_item_ids)
    order["exchange_payment_method_id"] = payment_method_id
    order["exchange_price_difference"] = diff_price

    return json.dumps(order)


def modify_pending_order_address(
    data: Dict[str, Any],
    order_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> str:
    # Check if the order exists and is pending
    orders = data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Modify the address
    order["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    return json.dumps(order)


def modify_pending_order_items(
    data: Dict[str, Any],
    order_id: str,
    item_ids: List[str],
    new_item_ids: List[str],
    payment_method_id: str,
) -> str:
    products, orders, users = data["products"], data["orders"], data["users"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Check if the items to be modified exist
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return f"Error: {item_id} not found"

    # Check new items exist, match old items, and are available
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"

    diff_price = 0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        product_id = item["product_id"]
        if not (
            new_item_id in products[product_id]["variants"]
            and products[product_id]["variants"][new_item_id]["available"]
        ):
            return f"Error: new item {new_item_id} not found or available"

        old_price = item["price"]
        new_price = products[product_id]["variants"][new_item_id]["price"]
        diff_price += new_price - old_price

    # Check if the payment method exists
    if payment_method_id not in users[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    # If the new item is more expensive, check if the gift card has enough balance
    payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < diff_price
    ):
        return "Error: insufficient gift card balance to pay for the new item"

    # Handle the payment or refund
    order["payment_history"].append(
        {
            "transaction_type": "payment" if diff_price > 0 else "refund",
            "amount": abs(diff_price),
            "payment_method_id": payment_method_id,
        }
    )
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= diff_price
        payment_method["balance"] = round(payment_method["balance"], 2)

    # Modify the order
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        item["item_id"] = new_item_id
        item["price"] = products[item["product_id"]]["variants"][new_item_id]["price"]
        item["options"] = products[item["product_id"]]["variants"][new_item_id][
            "options"
        ]
    order["status"] = "pending (item modified)"

    return json.dumps(order)


def modify_pending_order_payment(
    data: Dict[str, Any],
    order_id: str,
    payment_method_id: str,
) -> str:
    orders = data["orders"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Check if the payment method exists
    if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    # Check that the payment history should only have one payment
    if (
        len(order["payment_history"]) > 1
        or order["payment_history"][0]["transaction_type"] != "payment"
    ):
        return "Error: there should be exactly one payment for a pending order"

    # Check that the payment method is different
    if order["payment_history"][0]["payment_method_id"] == payment_method_id:
        return "Error: the new payment method should be different from the current one"

    amount = order["payment_history"][0]["amount"]
    payment_method = data["users"][order["user_id"]]["payment_methods"][
        payment_method_id
    ]

    # Check if the new payment method has enough balance if it is a gift card
    if payment_method["source"] == "gift_card" and payment_method["balance"] < amount:
        return "Error: insufficient gift card balance to pay for the order"

    # Modify the payment method
    order["payment_history"].extend(
        [
            {
                "transaction_type": "payment",
                "amount": amount,
                "payment_method_id": payment_method_id,
            },
            {
                "transaction_type": "refund",
                "amount": amount,
                "payment_method_id": order["payment_history"][0]["payment_method_id"],
            },
        ]
    )

    # If payment is made by gift card, update the balance
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= amount
        payment_method["balance"] = round(payment_method["balance"], 2)

    # If refund is made to a gift card, update the balance
    if "gift_card" in order["payment_history"][0]["payment_method_id"]:
        old_payment_method = data["users"][order["user_id"]]["payment_methods"][
            order["payment_history"][0]["payment_method_id"]
        ]
        old_payment_method["balance"] += amount
        old_payment_method["balance"] = round(old_payment_method["balance"], 2)

    return json.dumps(order)


def modify_user_address(
    data: Dict[str, Any],
    user_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> str:
    users = data["users"]
    if user_id not in users:
        return "Error: user not found"
    user = users[user_id]
    user["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    return json.dumps(user)


def return_delivered_order_items(
    data: Dict[str, Any], order_id: str, item_ids: List[str], payment_method_id: str
) -> str:
    orders = data["orders"]

    # Check if the order exists and is delivered
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "delivered":
        return "Error: non-delivered order cannot be returned"

    # Check if the payment method exists and is either the original payment method
    # or a gift card
    if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"
    if (
        "gift_card" not in payment_method_id
        and payment_method_id != order["payment_history"][0]["payment_method_id"]
    ):
        return (
            "Error: payment method should be either the original payment "
            "method or a gift card"
        )

    # Check if the items to be returned exist (there could be duplicate items)
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return "Error: some item not found"

    # Update the order status
    order["status"] = "return requested"
    order["return_items"] = sorted(item_ids)
    order["return_payment_method_id"] = payment_method_id

    return json.dumps(order)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

TOOLS = {
    # read-only
    "calculate": calculate,
    "find_user_id_by_email": find_user_id_by_email,
    "find_user_id_by_name_zip": find_user_id_by_name_zip,
    "get_order_details": get_order_details,
    "get_product_details": get_product_details,
    "get_user_details": get_user_details,
    "list_all_product_types": list_all_product_types,
    "think": think,
    "transfer_to_human_agents": transfer_to_human_agents,
    # write
    "cancel_pending_order": cancel_pending_order,
    "exchange_delivered_order_items": exchange_delivered_order_items,
    "modify_pending_order_address": modify_pending_order_address,
    "modify_pending_order_items": modify_pending_order_items,
    "modify_pending_order_payment": modify_pending_order_payment,
    "modify_user_address": modify_user_address,
    "return_delivered_order_items": return_delivered_order_items,
}

# Set of tools that mutate the DB (for reference / introspection).
WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)
