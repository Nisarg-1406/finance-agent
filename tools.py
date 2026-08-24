"""
tools.py — Mock Tools + Production-Grade Extensions

Original 4 stubs: UNTOUCHED (function signatures, data, docstrings preserved).
New tools added below the original TOOLS registry.

Production tool definition pattern applied to new tools (from guide §4):
  - Active-verb name
  - Explicit use_when / do_not_use_when
  - Exact arg types and formats
  - Output format description
  - Known limitations / edge cases

Separation principle: New tools are PURE PYTHON (no LLM, no I/O).
  They receive data as parameters — the agent calls the original tools to
  fetch live data, then calls these to compute analytics. This matches the
  project constraint: "LLM calls only where judgment is needed."

Session context manager: set_session_context(n) temporarily overrides
  CURRENT_SESSION for multi-user simulation. Thread-unsafe for prod
  (use request-scoped state in production).
"""

from contextlib import contextmanager
import math

# ─── Original session flag (DO NOT MODIFY) ───────────────────────────────────

# Session 1 = Monday, Nov 3, 2025 (salary just credited that morning)
# Session 2 = Thursday, Nov 6, 2025 (rent has been paid, a few more food orders)
CURRENT_SESSION = 1  # flip to 2 before running session 2


@contextmanager
def set_session_context(session_num: int):
    """
    Context manager to temporarily override CURRENT_SESSION.
    Use in multi-user simulation to avoid mutating global state permanently.

    Production note: replace with request-scoped session state (FastAPI dependency
    injection, or a contextvars.ContextVar per async task).
    """
    global CURRENT_SESSION
    old = CURRENT_SESSION
    CURRENT_SESSION = session_num
    try:
        yield
    finally:
        CURRENT_SESSION = old


# ─── Original Tool Stubs (DO NOT MODIFY) ─────────────────────────────────────

def get_recent_transactions(days: int) -> list[dict]:
    """
    Transactions from the last N days, relative to 'today' for the current session.
    Negative amount = debit (money out). Positive = credit (money in). INR.
    Filtering by `days` is left to the caller.
    """
    txns = [
        {"date": "2025-10-01", "amount": -25000, "category": "rent",          "merchant": "Landlord"},
        {"date": "2025-10-03", "amount": -1200,  "category": "food_delivery", "merchant": "Swiggy"},
        {"date": "2025-10-04", "amount": -1800,  "category": "food_delivery", "merchant": "Swiggy"},
        {"date": "2025-10-07", "amount": -4500,  "category": "shopping",      "merchant": "Myntra"},
        {"date": "2025-10-08", "amount": -1100,  "category": "food_delivery", "merchant": "Zomato"},
        {"date": "2025-10-10", "amount": -10000, "category": "investment",    "merchant": "MF SIP"},
        {"date": "2025-10-11", "amount": -950,   "category": "food_delivery", "merchant": "Swiggy"},
        {"date": "2025-10-13", "amount": -3200,  "category": "groceries",     "merchant": "BigBasket"},
        {"date": "2025-10-15", "amount": -1500,  "category": "entertainment", "merchant": "BookMyShow"},
        {"date": "2025-10-17", "amount": -2200,  "category": "food_delivery", "merchant": "Swiggy"},
        {"date": "2025-10-20", "amount": -890,   "category": "food_delivery", "merchant": "Zomato"},
        {"date": "2025-10-22", "amount": -2200,  "category": "fuel",          "merchant": "IOCL"},
        {"date": "2025-10-24", "amount": -1500,  "category": "food_delivery", "merchant": "Swiggy"},
        {"date": "2025-10-27", "amount": -1200,  "category": "food_delivery", "merchant": "Zomato"},
        {"date": "2025-10-28", "amount": -3500,  "category": "shopping",      "merchant": "Amazon"},
        {"date": "2025-10-30", "amount": -1400,  "category": "food_delivery", "merchant": "Swiggy"},
        {"date": "2025-11-01", "amount": 120000, "category": "salary",        "merchant": "Employer"},
        {"date": "2025-11-02", "amount": -650,   "category": "food_delivery", "merchant": "Swiggy"},
    ]
    if CURRENT_SESSION == 2:
        txns += [
            {"date": "2025-11-03", "amount": -1100,  "category": "food_delivery", "merchant": "Zomato"},
            {"date": "2025-11-04", "amount": -780,   "category": "food_delivery", "merchant": "Swiggy"},
            {"date": "2025-11-05", "amount": -25000, "category": "rent",          "merchant": "Landlord"},
            {"date": "2025-11-06", "amount": -1300,  "category": "food_delivery", "merchant": "Swiggy"},
        ]
    return txns


def get_account_balance() -> dict:
    """Current balances across the user's accounts (INR)."""
    if CURRENT_SESSION == 1:
        # Nov 3: Salary just credited
        return {
            "checking":      128000,
            "savings":       145000,
            "house_fund":     95000,
            "mutual_funds":  280000,
        }
    elif CURRENT_SESSION == 2:
        # Nov 6: Rent paid (Nov 5), some food orders
        return {
            "checking":       99820,
            "savings":       145000,
            "house_fund":     95000,
            "mutual_funds":  280000,
        }
    elif CURRENT_SESSION == 3:
        # Nov 11: Rent + SIP paid
        return {
            "checking":       89820,  # 99820 - 10000 (SIP on Nov 10)
            "savings":       145000,
            "house_fund":     95000,
            "mutual_funds":  290000,  # 280000 + 10000 (SIP contribution)
        }
    elif CURRENT_SESSION == 4:
        # Nov 16: Rent + SIP + Internet paid
        return {
            "checking":       86320,  # 89820 - 3500 (Internet on Nov 15)
            "savings":       145000,
            "house_fund":     95000,
            "mutual_funds":  290000,
        }
    elif CURRENT_SESSION == 5:
        # All November bills paid + house fund transfer + misc spending
        # Starting: 128000 (Nov 1 salary)
        # - 25000 (rent Nov 5) - 10000 (SIP Nov 10) - 3500 (Internet Nov 15)
        # - 8000 (CC Nov 20) - 30000 (house fund Nov 25)
        # = ~51,500
        return {
            "checking":       51500,
            "savings":       145000,
            "house_fund":    125000,  # 95000 + 30000 (transfer on Nov 25)
            "mutual_funds":  290000,  # 280000 + 10000 (SIP contribution)
        }


def get_upcoming_bills(days: int = 30) -> list[dict]:
    """Scheduled bills/payments due in the next N days from 'today' for the current session."""
    if CURRENT_SESSION == 1:
        return [
            {"date": "2025-11-05", "amount": 25000, "description": "Rent (auto-debit)"},
            {"date": "2025-11-10", "amount": 10000, "description": "SIP - Mutual Funds"},
            {"date": "2025-11-15", "amount": 3500,  "description": "Internet + Mobile"},
            {"date": "2025-11-20", "amount": 8000,  "description": "Credit Card (auto-debit)"},
        ]
    elif CURRENT_SESSION in [2, 3]:
        # Rent has been paid; the rest remain
        return [
            {"date": "2025-11-10", "amount": 10000, "description": "SIP - Mutual Funds"},
            {"date": "2025-11-15", "amount": 3500,  "description": "Internet + Mobile"},
            {"date": "2025-11-20", "amount": 8000,  "description": "Credit Card (auto-debit)"},
        ]
    elif CURRENT_SESSION == 4:
        # SIP paid on Nov 10; Internet and Credit Card remain
        return [
            {"date": "2025-11-15", "amount": 3500,  "description": "Internet + Mobile"},
            {"date": "2025-11-20", "amount": 8000,  "description": "Credit Card (auto-debit)"},
        ]
    else:  # Session 5 (Nov 28) - all November bills paid, show December bills
        return [
            {"date": "2025-12-05", "amount": 25000, "description": "Rent (auto-debit)"},
            {"date": "2025-12-10", "amount": 10000, "description": "SIP - Mutual Funds"},
            {"date": "2025-12-15", "amount": 3500,  "description": "Internet + Mobile"},
            {"date": "2025-12-20", "amount": 8000,  "description": "Credit Card (auto-debit)"},
        ]


def set_reminder(date: str, content: str) -> dict:
    """
    Log a reminder. `date` is 'YYYY-MM-DD'.
    Returns a confirmation dict with a reminder_id.
    """
    return {
        "status": "set",
        "reminder_id": f"rem_{abs(hash(content)) % 10000}",
        "date": date,
        "content": content,
    }


# ─── Original Tool Registry (preserved) ──────────────────────────────────────

TOOLS = {
    "get_recent_transactions": get_recent_transactions,
    "get_account_balance":     get_account_balance,
    "get_upcoming_bills":      get_upcoming_bills,
    "set_reminder":            set_reminder,
}


# ─── New Production-Grade Tools ────────────────

def calculate_current_balance_from_transactions(
    initial_balance: dict,
    transactions: list[dict],
    cutoff_date: str,
    pending_transfers: list[dict] | None = None,
) -> dict:
    """
    Calculate current account balances by applying transactions to initial balances.
    
    Use when: Agent needs to determine current balance after specific transactions
              have been processed (e.g., after rent payment, SIP deduction).
              Agent should check memory for any pending transfers/reminders.
    Do NOT use for: Getting initial/base balances (use get_account_balance).
    
    Args:
        initial_balance (dict): Starting balances from get_account_balance()
        transactions (list[dict]): Transactions from get_recent_transactions()
        cutoff_date (str): Calculate balance as of this date ('YYYY-MM-DD')
        pending_transfers (list[dict], optional): List of scheduled transfers from memory
            Format: [{"date": "YYYY-MM-DD", "amount": 30000, "from": "checking", "to": "house_fund"}]
    
    Returns:
        dict with updated balances for each account based on transactions and transfers
    
    Logic:
        - Salary credits go to checking
        - Rent, food, shopping, utilities debit from checking
        - SIP debits from checking, credits to mutual_funds
        - House fund transfers debit checking, credit house_fund
        - Pending transfers from memory are applied if date <= cutoff_date
    
    Limitation: Assumes all transactions affect checking unless specified.
    """
    from datetime import datetime
    
    # Copy initial balances
    balances = initial_balance.copy()
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
    
    # Apply each transaction up to cutoff date
    for txn in transactions:
        txn_date = datetime.strptime(txn["date"], "%Y-%m-%d")
        if txn_date > cutoff:
            continue
            
        amount = txn["amount"]
        category = txn.get("category", "")
        
        # Apply transaction to appropriate accounts
        if category == "salary":
            balances["checking"] = balances.get("checking", 0) + amount
        elif category == "investment":
            # SIP: debit checking, credit mutual_funds
            balances["checking"] = balances.get("checking", 0) + amount  # amount is negative
            balances["mutual_funds"] = balances.get("mutual_funds", 0) + abs(amount)
        elif category == "house_fund_transfer":
            # Transfer: debit checking, credit house_fund
            balances["checking"] = balances.get("checking", 0) + amount  # amount is negative
            balances["house_fund"] = balances.get("house_fund", 0) + abs(amount)
        else:
            # All other transactions affect checking
            balances["checking"] = balances.get("checking", 0) + amount
    
    # Apply pending transfers from memory if they've occurred by cutoff date
    if pending_transfers:
        for transfer in pending_transfers:
            transfer_date = datetime.strptime(transfer["date"], "%Y-%m-%d")
            if transfer_date <= cutoff:
                amount = transfer["amount"]
                from_account = transfer.get("from", "checking")
                to_account = transfer.get("to", "house_fund")
                
                balances[from_account] = balances.get(from_account, 0) - amount
                balances[to_account] = balances.get(to_account, 0) + amount
    
    return balances


def filter_transactions_by_month(
    transactions: list[dict],
    year: int,
    month: int,
    ) -> list[dict]:
    """
    Filter transactions to only include those from a specific month.
    
    Use when: User asks about spending "last month" or for a specific month.
              Prevents including transactions from the current month when asking
              about "last month".
    Do NOT use for: Getting all recent transactions (use get_recent_transactions).
    
    Args:
        transactions (list[dict]): Output of get_recent_transactions()
        year (int): Year to filter (e.g., 2025)
        month (int): Month to filter (1-12)
    
    Returns:
        list[dict]: Transactions only from the specified month
    
    Example:
        # User asks on Nov 3 about "last month" spending
        txns = get_recent_transactions(60)
        oct_txns = filter_transactions_by_month(txns, 2025, 10)
    """
    from datetime import datetime
    
    filtered = []
    for txn in transactions:
        txn_date = datetime.strptime(txn["date"], "%Y-%m-%d")
        if txn_date.year == year and txn_date.month == month:
            filtered.append(txn)
    
    return filtered

def calculate_sip_returns(
    monthly_investment: float,
    months: int,
    annual_return_rate: float = 0.10,  # 10% p.a. default for equity
    ) -> dict:
    """
    Calculate SIP (Systematic Investment Plan) returns with monthly compounding.
    
    Use when: User asks about SIP growth, investment returns, or comparing
              different SIP amounts. This accounts for monthly returns unlike
              calculate_savings_projection which is for simple savings.
    Do NOT use for: Non-investment savings (use calculate_savings_projection).
    
    Args:
        monthly_investment (float): SIP amount per month (INR)
        months (int): Investment duration in months
        annual_return_rate (float): Expected annual return (default 0.10 = 10% p.a.)
                                   Typical ranges: 8-12% for equity mutual funds
    
    Returns:
        dict with keys:
          - total_invested (float): Total principal invested
          - final_value (float): Corpus value after returns
          - returns_earned (float): Gains from investment
          - monthly_return_rate (float): Monthly rate used
          - months (int): Duration
    
    Formula: FV = PMT × [((1+r)^n - 1) / r] × (1+r)
    where r = monthly return rate, n = months
    
    Limitation: Assumes constant monthly investment and return rate.
    """
    if months < 1:
        return {"error": "months must be >= 1"}
    if monthly_investment <= 0:
        return {"error": "monthly_investment must be positive"}
    
    monthly_rate = annual_return_rate / 12
    
    if monthly_rate == 0:
        final_value = monthly_investment * months
    else:
        # SIP Future Value formula with compounding
        final_value = monthly_investment * (
            ((1 + monthly_rate) ** months - 1) / monthly_rate
        ) * (1 + monthly_rate)
    
    total_invested = monthly_investment * months
    returns_earned = final_value - total_invested
    
    return {
        "total_invested": round(total_invested, 2),
        "final_value": round(final_value, 2),
        "returns_earned": round(returns_earned, 2),
        "monthly_return_rate": round(monthly_rate, 6),
        "annual_return_rate": annual_return_rate,
        "months": months,
    }

def calculate_savings_projection(
    monthly_savings: float,
    months: int,
    annual_return_rate: float = 0.12,
) -> dict:
    """
    Calculate compound growth of a fixed monthly savings contribution.

    Use when: User asks how much they'll accumulate by a future date given a
              regular savings amount, or to validate if a savings plan is on
              track for a goal with investment returns.
    Do NOT use for: Checking current balances (use get_account_balance),
                    or transactions (use get_recent_transactions).

    Args:
        monthly_savings (float): Fixed amount saved each month (INR, positive number).
        months (int): Number of months to project. Must be >= 1.
        annual_return_rate (float): Expected annual return (default 0.12 = 12% p.a.,
                                   typical Indian equity mutual fund assumption).
                                   Use 0.0 for no-investment simple savings.

    Returns:
        dict with keys:
          - total_contributed (float): Raw principal over all months
          - total_with_returns (float): Compounded future value
          - interest_earned (float): Returns earned beyond contributions
          - monthly_return_rate (float): Monthly rate used
          - months (int): Input months projected

    Limitation: Assumes constant monthly savings. Does not account for inflation.
    """
    if months < 1:
        return {"error": "months must be >= 1"}
    if monthly_savings <= 0:
        return {"error": "monthly_savings must be a positive number"}

    monthly_rate = annual_return_rate / 12

    if monthly_rate == 0:
        future_value = monthly_savings * months
    else:
        # Future Value of Annuity: FV = PMT × [(1+r)^n − 1] / r
        future_value = monthly_savings * (((1 + monthly_rate) ** months - 1) / monthly_rate)

    total_contributed = monthly_savings * months
    interest_earned = future_value - total_contributed

    return {
        "total_contributed": round(total_contributed, 2),
        "total_with_returns": round(future_value, 2),
        "interest_earned": round(interest_earned, 2),
        "monthly_return_rate": round(monthly_rate, 6),
        "months": months,
    }

def analyze_spending_by_category(transactions: list[dict], category: str) -> dict:
    """
    Compute spending statistics for a given category from a transactions list.

    Use when: User asks how much they spent on a specific category (food, rent,
              shopping, etc.), or to identify overspending patterns.
              Always call get_recent_transactions() first, then pass result here.
    Do NOT use for: Checking balances or upcoming bills — use dedicated tools.

    Args:
        transactions (list[dict]): Output of get_recent_transactions().
        category (str): Category to filter. Known values: 'food_delivery',
                        'rent', 'shopping', 'groceries', 'investment',
                        'entertainment', 'fuel'. Case-insensitive.

    Returns:
        dict with keys:
          - category (str): The category analyzed
          - total_spent (float): Total absolute spend (positive number, INR)
          - transaction_count (int): Number of matching transactions
          - average_per_transaction (float): Mean spend per transaction
          - largest_transaction (float): Largest single spend
          - dates (list[str]): Dates of all matching transactions

    Limitation: Only covers categories present in the transactions list.
    """
    cat_lower = category.lower()
    matching = [t for t in transactions
                if t.get("category", "").lower() == cat_lower and t["amount"] < 0]

    if not matching:
        return {
            "category": category,
            "total_spent": 0.0,
            "transaction_count": 0,
            "average_per_transaction": 0.0,
            "largest_transaction": 0.0,
            "dates": [],
        }

    amounts = [abs(t["amount"]) for t in matching]
    return {
        "category": category,
        "total_spent": round(sum(amounts), 2),
        "transaction_count": len(amounts),
        "average_per_transaction": round(sum(amounts) / len(amounts), 2),
        "largest_transaction": round(max(amounts), 2),
        "dates": [t["date"] for t in matching],
    }

def calculate_goal_progress(
    goal_amount: float,
    current_saved: float,
    monthly_contribution: float,
    annual_return_rate: float = 0.12,
 ) -> dict:
    """
    Calculate progress toward a savings goal and estimate months to completion.

    Use when: User asks how far they are from a savings goal, or whether
              their current monthly contribution will meet a target.
              Use get_account_balance() to get current_saved first.
    Do NOT use for: Projecting investment returns in isolation
                    (use calculate_savings_projection instead).

    Args:
        goal_amount (float): Target amount in INR (e.g., 1500000 for ₹15 lakh).
        current_saved (float): Amount already saved toward this goal (INR).
        monthly_contribution (float): How much they plan to add each month (INR).
        annual_return_rate (float): Expected annual return rate (default 0.12).

    Returns:
        dict with keys:
          - goal_amount (float): The target
          - current_saved (float): Amount already in the fund
          - remaining (float): Amount still needed
          - percent_complete (float): 0.0–100.0
          - months_to_goal (int | None): Estimated months at current rate (None if rate=0 and contribution=0)
          - on_track (bool): Whether current progress matches linear projection

    Limitation: months_to_goal is an estimate; assumes constant monthly contribution.
    """
    remaining = max(0.0, goal_amount - current_saved)
    percent_complete = min(100.0, (current_saved / goal_amount) * 100) if goal_amount > 0 else 0.0

    months_to_goal = None
    if monthly_contribution > 0:
        monthly_rate = annual_return_rate / 12
        if monthly_rate == 0:
            months_to_goal = math.ceil(remaining / monthly_contribution)
        else:
            # Solve: FV = current_saved*(1+r)^n + PMT*[(1+r)^n - 1]/r = goal
            # Numerical approximation: iterate months
            accumulated = current_saved
            months_count = 0
            while accumulated < goal_amount and months_count < 600:  # cap at 50 years
                accumulated = accumulated * (1 + monthly_rate) + monthly_contribution
                months_count += 1
            months_to_goal = months_count if accumulated >= goal_amount else None

    return {
        "goal_amount": goal_amount,
        "current_saved": current_saved,
        "remaining": round(remaining, 2),
        "percent_complete": round(percent_complete, 1),
        "months_to_goal": months_to_goal,
        "on_track": months_to_goal is not None,
    }

def calculate_net_worth(balances: dict) -> dict:
    """
    Compute net worth summary from account balances.

    Use when: User asks for their total financial picture, net worth,
              or how all their money is distributed.
              Always call get_account_balance() first, then pass result here.
    Do NOT use for: Checking individual account balances (use get_account_balance directly).

    Args:
        balances (dict): Output of get_account_balance(). Keys: checking, savings,
                         house_fund, mutual_funds (all INR values).

    Returns:
        dict with keys:
          - total_net_worth (float): Sum of all accounts
          - liquid_assets (float): checking + savings (quickly accessible)
          - invested_assets (float): mutual_funds (market-dependent)
          - goal_savings (float): house_fund or other goal accounts
          - breakdown (dict): Percentage share of each account

    Limitation: Does not account for liabilities (loans, credit card debt).
    """
    total = sum(balances.values())
    liquid = balances.get("checking", 0) + balances.get("savings", 0)
    invested = balances.get("mutual_funds", 0)
    goal = balances.get("house_fund", 0)

    breakdown = {
        k: round((v / total) * 100, 1) if total > 0 else 0
        for k, v in balances.items()
    }

    return {
        "total_net_worth": round(total, 2),
        "liquid_assets": round(liquid, 2),
        "invested_assets": round(invested, 2),
        "goal_savings": round(goal, 2),
        "breakdown_percent": breakdown,
    }


def calculate_financial_health_score(
    monthly_income: float,
    monthly_fixed_expenses: float,
    monthly_savings: float,
    total_debt: float = 0.0,
) -> dict:
    """
    Compute a composite financial health score (0–100) for a user.

    Use when: User asks for an overall assessment of their financial health,
              or when you want to add context to spending/savings advice.
              Compute expenses from get_recent_transactions(), savings from
              their commitment, income from their profile.
    Do NOT use for: Point-in-time balance checks (use get_account_balance).

    Args:
        monthly_income (float): Gross monthly income post-tax (INR).
        monthly_fixed_expenses (float): Sum of fixed monthly outflows (rent + SIPs + bills).
        monthly_savings (float): Amount actually saved this month (INR).
        total_debt (float): Outstanding debt / credit card balance (INR, default 0).

    Returns:
        dict with keys:
          - score (int): 0–100 composite health score
          - grade (str): "Excellent / Good / Fair / Poor"
          - savings_rate (float): monthly_savings / monthly_income
          - expense_ratio (float): fixed_expenses / income
          - debt_to_income (float): debt / (income * 12)
          - recommendations (list[str]): actionable tips per dimension

    Limitation: Score is heuristic-based; not a certified financial metric.
    """
    savings_rate = monthly_savings / monthly_income if monthly_income > 0 else 0.0
    expense_ratio = monthly_fixed_expenses / monthly_income if monthly_income > 0 else 1.0
    debt_to_income = total_debt / (monthly_income * 12) if monthly_income > 0 else 0.0

    # Score components (each 0-25 points)
    savings_score = min(25, savings_rate * 100)  # 25% savings rate = full marks
    expense_score = max(0, 25 - (expense_ratio * 25))  # lower expense ratio = better
    debt_score = max(0, 25 - (debt_to_income * 50))    # 0 debt = full marks
    buffer_score = min(25, ((monthly_income - monthly_fixed_expenses - monthly_savings) / monthly_income) * 50)

    score = int(savings_score + expense_score + debt_score + buffer_score)
    score = max(0, min(100, score))

    if score >= 80: grade = "Excellent"
    elif score >= 60: grade = "Good"
    elif score >= 40: grade = "Fair"
    else: grade = "Poor"

    recs = []
    if savings_rate < 0.20: recs.append("Increase savings rate to at least 20% of income")
    if expense_ratio > 0.50: recs.append("Fixed expenses exceed 50% of income — review recurring costs")
    if debt_to_income > 0.30: recs.append("High debt-to-income ratio — prioritise debt payoff")
    if not recs: recs.append("Finances are in good shape — maintain discipline")

    return {
        "score": score,
        "grade": grade,
        "savings_rate": round(savings_rate, 3),
        "expense_ratio": round(expense_ratio, 3),
        "debt_to_income": round(debt_to_income, 3),
        "recommendations": recs,
    }


# ─── Extended Tool Registry ───────────────────────────────────────────────────

TOOL_REGISTRY = {
    # Original stubs
    "get_recent_transactions":         get_recent_transactions,
    "get_account_balance":             get_account_balance,
    "get_upcoming_bills":              get_upcoming_bills,
    "set_reminder":                    set_reminder,
    # Production tools
    "calculate_current_balance_from_transactions": calculate_current_balance_from_transactions,
    "calculate_sip_returns":           calculate_sip_returns,
    "calculate_savings_projection":    calculate_savings_projection,
    "analyze_spending_by_category":    analyze_spending_by_category,
    "calculate_goal_progress":         calculate_goal_progress,
    "calculate_net_worth":             calculate_net_worth,
    "calculate_financial_health_score": calculate_financial_health_score,
}
