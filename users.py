from typing import Any


# ─── User Profile Schema ──────────────────────────────────────────────────────

def _profile(name, age, city, income, goal):
    return {
        "name": name,
        "age": age,
        "city": city,
        "monthly_income_inr": income,
        "stated_goal": goal,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# USER 1: Priya Sharma (original project user)
# ═══════════════════════════════════════════════════════════════════════════════

PRIYA = _profile(
    "Priya Sharma", 28, "Bangalore", 120000,
    "Save ₹15 lakh in 2 years for a house down payment in Bangalore"
)

PRIYA_SESSIONS = {
    "1": [
        "I just got my salary credited. Help me figure out how much I can realistically save this month.",
        "I feel like I'm spending too much on food delivery. How much did I actually spend on it last month?",
        "Okay that's worse than I thought. Let's say I want to cut that in half AND put aside ₹30,000 for my house fund this month — is that realistic given my upcoming bills?",
        "Got it. Remind me to actually transfer the ₹30,000 to my house fund on the 25th.",
    ],
    "2": [
        "Hey, my colleague is selling his MacBook for ₹80,000, barely used. I've been wanting to upgrade. Should I buy it?",
    ],
     "3": [
        "I've been investing ₹10,000/month in SIPs. Is that enough given my 2-year timeline?",
        "What if I added ₹5,000 more per month to my SIP? Show me the difference.",
        "The math looks good. Set a reminder to review my SIP amount on December 15th.",
    ],
    "4": [
        "December is coming up. What ELSS funds should I consider for tax saving under 80C?",
        "I have about ₹40,000 left in my 80C limit. Should I put it all in ELSS or split between PPF and ELSS?",
        "Given my house goal in 2 years, does putting money in PPF (locked 15 years) make sense?",
    ],
    "5": [
        "It's been a month since we set up my savings plan. How am I tracking toward my ₹15 lakh goal?",
        "My friends want to plan a Goa trip for New Year. Budget would be around ₹15,000. Should I do it?",
        "Remind me to book flights before December 1st before prices spike.", 
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# USER 2: Arjun Mehta — 34yr, ₹2.5L/mo, early retirement at 45
# ═══════════════════════════════════════════════════════════════════════════════

ARJUN = _profile(
    "Arjun Mehta", 34, "Mumbai", 250000,
    "Retire at 45 with a corpus of ₹5 crore — 11 years to go"
)

ARJUN_SESSIONS = {
    "1": [
        "Just got my year-end bonus of ₹5 lakh. What should I do with it to maximize my retirement corpus?",
        "I already have ₹85,000/month going into SIPs. Should I lumpsum invest the bonus or stagger it?",
        "Assuming 12% annual returns, how long until I hit ₹5 crore at my current rate?",
        "Set a reminder to review my portfolio allocation on January 1st.",
    ],
    "2": [
        "I'm thinking about switching jobs — the new offer is ₹3 lakh/month but in Delhi. Does moving make financial sense?",
        "My rent in Mumbai is ₹45,000. Delhi rent would be similar. What changes in my plan?",
        "The new job has ESPOs worth potentially ₹20 lakh in 4 years. How do I factor that in?",
    ],
    "3": [ 
        "I went with the lumpsum investment of the bonus. Markets are down 8% since then. Should I panic?",
        "My SIPs are continuing. What's my current portfolio value assuming the dip?",
        "Is there a tax harvesting opportunity I should consider before March 31st?",
        "Remind me to review tax-loss harvest positions on March 20th.",
    ],
    "4": [
        "I decided to take the Delhi job. Higher income now. Help me rebuild my financial plan.",
        "New income is ₹3L, rent ₹42,000, lifestyle roughly same. How much more can I invest?",
        "At the new savings rate, does retirement at 45 become retirement at 43?",
    ],
    "5": [
        "What's my current liquid position? I don't want to disturb the SIPs.",
        "If I take a 5-year car loan for ₹6 lakh at 9% interest, what's the EMI and total cost?",
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# USER 3: Sneha Patel — 26yr, ₹75K/mo (consulting), emergency fund + Europe trip
# ═══════════════════════════════════════════════════════════════════════════════

SNEHA = _profile(
    "Sneha Patel", 26, "Pune", 75000,
    "Build a ₹2 lakh emergency fund and save ₹1.5 lakh for a Europe trip in 12 months"
)

SNEHA_SESSIONS = {
    "1": [
        "I'm a freelance consultant — income varies month to month. Last month was ₹75,000, before that ₹50,000. How do I plan with irregular income?",
        "I also have a student loan of ₹2.5 lakh at 10% interest. EMI is ₹5,500/month. Should I prepay it?",
        "What are my actual expenses? I spend on rent (₹18,000), food, travel, subscriptions roughly ₹15,000.",
        "Set a reminder to track my income for 3 months before committing to any investment plan.",
    ],
    "2": [
        "Good month — earned ₹90,000. I want to split this wisely. Emergency fund first, or loan prepayment?",
        "I currently have ₹25,000 in savings. How far am I from a 3-month emergency fund?",
        "My Europe trip is in 10 months. I need ₹1.5 lakh. Is that achievable alongside building savings?",
    ],
    "3": [  # tests semantic memory of irregular income strategy
        "Bad month — only ₹40,000 this month. Client delayed a payment. What should I not cut?",
        "Do I need to pause the Europe fund this month?",
        "What's my minimum viable spend this month to keep the plan alive?",
    ],
    "4": [
        "Loan update — I paid an extra ₹10,000 against principal this month. How much did that help?",
        "What's the loan outstanding and when will I be debt-free at current rate?",
        "With the loan gone sooner, does the Europe trip become easier?",
    ],
    "5": [
        "I got a long-term client contract — fixed ₹80,000/month for 6 months. Planning time.",
        "With guaranteed income for 6 months, rebuild my whole plan — emergency fund, Europe trip, loan.",
        "Should I open a recurring deposit for the Europe fund so I'm forced to save?",
        "Remind me to book Europe flights 6 months in advance — set that for March 1st.",
    ]
}


# ─── Consolidated User Registry ───────────────────────────────────────────────

ALL_USERS = {
    "priya_sharma": {
        "profile": PRIYA,
        "sessions": PRIYA_SESSIONS,
        "user_id": "priya_sharma",
    },
    "arjun_mehta": {
        "profile": ARJUN,
        "sessions": ARJUN_SESSIONS,
        "user_id": "arjun_mehta",
    },
    "sneha_patel": {
        "profile": SNEHA,
        "sessions": SNEHA_SESSIONS,
        "user_id": "sneha_patel",
    }
}


# ─── Per-User Tool Overrides for Simulation ───────────────────────────────────
# These override get_account_balance() and get_upcoming_bills() per user.
# simulate_users.py injects these so each user has realistic unique data.
# Original Priya data from tools.py is used for priya_sharma (no override needed).

USER_TOOL_OVERRIDES: dict[str, dict[str, Any]] = {
    "arjun_mehta": {
        "get_account_balance": {
            "checking": 285000,
            "savings": 320000,
            "retirement_corpus": 1850000,
            "mutual_funds": 1240000,
        },
        "get_upcoming_bills": [
            {"date": "2025-11-05", "amount": 45000, "description": "Rent (Mumbai)"},
            {"date": "2025-11-10", "amount": 85000, "description": "SIP - Equity MFs"},
            {"date": "2025-11-15", "amount": 5200,  "description": "Term Insurance Premium"},
            {"date": "2025-11-20", "amount": 12000, "description": "Credit Card"},
        ],
        "monthly_income_inr": 250000,
    },
    "sneha_patel": {
        "get_account_balance": {
            "checking": 42000,
            "savings": 25000,
            "europe_fund": 60000,
            "mutual_funds": 15000,
        },
        "get_upcoming_bills": [
            {"date": "2025-11-03", "amount": 18000, "description": "Rent (Pune)"},
            {"date": "2025-11-10", "amount": 5500,  "description": "Student Loan EMI"},
            {"date": "2025-11-15", "amount": 3200,  "description": "Mobile + Internet + OTT"},
        ],
        "monthly_income_inr": 75000,
    }
}


# ─── Session Date Mapping (5 sessions in November 2025) ──────────────────────

SESSION_DATES = {
    "1":  "Monday, November 3, 2025",        # Salary planning
    "2":  "Thursday, November 6, 2025",      # MacBook purchase (3 days later)
    "3":  "Monday, November 11, 2025",       # Tax planning (8 days after S1)
    "4":  "Saturday, November 16, 2025",     # SIP review (5 days after S3)
    "5":  "Friday, November 28, 2025",       # Progress check (25 days after S1)
}
