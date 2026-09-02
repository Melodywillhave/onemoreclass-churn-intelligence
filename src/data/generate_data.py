from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

RAW_DATA_DIR = Path("data/raw")


def reset_random_seed() -> None:
    global rng
    rng = np.random.default_rng(RANDOM_SEED)

def generate_customers(n_customers: int = 5000) -> pd.DataFrame:

    customer_ids = np.arange(
        100001,
        100001 + n_customers
    )

    signup_dates = (
        pd.to_datetime("2025-01-01")
        + pd.to_timedelta(
            rng.integers(
                0,
                365,
                size=n_customers
            ),
            unit="D"
        )
    )

    universities = rng.choice(
        ["XJTLU", "UoL"],
        size=n_customers,
        p=[0.70, 0.30]
    )

    class_years = rng.choice(
            ["Year 1", "Year 2", "Year 3", "Year 4"],
            size=n_customers,
            p=[0.27, 0.26, 0.25, 0.22]
        )
    ages = []

    for class_year in class_years:

        if class_year == "Year 1":
            age = rng.choice(
                [18, 19, 20],
                p=[0.70, 0.25, 0.05]
            )

        elif class_year == "Year 2":
            age = rng.choice(
                [19, 20, 21],
                p=[0.65, 0.30, 0.05]
            )

        elif class_year == "Year 3":
            age = rng.choice(
                [20, 21, 22],
                p=[0.60, 0.35, 0.05]
            )

        else:
            age = rng.choice(
                [21, 22],
                p=[0.65, 0.35]
            )

        ages.append(age)

    plan_types = rng.choice(
        ["Free", "Basic", "Premium"],
        size=n_customers,
        p=[0.45, 0.35, 0.20]
    )

    acquisition_channels = rng.choice(
        [
            "Organic",
            "Paid Search",
            "Social",
            "Referral"
        ],
        size=n_customers,
        p=[0.22, 0.18, 0.42, 0.18]
    )

    academic_groups = []

    for university in universities:

        if university == "XJTLU":

            academic_group = rng.choice(
                [
                    "Business & Economics",
                    "Computer Science & AI",
                    "Engineering",
                    "Mathematics & Statistics",
                    "Natural & Life Sciences",
                    "Humanities & Social Sciences",
                    "Architecture & Design"
                ],
                p=[
                    0.28,
                    0.19,
                    0.16,
                    0.12,
                    0.09,
                    0.09,
                    0.07
                ]
            )

        else:

            academic_group = rng.choice(
                [
                    "Business & Economics",
                    "Computer Science & AI",
                    "Engineering",
                    "Mathematics & Statistics",
                    "Natural & Life Sciences",
                    "Humanities & Social Sciences",
                    "Architecture & Design"
                ],
                p=[
                    0.20,
                    0.15,
                    0.15,
                    0.11,
                    0.14,
                    0.17,
                    0.08
                ]
            )

        academic_groups.append(
            academic_group
        )

    countries = np.where(
        universities == "XJTLU",
        "China",
        "UK"
    )

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "signup_date": signup_dates,
            "university": universities,
            "country": countries,
            "age": ages,
            "class_year": class_years,
            "academic_group": academic_groups,
            "plan_type": plan_types,
            "acquisition_channel": acquisition_channels
        }
    )

    return customers

def save_customers(customers: pd.DataFrame) -> None:

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RAW_DATA_DIR / "customers.csv"

    customers.to_csv(
        output_path,
        index=False
    )

    print(f"Customers saved to: {output_path}")

def generate_latent_profiles(
    customers: pd.DataFrame
) -> pd.DataFrame:

    profiles = customers[
        ["customer_id"]
    ].copy()

    profiles["engagement_propensity"] = rng.beta(
        2.5,
        2.0,
        size=len(customers)
    )

    profiles["purchase_propensity"] = rng.beta(
        2.0,
        3.0,
        size=len(customers)
    )

    profiles["price_sensitivity"] = rng.beta(
        2.5,
        2.5,
        size=len(customers)
    )

    profiles["support_propensity"] = rng.beta(
        1.5,
        5.0,
        size=len(customers)
    )

    profiles["behavior_type"] = rng.choice(
        [
            "stable",
            "declining",
            "growing"
        ],
        size=len(customers),
        p=[
            0.55,
            0.30,
            0.15
        ]
    )

    return profiles

def generate_course_catalog() -> pd.DataFrame:

    courses = [
        # XJTLU
        {
            "course_id": "XJ001",
            "university": "XJTLU",
            "academic_group": "Mathematics & Statistics",
            "course_name": "Calculus I",
            "recorded_price": 129,
            "live_price": 799,
            "currency": "RMB",
        },
        {
            "course_id": "XJ002",
            "university": "XJTLU",
            "academic_group": "Mathematics & Statistics",
            "course_name": "Statistics",
            "recorded_price": 159,
            "live_price": 899,
            "currency": "RMB",
        },
        {
            "course_id": "XJ005",
            "university": "XJTLU",
            "academic_group": "Mathematics & Statistics",
            "course_name": "Linear Algebra",
            "recorded_price": 149,
            "live_price": 849,
            "currency": "RMB",
        },
        {
            "course_id": "XJ006",
            "university": "XJTLU",
            "academic_group": "Mathematics & Statistics",
            "course_name": "Probability & Random Processes",
            "recorded_price": 199,
            "live_price": 999,
            "currency": "RMB",
        },
        {
            "course_id": "XJ003",
            "university": "XJTLU",
            "academic_group": "Computer Science & AI",
            "course_name": "Python Programming",
            "recorded_price": 199,
            "live_price": 899,
            "currency": "RMB",
        },
        {
            "course_id": "XJ007",
            "university": "XJTLU",
            "academic_group": "Computer Science & AI",
            "course_name": "Data Structures & Algorithms",
            "recorded_price": 219,
            "live_price": 949,
            "currency": "RMB",
        },
        {
            "course_id": "XJ004",
            "university": "XJTLU",
            "academic_group": "Business & Economics",
            "course_name": "Microeconomics",
            "recorded_price": 129,
            "live_price": 799,
            "currency": "RMB",
        },
        {
            "course_id": "XJ008",
            "university": "XJTLU",
            "academic_group": "Engineering",
            "course_name": "Digital Logic",
            "recorded_price": 179,
            "live_price": 899,
            "currency": "RMB",
        },
        {
            "course_id": "XJ009",
            "university": "XJTLU",
            "academic_group": "Engineering",
            "course_name": "Thermodynamics",
            "recorded_price": 189,
            "live_price": 949,
            "currency": "RMB",
        },
        {
            "course_id": "XJ010",
            "university": "XJTLU",
            "academic_group": "Natural & Life Sciences",
            "course_name": "Biochemistry",
            "recorded_price": 149,
            "live_price": 799,
            "currency": "RMB",
        },
        {
            "course_id": "XJ011",
            "university": "XJTLU",
            "academic_group": "Natural & Life Sciences",
            "course_name": "Molecular Biology",
            "recorded_price": 169,
            "live_price": 899,
            "currency": "RMB",
        },
        {
            "course_id": "XJ012",
            "university": "XJTLU",
            "academic_group": "Humanities & Social Sciences",
            "course_name": "Chinese Literature",
            "recorded_price": 99,
            "live_price": 699,
            "currency": "RMB",
        },
        {
            "course_id": "XJ013",
            "university": "XJTLU",
            "academic_group": "Humanities & Social Sciences",
            "course_name": "International Relations",
            "recorded_price": 109,
            "live_price": 699,
            "currency": "RMB",
        },
        {
            "course_id": "XJ014",
            "university": "XJTLU",
            "academic_group": "Architecture & Design",
            "course_name": "Architectural Design Studio",
            "recorded_price": 259,
            "live_price": 999,
            "currency": "RMB",
        },
        {
            "course_id": "XJ015",
            "university": "XJTLU",
            "academic_group": "Architecture & Design",
            "course_name": "Urban Planning",
            "recorded_price": 239,
            "live_price": 999,
            "currency": "RMB",
        },

        # UoL
        {
            "course_id": "UL001",
            "university": "UoL",
            "academic_group": "Mathematics & Statistics",
            "course_name": "Calculus I",
            "recorded_price": 79,
            "live_price": 499,
            "currency": "GBP",
        },
        {
            "course_id": "UL002",
            "university": "UoL",
            "academic_group": "Mathematics & Statistics",
            "course_name": "Statistics",
            "recorded_price": 99,
            "live_price": 599,
            "currency": "GBP",
        },
        {
            "course_id": "UL005",
            "university": "UoL",
            "academic_group": "Mathematics & Statistics",
            "course_name": "Linear Algebra",
            "recorded_price": 89,
            "live_price": 549,
            "currency": "GBP",
        },
        {
            "course_id": "UL006",
            "university": "UoL",
            "academic_group": "Mathematics & Statistics",
            "course_name": "Probability & Random Processes",
            "recorded_price": 109,
            "live_price": 649,
            "currency": "GBP",
        },
        {
            "course_id": "UL003",
            "university": "UoL",
            "academic_group": "Computer Science & AI",
            "course_name": "Python Programming",
            "recorded_price": 129,
            "live_price": 599,
            "currency": "GBP",
        },
        {
            "course_id": "UL007",
            "university": "UoL",
            "academic_group": "Computer Science & AI",
            "course_name": "Data Structures & Algorithms",
            "recorded_price": 149,
            "live_price": 649,
            "currency": "GBP",
        },
        {
            "course_id": "UL004",
            "university": "UoL",
            "academic_group": "Business & Economics",
            "course_name": "Microeconomics",
            "recorded_price": 79,
            "live_price": 499,
            "currency": "GBP",
        },
        {
            "course_id": "UL008",
            "university": "UoL",
            "academic_group": "Engineering",
            "course_name": "Digital Logic",
            "recorded_price": 99,
            "live_price": 599,
            "currency": "GBP",
        },
        {
            "course_id": "UL009",
            "university": "UoL",
            "academic_group": "Engineering",
            "course_name": "Thermodynamics",
            "recorded_price": 109,
            "live_price": 649,
            "currency": "GBP",
        },
        {
            "course_id": "UL010",
            "university": "UoL",
            "academic_group": "Natural & Life Sciences",
            "course_name": "Biochemistry",
            "recorded_price": 89,
            "live_price": 499,
            "currency": "GBP",
        },
        {
            "course_id": "UL011",
            "university": "UoL",
            "academic_group": "Natural & Life Sciences",
            "course_name": "Molecular Biology",
            "recorded_price": 99,
            "live_price": 549,
            "currency": "GBP",
        },
        {
            "course_id": "UL012",
            "university": "UoL",
            "academic_group": "Humanities & Social Sciences",
            "course_name": "Chinese Literature",
            "recorded_price": 59,
            "live_price": 399,
            "currency": "GBP",
        },
        {
            "course_id": "UL013",
            "university": "UoL",
            "academic_group": "Humanities & Social Sciences",
            "course_name": "International Relations",
            "recorded_price": 69,
            "live_price": 399,
            "currency": "GBP",
        },
        {
            "course_id": "UL014",
            "university": "UoL",
            "academic_group": "Architecture & Design",
            "course_name": "Architectural Design Studio",
            "recorded_price": 169,
            "live_price": 699,
            "currency": "GBP",
        },
        {
            "course_id": "UL015",
            "university": "UoL",
            "academic_group": "Architecture & Design",
            "course_name": "Urban Planning",
            "recorded_price": 149,
            "live_price": 699,
            "currency": "GBP",
        },
    ]

    return pd.DataFrame(courses)


def save_course_catalog(
    course_catalog: pd.DataFrame
) -> None:

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RAW_DATA_DIR / "courses.csv"

    course_catalog.to_csv(
        output_path,
        index=False
    )

    print(f"Course catalog saved to: {output_path}")


def generate_activities(
    customers: pd.DataFrame,
    latent_profiles: pd.DataFrame,
    start_date: str = "2026-01-01",
    n_days: int = 180
) -> pd.DataFrame:

    dates = pd.date_range(
        start=start_date,
        periods=n_days,
        freq="D"
    )

    customer_profiles = customers[
        ["customer_id"]
    ].merge(
        latent_profiles,
        on="customer_id",
        how="left"
    )

    rows = []

    for customer in customer_profiles.itertuples():

        base_engagement = (
            customer.engagement_propensity
        )

        behavior_type = (
            customer.behavior_type
        )

        change_start = rng.integers(
            int(n_days * 0.45),
            int(n_days * 0.65)
        )

        # ---------------------------------------------------------
        # Customer-specific behavioral trajectory
        # ---------------------------------------------------------

        if behavior_type == "declining":

            # Decline can begin at different points in the semester
            # instead of being concentrated in the second half.
            change_start = rng.integers(
                int(n_days * 0.20),
                int(n_days * 0.70)
            )

            # Customers differ in how quickly their engagement changes.
            change_duration = rng.integers(
                45,
                85
            )

            # Moderate but meaningful decline.
            change_strength = rng.uniform(
                0.35,
                0.70
            )

        elif behavior_type == "growing":

            change_start = rng.integers(
                int(n_days * 0.20),
                int(n_days * 0.70)
            )

            change_duration = rng.integers(
                45,
                85
            )

            change_strength = rng.uniform(
                0.20,
                0.45
            )

        else:

            # Stable customers can still fluctuate slightly,
            # but should not have a systematic trajectory.
            change_start = rng.integers(
                int(n_days * 0.20),
                int(n_days * 0.70)
            )

            change_duration = rng.integers(
                45,
                85
            )

            change_strength = rng.uniform(
                0.00,
                0.05
            )

        for day_index, activity_date in enumerate(dates):

            engagement_multiplier = 1.0

            if day_index >= change_start:

                change_progress = np.clip(
                    (
                        day_index - change_start
                    ) / change_duration,
                    0.0,
                    1.0
                )

                if behavior_type == "declining":

                    engagement_multiplier = max(
                        0.20,
                        1.0
                        - change_strength
                        * change_progress
                    )

                elif behavior_type == "growing":

                    engagement_multiplier = (
                        1.0
                        + change_strength
                        * change_progress
                    )

            effective_engagement = (
                base_engagement
                * engagement_multiplier
            )

            login_count = rng.poisson(
                0.2
                + 2.5
                * effective_engagement
            )

            videos_watched = rng.poisson(
                0.3
                + 1.8
                * login_count
            )

            minutes_watched = (
                0.0
                if videos_watched == 0
                else max(
                    1,
                    rng.normal(
                        loc=videos_watched * 12,
                        scale=videos_watched * 3
                    )
                )
            )

            courses_started = rng.binomial(
                1,
                min(
                    0.30,
                    0.03
                    + effective_engagement
                    * 0.20
                )
            )

            courses_completed = rng.binomial(
                1,
                min(
                    0.20,
                    effective_engagement
                    * 0.15
                )
            )

            assignments_completed = rng.poisson(
                effective_engagement
                * 0.7
            )

            rows.append(
                {
                    "customer_id": customer.customer_id,
                    "activity_date": activity_date,
                    "login_count": login_count,
                    "videos_watched": videos_watched,
                    "minutes_watched": round(
                        float(minutes_watched),
                        2
                    ),
                    "courses_started": courses_started,
                    "courses_completed": courses_completed,
                    "assignments_completed": assignments_completed
                }
            )

    return pd.DataFrame(rows)

def save_activities(
    activities: pd.DataFrame
) -> None:

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RAW_DATA_DIR
        / "activities.csv"
    )

    activities.to_csv(
        output_path,
        index=False
    )

    print(
        f"Activities saved to: "
        f"{output_path}"
    )

def generate_transaction_day(
    n_days: int,
    behavior_type: str | None = None
) -> tuple[int, str]:

    if behavior_type == "declining":
        period_weights = {
            "semester_start": 0.50,
            "midterm": 0.30,
            "final": 0.20,
        }
    elif behavior_type == "growing":
        period_weights = {
            "semester_start": 0.35,
            "midterm": 0.35,
            "final": 0.30,
        }
    else:
        period_weights = {
            "semester_start": 0.45,
            "midterm": 0.30,
            "final": 0.25,
        }

    purchase_period = rng.choice(
        list(period_weights.keys()),
        p=list(period_weights.values())
    )

    if purchase_period == "semester_start":
        day = rng.integers(
            0,
            min(25, n_days)
        )

    elif purchase_period == "midterm":
        day = rng.integers(
            max(60, 0),
            min(90, n_days)
        )

    else:
        day = rng.integers(
            max(135, 0),
            min(170, n_days)
        )

    return int(day), purchase_period


def choose_delivery_mode(
    purchase_period: str,
    price_sensitivity: float
) -> str:

    if purchase_period == "semester_start":

        base_live_probability = 0.45
        live_probability = (
            base_live_probability
            - 0.30 * price_sensitivity
        )

    elif purchase_period == "midterm":

        base_live_probability = 0.25
        live_probability = (
            base_live_probability
            - 0.12 * price_sensitivity
        )

    else:

        base_live_probability = 0.12
        live_probability = (
            base_live_probability
            - 0.05 * price_sensitivity
        )

    live_probability = np.clip(
        live_probability,
        0.03,
        0.50
    )

    return rng.choice(
        ["Recorded", "Live"],
        p=[
            1 - live_probability,
            live_probability
        ]
    )


def choose_promotion_tier(
    price_sensitivity: float,
    purchase_period: str
) -> str:

    if price_sensitivity >= 0.75:
        weights = {
            "None": 0.55,
            "10% OFF": 0.20,
            "15% OFF": 0.15,
            "20% OFF": 0.10,
        }
    elif price_sensitivity >= 0.50:
        weights = {
            "None": 0.68,
            "10% OFF": 0.17,
            "15% OFF": 0.10,
            "20% OFF": 0.05,
        }
    else:
        weights = {
            "None": 0.82,
            "10% OFF": 0.10,
            "15% OFF": 0.05,
            "20% OFF": 0.03,
        }

    if purchase_period == "final":
        weights["None"] += 0.08
        weights["10% OFF"] -= 0.04
        weights["15% OFF"] -= 0.02
        weights["20% OFF"] -= 0.02

    total = sum(weights.values())
    normalized = {k: v / total for k, v in weights.items()}

    return rng.choice(
        list(normalized.keys()),
        p=list(normalized.values())
    )


def choose_course_for_customer(
    customer_row: pd.Series,
    course_catalog: pd.DataFrame,
) -> pd.Series:

    eligible_courses = course_catalog[
        (course_catalog["university"] == customer_row["university"]) &
        (course_catalog["academic_group"] == customer_row["academic_group"])
    ]

    if eligible_courses.empty:
        eligible_courses = course_catalog[
            course_catalog["university"] == customer_row["university"]
        ]

    if eligible_courses.empty:
        eligible_courses = course_catalog

    sampled_index = rng.integers(
        len(eligible_courses)
    )

    return eligible_courses.iloc[sampled_index]


def generate_transactions(
    customers: pd.DataFrame,
    latent_profiles: pd.DataFrame,
    course_catalog: pd.DataFrame,
    start_date: str = "2026-01-01",
    n_days: int = 180
) -> pd.DataFrame:

    customer_profiles = customers[
        ["customer_id", "university", "academic_group"]
    ].merge(
        latent_profiles,
        on="customer_id",
        how="left"
    )

    rows = []

    for customer in customer_profiles.itertuples(index=False):

        base_purchase_rate = (
            0.45
            + 0.90 * customer.purchase_propensity
            + 0.60 * customer.engagement_propensity
        )

        if customer.behavior_type == "declining":
            base_purchase_rate *= 0.82
        elif customer.behavior_type == "growing":
            base_purchase_rate *= 1.18

        base_purchase_rate = np.clip(
            base_purchase_rate,
            0.10,
            2.20
        )

        n_transactions = rng.poisson(
            lam=base_purchase_rate
        )

        if (
            n_transactions == 0
            and customer.purchase_propensity > 0.75
            and rng.random() < 0.25
        ):
            n_transactions = 1

        for _ in range(n_transactions):

            transaction_day, purchase_period = generate_transaction_day(
                n_days=n_days,
                behavior_type=customer.behavior_type
            )

            transaction_date = pd.Timestamp(start_date) + pd.to_timedelta(
                transaction_day,
                unit="D"
            )

            course_row = choose_course_for_customer(
                customer_row=customer._asdict(),
                course_catalog=course_catalog,
            )

            delivery_mode = choose_delivery_mode(
                purchase_period=purchase_period,
                price_sensitivity=customer.price_sensitivity
            )

            list_price = (
                course_row["live_price"]
                if delivery_mode == "Live"
                else course_row["recorded_price"]
            )

            promotion_tier = choose_promotion_tier(
                price_sensitivity=customer.price_sensitivity,
                purchase_period=purchase_period
            )

            promotion_tier = promotion_tier or "None"
            discount_rate = 0.0
            discount_used = False

            if promotion_tier == "10% OFF":
                discount_used = True
                discount_rate = 0.10
            elif promotion_tier == "15% OFF":
                discount_used = True
                discount_rate = 0.15
            elif promotion_tier == "20% OFF":
                discount_used = True
                discount_rate = 0.20
            else:
                promotion_tier = "None"
                discount_rate = 0.0
                discount_used = False

            final_amount = round(
                list_price * (1 - discount_rate),
                2
            )

            rows.append(
                {
                    "transaction_id": f"TX{len(rows) + 1:08d}",
                    "customer_id": customer.customer_id,
                    "transaction_date": transaction_date,
                    "purchase_period": purchase_period,
                    "course_id": course_row["course_id"],
                    "course_name": course_row["course_name"],
                    "delivery_mode": delivery_mode,
                    "list_price": float(list_price),
                    "currency": course_row["currency"],
                    "discount_used": discount_used,
                    "discount_rate": round(float(discount_rate), 4),
                    "promotion_tier": promotion_tier,
                    "final_amount": final_amount,
                    "payment_status": rng.choice(
                        ["Paid", "Pending", "Failed"],
                        p=[0.96, 0.03, 0.01]
                    )
                }
            )

    transactions = pd.DataFrame(rows)

    if not transactions.empty:
        transactions["transaction_date"] = pd.to_datetime(
            transactions["transaction_date"]
        )

    return transactions


def generate_interactions(
    customers: pd.DataFrame,
    latent_profiles: pd.DataFrame,
    start_date: str = "2026-01-01",
    n_days: int = 180,
) -> pd.DataFrame:

    service_issue_types = {
        "support_ticket",
        "complaint",
        "refund_request",
        "billing_inquiry",
    }

    customer_profiles = customers[
        ["customer_id"]
    ].merge(
        latent_profiles,
        on="customer_id",
        how="left"
    )

    rows = []

    for customer in customer_profiles.itertuples(index=False):

        base_rate = (
            0.6
            + 1.2 * customer.support_propensity
            + 0.8 * customer.engagement_propensity
        )

        if customer.behavior_type == "declining":
            base_rate *= 1.9
        elif customer.behavior_type == "growing":
            base_rate *= 1.1

        base_rate = float(np.clip(base_rate, 0.5, 6.0))
        n_interactions = rng.poisson(lam=base_rate)

        if customer.behavior_type == "declining" and customer.support_propensity > 0.55:
            n_interactions += rng.poisson(1.4)

        for _ in range(n_interactions):
            if customer.behavior_type == "declining" and rng.random() < 0.55:
                interaction_type = rng.choice(
                    [
                        "complaint",
                        "refund_request",
                        "support_ticket",
                        "billing_inquiry",
                        "general_inquiry",
                        "course_feedback",
                    ],
                    p=[0.22, 0.20, 0.20, 0.12, 0.16, 0.10],
                )
            else:
                interaction_type = rng.choice(
                    [
                        "general_inquiry",
                        "course_feedback",
                        "billing_inquiry",
                        "support_ticket",
                        "complaint",
                    ],
                    p=[0.28, 0.25, 0.18, 0.18, 0.11],
                )

            complaint_flag = int(interaction_type == "complaint")
            refund_request_flag = int(interaction_type == "refund_request")
            requires_resolution = interaction_type in service_issue_types

            if requires_resolution:
                if customer.behavior_type == "declining":
                    relative_day = rng.beta(3.0, 1.5)
                    event_day = int(relative_day * (n_days - 1))
                else:
                    event_day = int(rng.integers(0, n_days))

                issue_resolved = int(rng.random() < (0.35 if customer.behavior_type == "declining" else 0.75))

                if issue_resolved == 1:
                    resolution_time_hours = round(
                        float(
                            np.clip(
                                rng.normal(loc=10.0, scale=6.0),
                                1.0,
                                72.0,
                            )
                        ),
                        2,
                    )
                else:
                    resolution_time_hours = round(
                        float(
                            np.clip(
                                rng.normal(loc=64.0, scale=30.0),
                                12.0,
                                240.0,
                            )
                        ),
                        2,
                    )

                if complaint_flag == 1:
                    satisfaction_score = round(
                        float(
                            np.clip(
                                rng.normal(
                                    loc=2.1 if issue_resolved else 1.2,
                                    scale=1.1,
                                ),
                                0.0,
                                5.0,
                            )
                        ),
                        2,
                    )
                else:
                    satisfaction_score = round(
                        float(
                            np.clip(
                                rng.normal(
                                    loc=3.8 if issue_resolved else 2.6,
                                    scale=1.0,
                                ),
                                0.0,
                                5.0,
                            )
                        ),
                        2,
                    )
            else:
                if customer.behavior_type == "declining":
                    event_day = int(rng.beta(1.6, 2.0) * (n_days - 1))
                else:
                    event_day = int(rng.integers(0, n_days))

                issue_resolved = np.nan
                resolution_time_hours = np.nan
                satisfaction_score = round(
                    float(
                        np.clip(
                            rng.normal(
                                loc=4.4,
                                scale=0.7,
                            ),
                            0.0,
                            5.0,
                        )
                    ),
                    2,
                )

            event_date = pd.Timestamp(start_date) + pd.to_timedelta(
                event_day,
                unit="D",
            )

            rows.append(
                {
                    "interaction_id": f"INT{len(rows) + 1:08d}",
                    "customer_id": customer.customer_id,
                    "interaction_date": event_date,
                    "interaction_type": interaction_type,
                    "complaint_flag": complaint_flag,
                    "refund_request_flag": refund_request_flag,
                    "issue_resolved": issue_resolved,
                    "resolution_time_hours": resolution_time_hours,
                    "satisfaction_score": satisfaction_score,
                }
            )

    interactions = pd.DataFrame(rows)

    if not interactions.empty:
        interactions["interaction_date"] = pd.to_datetime(
            interactions["interaction_date"]
        )

    return interactions


def save_interactions(
    interactions: pd.DataFrame
) -> None:

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RAW_DATA_DIR / "interactions.csv"

    interactions.to_csv(
        output_path,
        index=False,
    )

    print(f"Interactions saved to: {output_path}")


def save_transactions(
    transactions: pd.DataFrame
) -> None:

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = RAW_DATA_DIR / "transactions.csv"

    transactions.to_csv(
        output_path,
        index=False
    )

    print(f"Transactions saved to: {output_path}")


if __name__ == "__main__":

    reset_random_seed()

    customers = generate_customers()

    latent_profiles = generate_latent_profiles(
        customers
    )

    course_catalog = generate_course_catalog()

    activities = generate_activities(
        customers,
        latent_profiles
    )

    transactions = generate_transactions(
        customers,
        latent_profiles,
        course_catalog
    )

    interactions = generate_interactions(
        customers,
        latent_profiles
    )

    save_customers(
        customers
    )

    save_course_catalog(
        course_catalog
    )

    save_activities(
        activities
    )

    save_transactions(
        transactions
    )

    save_interactions(
        interactions
    )

    print("\nCustomers:")
    print(
        customers.head()
    )

    print("\nCustomer shape:")
    print(
        customers.shape
    )

    print("\nCourse catalog:")
    print(
        course_catalog.head()
    )

    print("\nActivities:")
    print(
        activities.head()
    )

    print("\nActivity shape:")
    print(
        activities.shape
    )

    print("\nTransactions:")
    print(
        transactions.head()
    )

    print("\nTransaction shape:")
    print(
        transactions.shape
    )

    print("\nInteractions:")
    print(
        interactions.head()
    )

    print("\nInteraction shape:")
    print(
        interactions.shape
    )

    print("\nLatent behavior distribution:")
    print(
        latent_profiles["behavior_type"]
        .value_counts(normalize=True)
    )