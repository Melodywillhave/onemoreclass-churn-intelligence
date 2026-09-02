from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"

PREDICTIONS_PATH = Path(
    "artifacts/predictions/churn_predictions.parquet"
)


st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Retention Intelligence")
st.caption(
    "AI-powered churn risk analysis and retention decision support"
)


@st.cache_data
def load_high_risk_customers() -> pd.DataFrame:
    predictions = pd.read_parquet(
        PREDICTIONS_PATH
    )

    high_risk = (
        predictions.loc[
            predictions["risk_tier"] == "High",
            [
                "customer_id",
                "risk_rank",
                "churn_probability",
                "primary_driver",
                "secondary_driver",
            ],
        ]
        .sort_values("risk_rank")
        .reset_index(drop=True)
    )

    return high_risk


high_risk = load_high_risk_customers()


# Sidebar filters
st.sidebar.header("Queue Filters")

driver_options = [
    "All",
    *sorted(
        high_risk["primary_driver"]
        .dropna()
        .unique()
        .tolist()
    ),
]

driver_filter = st.sidebar.selectbox(
    "Primary Driver",
    options=driver_options,
)

if driver_filter == "All":
    filtered_customers = high_risk.copy()
else:
    filtered_customers = high_risk.loc[
        high_risk["primary_driver"]
        == driver_filter
    ].copy()

st.sidebar.metric(
    "High-Risk Customers",
    len(filtered_customers),
)


# High-risk queue
st.subheader("High-Risk Customer Queue")

queue_display = (
    filtered_customers.head(10)[
        [
            "risk_rank",
            "customer_id",
            "churn_probability",
            "primary_driver",
        ]
    ]
    .copy()
)

queue_display["churn_probability"] = (
    queue_display["churn_probability"]
    .map(lambda value: f"{value:.1%}")
)

queue_display = queue_display.rename(
    columns={
        "risk_rank": "Rank",
        "customer_id": "Customer ID",
        "churn_probability": "Churn Risk",
        "primary_driver": "Primary Driver",
    }
)

st.dataframe(
    queue_display,
    hide_index=True,
    use_container_width=True,
)


# Customer selection
customer_options = (
    filtered_customers["customer_id"]
    .astype(int)
    .tolist()
)

customer_rank_map = dict(
    zip(
        filtered_customers["customer_id"],
        filtered_customers["risk_rank"],
    )
)

selected_customer_id = st.selectbox(
    "Select Customer for Analysis",
    options=customer_options,
    format_func=lambda customer_id: (
        f"{customer_id} | "
        f"Rank #{int(customer_rank_map[customer_id])}"
    ),
)

analyze_button = st.button(
    "Analyze Customer",
    type="primary",
)


if analyze_button:
    customer_id = int(selected_customer_id)

    with st.spinner("Analyzing customer..."):
        try:
            risk_response = requests.get(
                f"{API_BASE_URL}/customers/{customer_id}/risk",
                timeout=30,
            )

            if risk_response.status_code != 200:
                try:
                    error_detail = risk_response.json().get(
                        "detail",
                        "Unknown API error",
                    )
                except ValueError:
                    error_detail = risk_response.text

                st.error(
                    f"Could not load customer risk: "
                    f"{error_detail}"
                )
                st.stop()

            risk = risk_response.json()

            recommendation_response = requests.post(
                f"{API_BASE_URL}/recommendations/{customer_id}",
                timeout=120,
            )

            if recommendation_response.status_code != 200:
                try:
                    error_detail = (
                        recommendation_response
                        .json()
                        .get(
                            "detail",
                            "Unknown API error",
                        )
                    )
                except ValueError:
                    error_detail = (
                        recommendation_response.text
                    )

                st.error(
                    f"Could not generate recommendation: "
                    f"{error_detail}"
                )
                st.stop()

            recommendation = (
                recommendation_response.json()
            )

        except requests.RequestException as exc:
            st.error(
                "Could not connect to the API. "
                "Make sure FastAPI is running."
            )
            st.caption(str(exc))
            st.stop()

    st.success("Analysis complete")

    st.divider()

    # Risk assessment
    st.subheader("Customer Risk Assessment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Churn Risk",
            f"{risk['churn_probability']:.1%}",
        )

    with col2:
        st.metric(
            "Risk Tier",
            risk["risk_tier"],
        )

    with col3:
        st.metric(
            "Risk Rank",
            f"#{risk['risk_rank']}",
        )

    with col4:
        st.metric(
            "Primary Driver",
            risk["primary_driver"],
        )

    st.caption(
        f"Secondary Driver: "
        f"{risk['secondary_driver']} "
        f"• Snapshot: {risk['snapshot_date']}"
    )

    st.divider()

    # AI recommendation
    st.subheader("AI Retention Recommendation")

    action = recommendation[
        "recommended_action"
    ]

    st.markdown(
        f"### {action['action']}"
    )

    st.write(
        action["details"]
    )

    # Risk summary
    st.subheader("Risk Summary")

    st.write(
        recommendation["risk_summary"]
    )

    # Supporting evidence
    st.subheader("Supporting Evidence")

    evidence = recommendation[
        "supporting_evidence"
    ]

    if evidence:
        for item in evidence:
            st.markdown(
                f"- {item}"
            )
    else:
        st.info(
            "No strong supporting risk signal "
            "was identified."
        )

    # Secondary consideration
    st.subheader("Secondary Consideration")

    st.write(
        recommendation[
            "secondary_consideration"
        ]
    )

    # Policy rationale
    st.subheader("Policy Rationale")

    for rationale in recommendation[
        "policy_rationale"
    ]:
        with st.expander(
            rationale["section"]
        ):
            st.write(
                f"**Source:** "
                f"{rationale['source']}"
            )
            st.write(
                rationale["reason"]
            )

    # Guardrails
    st.subheader("Guardrails")

    for guardrail in recommendation[
        "guardrails"
    ]:
        st.markdown(
            f"- {guardrail}"
        )