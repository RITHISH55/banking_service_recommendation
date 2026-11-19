"""
Professional Streamlit dashboard for the banking recommendation system.
"""
import pandas as pd
import streamlit as st
from pathlib import Path

from data_generator import PRODUCTS
from inference import (EXAMPLE_PROFILE, analyse_feedback_text, label, load_bundle,
                       profile_to_features, score_existing_customer, score_features)
from preprocessing import ALL_FEATURES, clean_customers, engineer_features

st.set_page_config(
    page_title="Nexora Banking Intelligence",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:global(*) { font-family: 'Inter', sans-serif; }
.stApp { background: #f4f7fb; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101c33 0%, #17284a 100%);
}
[data-testid="stSidebar"] * { color: #eef4ff; }
[data-testid="stSidebar"] .stRadio label { padding: 8px 10px; border-radius: 8px; }
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,.08); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1380px; padding-top: 1.6rem; padding-bottom: 2rem; }
.app-header {
    background: linear-gradient(115deg, #10203d 0%, #1d4776 70%, #2d718f 100%);
    border-radius: 18px; padding: 24px 30px; color: white; margin-bottom: 20px;
    box-shadow: 0 12px 30px rgba(22, 45, 82, .12);
}
.app-header h1 { margin: 0 0 6px 0; font-size: 29px; letter-spacing: -.5px; }
.app-header p { margin: 0; color: #dbe8f6; font-size: 14px; }
.section-title { color: #17243b; font-size: 19px; font-weight: 700; margin: 4px 0 12px; }
.metric-card {
    background: white; border: 1px solid #e6edf5; border-radius: 14px; padding: 17px 18px;
    min-height: 94px; box-shadow: 0 5px 18px rgba(37, 57, 88, .05);
}
.metric-card .label { color: #718096; font-size: 12px; font-weight: 600; text-transform: uppercase; }
.metric-card .value { color: #17243b; font-size: 24px; font-weight: 700; margin-top: 5px; }
.panel {
    background: white; border: 1px solid #e6edf5; border-radius: 16px;
    padding: 18px; box-shadow: 0 5px 18px rgba(37, 57, 88, .05); height: 100%;
}
.recommendation {
    border: 1px solid #e2eaf3; border-left: 4px solid #2d718f; background: #fbfdff;
    border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;
}
.recommendation .rank { color: #2d718f; font-size: 12px; font-weight: 700; }
.recommendation .name { color: #17243b; font-size: 15px; font-weight: 700; margin: 3px 0 5px; }
.recommendation .reason { color: #66758a; font-size: 13px; line-height: 1.45; }
.sentiment-positive { color: #17795b; } .sentiment-neutral { color: #7a5d16; }
.sentiment-negative { color: #b13a3a; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
button[kind="primary"] { border-radius: 9px; background: #1d4776; border: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except FileNotFoundError:
    st.error("Models are not available. Run `python main.py` first.")
    st.stop()


DATA_DIR = Path(__file__).parent / "data"
CUSTOMERS_CSV = DATA_DIR / "customers_raw.csv"
USAGE_CSV = DATA_DIR / "product_usage.csv"


def sync_saved_customers():
    try:
        raw = pd.read_csv(CUSTOMERS_CSV)
        usage = pd.read_csv(USAGE_CSV)
    except FileNotFoundError:
        return

    clean, _ = clean_customers(raw)
    features = engineer_features(clean).set_index("customer_id")
    usage = usage.set_index("customer_id")

    known = set(bundle["customers"].index.astype(str))
    saved_ids = [customer_id for customer_id in features.index.astype(str)
                 if customer_id not in known]
    if not saved_ids:
        return

    bundle["customers"] = pd.concat(
        [bundle["customers"], features.loc[saved_ids]], axis=0)
    bundle["ownership"] = pd.concat([
        bundle["ownership"],
        usage.loc[saved_ids, PRODUCTS],
    ], axis=0)
    for customer_id in saved_ids:
        if customer_id not in bundle["test_ids"]:
            bundle["test_ids"].append(customer_id)
        bundle["neg_areas"].setdefault(customer_id, set())


sync_saved_customers()


def next_customer_id():
    customer_ids = set(bundle["customers"].index.astype(str))
    try:
        saved = pd.read_csv(CUSTOMERS_CSV, usecols=["customer_id"])
        customer_ids.update(saved["customer_id"].astype(str))
    except (FileNotFoundError, ValueError):
        pass

    number = 1
    while f"CUST{number:05d}" in customer_ids:
        number += 1
    return f"CUST{number:05d}"


def add_customer(customer_id, customer_name, profile, owned):
    customer_id = customer_id.strip()
    customer_name = customer_name.strip()
    if not customer_id:
        raise ValueError("Customer ID is required.")
    if not customer_name:
        raise ValueError("Customer name is required.")

    existing_ids = set(pd.read_csv(CUSTOMERS_CSV)["customer_id"].astype(str))
    if customer_id in existing_ids:
        raise ValueError(f"Customer ID '{customer_id}' already exists.")

    profile = {**EXAMPLE_PROFILE, **profile, "customer_id": customer_id}

    customers = pd.read_csv(CUSTOMERS_CSV)
    if "customer_name" not in customers.columns:
        customers.insert(1, "customer_name", "")
    customer_row = {column: profile.get(column, None)
                    for column in customers.columns}
    customer_row["customer_id"] = customer_id
    customer_row["customer_name"] = customer_name
    pd.concat([customers, pd.DataFrame([customer_row])], ignore_index=True) \
        .to_csv(CUSTOMERS_CSV, index=False)

    usage = pd.read_csv(USAGE_CSV)
    usage_row = {"customer_id": customer_id}
    usage_row.update({product: int(product in owned) for product in PRODUCTS})
    pd.concat([usage, pd.DataFrame([usage_row])], ignore_index=True) \
        .to_csv(USAGE_CSV, index=False)

    features = profile_to_features(profile)
    bundle["customers"] = pd.concat([bundle["customers"], features], axis=0)
    bundle["ownership"] = pd.concat([
        bundle["ownership"],
        pd.DataFrame(
            [[int(product in owned) for product in PRODUCTS]],
            columns=PRODUCTS,
            index=[customer_id],
        ),
    ], axis=0)
    if customer_id not in bundle["test_ids"]:
        bundle["test_ids"].append(customer_id)
    bundle["neg_areas"][customer_id] = set()
    return customer_id


def render_metrics(items):
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )


def show_results(probs, owned, recs):
    left, right = st.columns([1.15, 0.85], gap="medium")

    with left:
        st.markdown('<div class="section-title">Service interest</div>', unsafe_allow_html=True)
        interest = pd.DataFrame({
            "Service": [label(product) for product in probs.index],
            "Interest": probs.values,
            "Status": ["Current product" if product in owned else "Opportunity"
                       for product in probs.index],
        }).sort_values("Interest", ascending=False)
        st.dataframe(
            interest,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Interest": st.column_config.ProgressColumn(
                    "Interest score", min_value=0.0, max_value=1.0, format="%.0f%%"),
                "Status": st.column_config.TextColumn("Relationship"),
            },
        )

    with right:
        st.markdown('<div class="section-title">Recommended actions</div>', unsafe_allow_html=True)
        if recs.empty:
            st.info("No additional products are recommended.")
        for _, recommendation in recs.iterrows():
            st.markdown(
                f'<div class="recommendation">'
                f'<div class="rank">RECOMMENDATION {int(recommendation["rank"])}</div>'
                f'<div class="name">{recommendation["product_name"]}</div>'
                f'<div class="reason">{recommendation["reason"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if recommendation["caution"]:
                st.warning(recommendation["caution"])


with st.sidebar:
    st.markdown("## Nexora")
    st.caption("BANKING INTELLIGENCE")
    st.divider()
    page = st.radio(
        "Workspace",
        ["Customer intelligence", "Profile simulator", "Feedback analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Model status")
    st.success("Bundle loaded")


st.markdown(
    '<div class="app-header">'
    '<h1>Personalized Banking Intelligence</h1>'
    '<p>Predict customer interest, prioritize cross-sell actions, and understand feedback.</p>'
    '</div>',
    unsafe_allow_html=True,
)

render_metrics([
    ("Customers", f"{len(bundle['customers']):,}"),
    ("Product models", "6"),
    ("Customer segments", str(len(bundle["segment_names"]))),
    ("Model bundle", "Ready"),
])

if page == "Customer intelligence":
    st.markdown('<div class="section-title">Existing customer</div>', unsafe_allow_html=True)
    try:
        customer_directory = pd.read_csv(
            CUSTOMERS_CSV, usecols=["customer_id", "customer_name"])
        customer_names = dict(zip(
            customer_directory["customer_id"].astype(str),
            customer_directory["customer_name"].fillna(""),
        ))
    except (FileNotFoundError, ValueError):
        customer_names = {}

    search_query = st.text_input(
        "Search customer",
        placeholder="Enter name or customer ID",
        key="customer_search",
    ).strip()

    all_customer_ids = list(dict.fromkeys(
        [str(customer_id) for customer_id in bundle["customers"].index]
        + [str(customer_id) for customer_id in customer_names]
    ))
    query = search_query.upper()
    if query:
        if query in [customer_id.upper() for customer_id in all_customer_ids]:
            customer_options = [customer_id for customer_id in all_customer_ids
                               if customer_id.upper() == query]
        else:
            customer_options = [
                customer_id for customer_id in all_customer_ids
                if query in customer_id.upper()
                or query in customer_names.get(customer_id, "").upper()
            ]
    else:
        customer_options = all_customer_ids

    if not customer_options:
        st.info("No customers match this search.")
        st.stop()

    cid = st.selectbox(
        "Select customer",
        customer_options,
        format_func=lambda customer_id: (
            f"{customer_names.get(customer_id, '')} ({customer_id})"
            if customer_names.get(customer_id)
            else customer_id
        ),
        key="selected_customer",
    )
    feature_row, owned, probs, recs = score_existing_customer(bundle, cid)

    st.markdown('<div class="section-title">Customer profile</div>',
                unsafe_allow_html=True)
    customer_name = customer_names.get(cid, "")
    render_metrics([
        ("Customer", customer_name or cid),
        ("Customer ID", cid),
        ("Age", f"{int(feature_row['age']) if pd.notna(feature_row['age']) else '-'}"),
        ("Occupation", str(feature_row["occupation"])),
    ])

    profile_col1, profile_col2, profile_col3 = st.columns(3)
    with profile_col1:
        st.markdown(
            '<div class="panel"><strong>Financial profile</strong>'
            f'<p>Monthly income: INR {feature_row["income"]:,.0f}</p>'
            f'<p>Average balance: INR {feature_row["avg_balance"]:,.0f}</p>'
            f'<p>Monthly spend: INR {feature_row["monthly_spend"]:,.0f}</p>'
            f'<p>Savings rate: {feature_row["savings_rate"]:.1%}</p>'
            f'<p>Credit score: {int(feature_row["credit_score"])}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with profile_col2:
        st.markdown(
            '<div class="panel"><strong>Customer profile</strong>'
            f'<p>Gender: {feature_row["gender"]}</p>'
            f'<p>City tier: {feature_row["city_tier"]}</p>'
            f'<p>Tenure: {feature_row["tenure_years"]:.1f} years</p>'
            f'<p>Dependents: {int(feature_row["num_dependents"])}</p>'
            f'<p>Salary account: {"Yes" if feature_row["has_salary_account"] else "No"}</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with profile_col3:
        st.markdown(
            '<div class="panel"><strong>Digital activity</strong>'
            f'<p>App logins: {int(feature_row["login_count"])}/month</p>'
            f'<p>UPI transactions: {int(feature_row["upi_txn_count"])}/month</p>'
            f'<p>Card transactions: {int(feature_row["card_txn_count"])}/month</p>'
            f'<p>Branch visits: {int(feature_row["branch_visits"])}/month</p>'
            f'<p>Existing EMI ratio: {feature_row["existing_emi_ratio"]:.1%}</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Current products</div>',
                unsafe_allow_html=True)
    if owned:
        st.write(" • ".join(label(product) for product in owned))
    else:
        st.write("No products currently recorded")

    show_results(probs, owned, recs)

    feedback = bundle["feedback_scored"]
    feedback = feedback[feedback["customer_id"] == cid]
    if not feedback.empty:
        st.markdown('<div class="section-title">Customer feedback history</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            feedback[["date", "text", "rating", "pred_sentiment", "topic"]],
            hide_index=True,
            use_container_width=True,
        )

elif page == "Profile simulator":
    st.markdown('<div class="section-title">Create a new customer profile</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    customer_id = col1.text_input("Customer ID", value="")
    customer_name = col2.text_input("Customer name", value="")
    occupation = col3.selectbox(
        "Occupation", ["Salaried", "Self-Employed", "Business",
                       "Student", "Retired", "Homemaker"])
    age = col1.number_input("Age", 18, 90, int(EXAMPLE_PROFILE["age"]))
    income = col2.number_input(
        "Monthly income (INR)", 0, 1_000_000,
        int(EXAMPLE_PROFILE["income"]), 5000)
    credit = col1.slider("Credit score", 300, 900, int(EXAMPLE_PROFILE["credit_score"]))
    balance = col2.number_input(
        "Average balance (INR)", 0, 10_000_000,
        int(EXAMPLE_PROFILE["avg_balance"]), 10000)
    logins = col3.slider("App logins per month", 0, 60, int(EXAMPLE_PROFILE["login_count"]))
    gender = col1.selectbox("Gender", ["Female", "Male", "Other"])
    city_tier = col2.selectbox("City tier", ["Tier-1", "Tier-2", "Tier-3"])
    tenure = col3.number_input(
        "Customer tenure (years)", 0.0, 50.0,
        float(EXAMPLE_PROFILE["tenure_years"]), 0.5)
    owned = st.multiselect("Existing products", PRODUCTS, format_func=label)

    profile = {
        **EXAMPLE_PROFILE,
        "age": age,
        "income": income,
        "occupation": occupation,
        "credit_score": credit,
        "avg_balance": balance,
        "login_count": logins,
        "gender": gender,
        "city_tier": city_tier,
        "tenure_years": tenure,
        "mobile_app_user": int(logins > 0),
    }

    action1, action2 = st.columns(2)
    if action1.button("Generate recommendation", type="primary", use_container_width=True):
        probs, recs = score_features(
            bundle, profile_to_features(profile), owned)
        show_results(probs, owned, recs)
    if action2.button("Add customer", use_container_width=True):
        try:
            saved_customer_id = add_customer(
                customer_id, customer_name, profile, owned)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(
                f"Customer {customer_name.strip()} ({saved_customer_id}) "
                "was added successfully.")
            st.caption("The customer is now available in Customer intelligence.")

else:
    st.markdown('<div class="section-title">Analyze customer feedback</div>',
                unsafe_allow_html=True)
    text = st.text_area(
        "Feedback text",
        "The mobile app keeps crashing and support never picks up the phone.",
        height=130,
    )
    if st.button("Analyze sentiment", type="primary", use_container_width=True):
        if text.strip():
            result = analyse_feedback_text(bundle, text)
            sentiment = result["sentiment"]
            sentiment_class = f"sentiment-{sentiment}"
            left, right = st.columns([0.35, 0.65])
            with left:
                st.markdown(
                    f'<div class="panel"><div class="label">Predicted sentiment</div>'
                    f'<h2 class="{sentiment_class}">{sentiment.upper()}</h2>'
                    f'<p><strong>Topic:</strong> {result["topic"].replace("_", " ").title()}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with right:
                probabilities = pd.Series(result["probabilities"]).sort_values(ascending=False)
                st.bar_chart(probabilities, use_container_width=True)
