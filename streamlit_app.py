import streamlit as st
import pandas as pd
import os
from datetime import datetime
import csv
import uuid
import requests
import numpy as np
from pathlib import Path
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from geopy.geocoders import OpenCage
from geopy.extra.rate_limiter import RateLimiter
from typing import Optional, Tuple
from recommender import recommend_jobs  # Rule-based model
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import (
    KMeans, DBSCAN, MeanShift, OPTICS, SpectralClustering,
    AgglomerativeClustering, Birch, AffinityPropagation
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
import pickle
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Constants ---
INTERACTION_LOG   = "/tmp/user_interactions.csv"
BASE_API_URL      = "http://44.211.129.99:7860"
FLOW_ID           = "6fdd59ed-0109-491b-8576-3bf4932add58"
GEOCODE_API_KEY   = "e16212d2c51a4da288bf22c3dced407d"
CACHE_FILE        = Path("location_cache.csv")
PCA_COMPONENTS    = 50
N_CLUSTERS      = 5

tuned_algorithms = {
    "KMeans": KMeans(n_clusters=N_CLUSTERS, random_state=42),
    "DBSCAN": DBSCAN(eps=1.0, min_samples=4),
    "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=15, min_samples=7),
    "Agglomerative": AgglomerativeClustering(n_clusters=N_CLUSTERS),
    "GMM": GaussianMixture(n_components=N_CLUSTERS, random_state=42),
    "Birch": Birch(n_clusters=N_CLUSTERS),
    "MeanShift": MeanShift(bandwidth=2.0),
    "OPTICS": OPTICS(min_samples=5, xi=0.05),
    "Spectral": SpectralClustering(n_clusters=N_CLUSTERS, affinity="nearest_neighbors"),
    "AffinityProp": AffinityPropagation(damping=0.9, preference=-50),
}

def run_tuned_clustering(job_pca: np.ndarray):
    results = []
    for name, model in tuned_algorithms.items():
        labels = model.fit_predict(job_pca) if hasattr(model, "fit_predict") else model.fit(job_pca).predict(job_pca)
        if len(set(labels)) <= 1:
            results.append((name, -1.0, np.inf, labels))
        else:
            sil = silhouette_score(job_pca, labels)
            db = davies_bouldin_score(job_pca, labels)
            results.append((name, sil, db, labels))
    best = max(results, key=lambda x: x[1])
    return best[0], best[3]

def log_interaction(user_id: str, action: str, details: dict):
    """
    Append interaction data to CSV file.
    :param user_id: Unique user/session id
    :param action: Description of user action (e.g., "Login Attempt", "OTP Verified", "Run Recommendation")
    :param details: Dict of additional relevant info (can be empty)
    """
    filepath = INTERACTION_LOG
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'user_id', 'action', 'details'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user_id': user_id,
            'action': action,
            'details': str(details)
        })

# Load static data
jobs_df = pd.read_csv("jobs.csv")
available_skills = sorted(jobs_df["Job type"].dropna().unique().tolist())
indian_states = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa',
    'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
    'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland',
    'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttarakhand', 'Uttar Pradesh', 'West Bengal', 'Andaman and Nicobar Islands',
    'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu', 'Lakshadweep',
    'Delhi', 'Puducherry'
]

# --- Session State Defaults ---
for key, default in {
    'authenticated': False,
    'page': 'login',
    'login_trigger': 0,
    'user_data': {},
    'recommendations': None,
    'interaction_trigger': 0,
    'generated_otp': None,
    'user_role': 'user',
    'messages': [],
    'session_id': str(uuid.uuid4())
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def run_flow(user_message, session_id, user_name, tweaks=None, api_key=None):
    api_url = f"{BASE_API_URL}/api/v1/run/{FLOW_ID}"
    payload = {
        "session_id": session_id,
        "input_value": user_message,
        "input_type": "chat",
        "output_type": "chat",
        "tweaks": {
            "ChatInput-aAzUo": {"session_id": session_id},
            "TextInput-ujdax": {"input_value": user_message},
            "Memory-YVR39": {"session_id": session_id},
            "ChatOutput-8QykV": {"session_id": session_id},
        }
    }
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
        raise
    except Exception as err:
        print(f"An error occurred: {err}")
        raise


# --- Geocoding & Embedding Helpers ---
@st.cache_resource
def init_geocoder():
    return RateLimiter(OpenCage(api_key=GEOCODE_API_KEY).geocode, min_delay_seconds=1)

@st.cache_resource
def init_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_location_cache() -> dict:
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE).set_index("location")[['lat','lon']].to_dict('index')
    return {}

@st.cache_data
def get_coordinates(name: str, cache: dict) -> Optional[Tuple[float, float]]:
    if pd.isna(name): return None
    if name in cache:
        return (cache[name]['lat'], cache[name]['lon'])
    loc = init_geocoder()(name)
    if loc:
        coords = (loc.latitude, loc.longitude)
        cache[name] = {'lat': coords[0], 'lon': coords[1]}
        pd.DataFrame.from_dict({name: coords}, orient='index', columns=['lat','lon']) \
          .to_csv(CACHE_FILE, mode='a', header=not CACHE_FILE.exists(), index_label='location')
        return coords
    return None

# --- PAGE: LOGIN ---
if st.session_state.page == 'login':
    st.title("🔐 Login")
    phone = st.text_input("Phone")
    name  = st.text_input("Name")

    phone_valid = phone.isdigit() and len(phone) == 10
    name_valid  = bool(name.strip())
    can_send    = phone_valid and name_valid

    if st.button("Send OTP"):
        if not phone_valid:
            st.error("Please enter a valid 10-digit phone number.")
        elif not name_valid:
            st.error("Name cannot be empty.")
        else:
            st.session_state.generated_otp = "123456"
            st.session_state.user_data["name"] = name.strip()
            st.session_state.user_data["phone"] = phone 
            st.success("OTP sent! Use 123456 for demo.")
            # Log OTP send event
            log_interaction(
            user_id=st.session_state.session_id if 'session_id' in st.session_state else 'unknown',
            action="Send OTP",
            details={"phone": phone, "name": name}
        )

    if st.session_state.generated_otp:
        otp = st.text_input("Enter OTP", key="otp_input")
        if st.button("Verify OTP"):
            if otp == st.session_state.generated_otp:
                # Clear previous chat
                st.session_state['messages']   = []
                # st.session_state['session_id'] = str(uuid.uuid4())
                st.session_state["session_id"] = st.session_state.user_data["phone"] 

                st.session_state.authenticated  = True
                st.session_state.user_role      = 'user'
                st.session_state.page           = 'main'
                st.session_state.login_trigger += 1
                st.rerun()
                # Log successful OTP verification
                log_interaction(
                    user_id=st.session_state.session_id,
                    action="OTP Verified",
                    details={"otp_entered": otp, "status": "success"}
                )
            else:
                st.error("Incorrect OTP")
                # Log failed OTP verification
                log_interaction(
                    user_id=st.session_state.session_id if 'session_id' in st.session_state else 'unknown',
                    action="OTP Verified",
                    details={"otp_entered": otp, "status": "failure"}
                )

    st.markdown("---")
    st.subheader("Admin Access")
    admin_email = st.text_input("Work Email", key="admin_email_input")
    if st.button("Admin Access"):
        if admin_email.lower().endswith("@innodatatics.com"):
            st.session_state['messages']   = []
            st.session_state['session_id'] = str(uuid.uuid4())
            st.session_state.authenticated = True
            st.session_state.user_role     = 'admin'
            st.session_state.page          = 'admin_view'
            st.success("Admin access granted.")
            st.rerun()
        else:
            st.error("Access denied. Use a valid email.")

# --- PAGE: MAIN APP ---
elif st.session_state.page == 'main' and st.session_state.authenticated:
    st.title("🧠 AI Job Recommender")

    # Top bar: Logout on left, Chatbot Help on right
    col_left, col_spacer, col_right = st.columns([1, 3, 1])
    with col_left:
        if st.button("Logout"):
            log_interaction(
                user_id=st.session_state.session_id if 'session_id' in st.session_state else 'unknown',
                action="Logout",
                details={}
            )
            for k in ['authenticated', 'generated_otp', 'recommendations', 'user_data']:
                st.session_state[k] = False if isinstance(st.session_state[k], bool) else {}
            st.session_state['messages'] = []
            st.session_state['session_id'] = str(uuid.uuid4())
            st.session_state.page = 'login'
            st.rerun()
    with col_right:
        if st.button("Chatbot Help"):
            st.session_state.page = 'chatbot'
            st.rerun()

    with st.form("user_form"):
        name         = st.text_input("Name", value=st.session_state.user_data.get("name",""))
        age          = st.number_input("Age", 18, 90, value=st.session_state.user_data.get("age", 30))
        location     = st.selectbox("Location (State)", indian_states, index=indian_states.index(st.session_state.user_data.get("location", indian_states[0])))
        skills       = st.multiselect("Skills (Job Types)", available_skills, default=st.session_state.user_data.get("skills", "").split(", ") if st.session_state.user_data.get("skills") else [])
        salary       = st.number_input("Expected Monthly Salary (INR)", min_value=0, value=st.session_state.user_data.get("salary", 0))
        top_n        = st.slider("Number of Recommendations", 1, 20, value=st.session_state.user_data.get("top_n", 3))
        submitted    = st.form_submit_button("Save Profile")

    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
        elif not skills:
            st.error("Please select at least one skill.")
        else:
            st.session_state.user_data = {
                "name": name,
                "age": age,
                "location": location,
                "skills": ", ".join(skills),
                "salary": salary,
                "top_n": top_n,
                "session_id": str(uuid.uuid4()),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success("Profile saved! Now you can proceed to recommendations.")
            # Log profile saved event
            log_interaction(
                user_id=st.session_state.session_id,
                action="Profile Saved",
                details=st.session_state.user_data
            )
            
    # Check if profile is saved with skills (enable buttons only then)
    profile_ready = (
        st.session_state.get("user_data")
        and st.session_state.user_data.get("skills")
        and len(st.session_state.user_data.get("skills")) > 0
    )
     # --- Bottom buttons ---
    col1, col2 = st.columns(2)
    if col1.button("Unsupervised Recommendation"):
        if st.session_state.user_data and st.session_state.user_data.get("skills"):
            st.session_state.page = 'unsupervised'
            st.rerun()
        else:
            st.warning("Please fill your profile in the form before proceeding.")

    if col2.button("Rule-Based Recommendation"):
        if st.session_state.user_data and st.session_state.user_data.get("skills"):
            st.session_state.page = 'rule_based'
            st.rerun()
        else:
            st.warning("Please fill your profile in the form before proceeding.")

elif st.session_state.page == 'rule_based' and st.session_state.authenticated:
    st.title("📋 Rule-Based Job Recommendation")

    if st.button("🔙 Back"):
        st.session_state.page = 'main'
        st.rerun()

    # Sidebar for worker profile
    st.sidebar.header("Worker Profile")
    w_nm = st.sidebar.text_input("Name", st.session_state.user_data.get("name", "John Doe"))
    w_city = st.sidebar.text_input("City", st.session_state.user_data.get("location", "Mumbai"))
    w_skill = st.sidebar.text_input("Skills (comma-separated)", st.session_state.user_data.get("skills", "Plumber"))
    w_sal = st.sidebar.number_input("Monthly Wage (₹)", 0, value=int(st.session_state.user_data.get("salary", 30000)))
    top_n = st.sidebar.slider("Top N", 1, 20, st.session_state.user_data.get("top_n", 5))

    if st.sidebar.button("Run Rule-Based Recommendation"):
        # Prepare skills list
        skills_list = [s.strip() for s in w_skill.split(",") if s.strip()]
        if not w_nm.strip():
            st.sidebar.error("Please enter your name.")
        elif not skills_list:
            st.sidebar.error("Please enter at least one skill.")
        else:
            recs = recommend_jobs(
                user_name=w_nm,
                user_age=st.session_state.user_data.get("age", 30),  # or add age input if needed
                user_location=w_city,
                user_skills=", ".join(skills_list),
                expected_salary=w_sal,
                top_n=top_n
            )
            st.session_state.recommendations = recs
            # Log recommendation request
            log_interaction(
                user_id=st.session_state.session_id,
                action="Run Rule-Based Recommendation",
                details={
                    "user_name": w_nm,
                    "user_age": st.session_state.user_data.get("age", 30),
                    "user_location": w_city,
                    "user_skills": w_skill,
                    "expected_salary": w_sal,
                    "top_n": top_n,
                    "recommendation_count": len(recs) if recs is not None else 0}
                    )

    # Display recommendations if available
    recs = st.session_state.get('recommendations')
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        for idx, row in recs.iterrows():
            with st.expander(f"📌 {row['Company']}"):
                st.write(f"**Job type:** {row['Job type']}")
                st.write(f"**State:** {row['State']}")
                st.write(f"**Match score:** {row['match_score']}")
                if st.button(f"Interested in {row['Company']}", key=f"int_rule_{idx}"):
                    st.success("Your interest has been logged!")
                    log_interaction(
                        user_id=st.session_state.session_id,
                        action="Job Interest Clicked",
                        details={"company": row["Company"], "job_type": row["Job type"], "state": row["State"]}
                    )
    elif recs is not None:
        st.warning("No jobs found matching your profile.")


# --- PAGE: CHATBOT ---
elif st.session_state.page == 'chatbot':
    st.title("💬 InnoDatatics Chat")

    if st.button("🔙 Back to Recommender"):
        st.session_state.page = 'main'
        st.rerun()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Get new user message
    prompt = st.chat_input("Type your message…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # with st.chat_message("user"):
        #     st.markdown(prompt)
      
        # Log user input
        log_interaction(
            user_id=st.session_state.session_id,
            action="Chatbot User Prompt",
            details={"prompt": prompt}
        )

        # Show user message
        # st.session_state.messages.append({"role": "user", "content": prompt})
        # with st.chat_message("user"):
        #     st.markdown(prompt)
        

        # Generate assistant reply
        reply = run_flow(
            user_message=prompt,
            session_id=st.session_state.session_id,
            user_name=st.session_state.user_data.get("name", "")
        )

        # Show assistant reply
        # with st.chat_message("assistant"):
        #     st.markdown(reply)
        # st.session_state.messages.append({"role": "assistant", "content": reply})


        # Log assistant reply
        log_interaction(
            user_id=st.session_state.session_id,
            action="Chatbot Assistant Reply",
            details={"reply": reply}
        )

# --- PAGE: UNSUPERVISED ---

elif st.session_state.page == 'unsupervised':

 
    # Define and ensure model directory exists
    PCA_MODEL_PATH = Path("models/pca_model.pkl")
    CLUSTER_MODEL_PATH = Path("models/cluster_model.pkl")
    PCA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    st.title("🤖 Unsupervised Job Recommendation")
    if st.button("🔙 Back"):
        st.session_state.page = 'main' 
        st.rerun()

    # Sidebar for worker profile
    st.sidebar.header("Worker Profile")
    w_nm = st.sidebar.text_input("Name", st.session_state.user_data.get("name", "John Doe"))
    w_city = st.sidebar.text_input("City", st.session_state.user_data.get("location", "Mumbai"))
    w_skill = st.sidebar.text_input("Skills (comma-separated)", st.session_state.user_data.get("skills", "Plumber"))
    w_sal = st.sidebar.number_input("Monthly Wage (₹)", 0, value=int(st.session_state.user_data.get("salary", 30000)))
    top_n = st.sidebar.slider("Top N", 1, 20, st.session_state.user_data.get("top_n", 5))
    run_btn = st.sidebar.button("Run Unsupervised")

        # --- Session State Setup ---
    if "interested_unsup" not in st.session_state:
        st.session_state.interested_unsup = set()
    if "unsup_top_jobs" not in st.session_state:
        st.session_state.unsup_top_jobs = []
    if "run_unsup_done" not in st.session_state:
        st.session_state.run_unsup_done = False
 
    # --- Run the Model ---
    if run_btn or st.session_state.run_unsup_done:
        if run_btn:
            st.session_state.run_unsup_done = True  # Mark the logic as completed once

            df_uns = jobs_df.copy()
            df_uns['Avg_salary'] = (df_uns['Min salary'] + df_uns['Max salary']) / 2
            mean_sal = df_uns.loc[df_uns['Avg_salary'] != 0, 'Avg_salary'].mean()
            df_uns['Avg_salary'].replace(0, mean_sal, inplace=True)
            df_uns['job_text'] = df_uns['Job type'] + " role in " + df_uns['State'] + ". Avg ₹" + df_uns['Avg_salary'].astype(int).astype(str)

            mdl = init_model()
            emb = mdl.encode(df_uns['job_text'].tolist(), show_progress_bar=False)
            scaler = MinMaxScaler().fit(emb)
            emb_s = scaler.transform(emb)

            if PCA_MODEL_PATH.exists():
                with open(PCA_MODEL_PATH, 'rb') as f:
                    pca = pickle.load(f)
            else:
                pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
                pca.fit(emb_s)
                with open(PCA_MODEL_PATH, 'wb') as f:
                    pickle.dump(pca, f)

            job_pca = pca.transform(emb_s)

            if CLUSTER_MODEL_PATH.exists():
                with open(CLUSTER_MODEL_PATH, 'rb') as f:
                    clustering_model = pickle.load(f)
                    labels = clustering_model.labels_ if hasattr(clustering_model, "labels_") else clustering_model.predict(job_pca)
                    algo_name = type(clustering_model).__name__
            else:
                algo_name, labels = run_tuned_clustering(job_pca)
                clustering_model = tuned_algorithms[algo_name]
                clustering_model.fit(job_pca)
                with open(CLUSTER_MODEL_PATH, 'wb') as f:
                    pickle.dump(clustering_model, f)

            df_uns['cluster'] = labels
            st.success(f"Best algorithm: {algo_name}")

            skill_texts = [f"{sk.strip()} seeking role in {w_city}" for sk in w_skill.split(',')]
            skill_emb = mdl.encode(skill_texts, show_progress_bar=False)
            emb_w = scaler.transform(skill_emb)
            w_pca = pca.transform(emb_w).mean(axis=0).reshape(1, -1)
            dists = np.linalg.norm(job_pca - w_pca, axis=1)
            worker_cl = int(df_uns.loc[dists.argmin(), 'cluster'])
            st.write(f"Worker assigned to cluster **{worker_cl}**")
            worker_emb = skill_emb.mean(axis=0).reshape(1, -1)
            sims = cosine_similarity(worker_emb, emb).flatten()
            df_uns['sim'] = sims
            subset = df_uns[df_uns['cluster'] == worker_cl]
            top_jobs = subset.nlargest(top_n, 'sim')

            st.session_state.unsup_top_jobs = top_jobs.to_dict('records')
            st.session_state.worker_cluster = worker_cl

        # --- Display Results ---
        st.subheader(f"Top {top_n} jobs in cluster {st.session_state.worker_cluster}")
        for idx, row in enumerate(st.session_state.unsup_top_jobs):
            with st.expander(f"📌 {row['Company']}"):
                st.write(f"**Job type:** {row['Job type']}")
                st.write(f"**State:** {row['State']}")
                st.write(f"**Similarity score:** {row['sim']:.2f}")

                job_key = f"{row['Company']}_{row['Job type']}_{row['State']}"

                if job_key not in st.session_state.interested_unsup:
                    if st.button(f"Interested in {row['Company']}", key=f"int_unsup_{idx}"):
                        st.session_state.interested_unsup.add(job_key)
                        log_interaction(
                            user_id=st.session_state.session_id,
                            action="Job Interest Clicked (Unsupervised)",
                            details={
                                "company": row["Company"],
                                "job_type": row["Job type"],
                                "state": row["State"],
                                "similarity_score": row["sim"]
                            }
                        )
                        st.success("Your interest has been logged!")
                        st.rerun()  # force re-render to show updated interest
                else:
                    st.info("✅ Interest already logged.")

# --- PAGE: ADMIN VIEW ---
elif st.session_state.page == 'admin_view' and st.session_state.authenticated:
    st.title("🛠 Admin Panel")
    if st.button("Logout"):
        st.session_state.authenticated  = False
        st.session_state.user_role      = 'user'
        st.session_state.page           = 'login'
        st.session_state['messages']    = []
        st.session_state['session_id']  = str(uuid.uuid4())
        st.rerun()

    action = st.radio("Choose action:", ["View Dashboard", "Download Interaction Data", "Append to jobs.csv"])
    if action == "View Dashboard":
        try:
            exec(open("dashboard10.py").read())
        except Exception as e:
            st.error(f"Failed to load dashboard: {e}")
          
    elif action == "Download Interaction Data":
        st.subheader("📄 Download Interaction Logs")
        if os.path.exists(INTERACTION_LOG):
            with open(INTERACTION_LOG, 'rb') as f:
                st.download_button(
                    label="📥 Download user_interactions.csv",
                    data=f,
                    file_name="user_interactions.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No interaction data found.")
        
    elif action == "Append to jobs.csv":
        st.subheader("📥 Download jobs.csv Template (Headers Only)")
        if os.path.exists("jobs.csv"):
            try:
                # Read only headers, no rows
                df = pd.read_csv("jobs.csv", nrows=0)
                csv_buffer = df.to_csv(index=False)
                st.download_button(
                    label="Download jobs_template.csv",
                    data=csv_buffer,
                    file_name="jobs_template.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error preparing template: {e}")
        else:
            st.warning("jobs.csv not found. Cannot create template.")
        
        st.subheader("⬆️ Upload new jobs CSV to append")
        uploaded_file = st.file_uploader("Upload CSV file with new jobs", type=["csv"])
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                if st.button("Append Job(s)"):
                    existing = pd.read_csv("jobs.csv") if os.path.exists("jobs.csv") else pd.DataFrame()
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined.to_csv("jobs.csv", index=False)
                    st.success(f"Appended {len(new_df)} job(s) successfully.")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
    
