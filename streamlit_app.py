# # # import streamlit as st
# # # import pandas as pd
# # # import os
# # # import time
# # # from datetime import datetime
# # # import uuid
# # # import sqlite3
# # # import bcrypt
# # # import requests
# # # from recommender import recommend_jobs

# # # # ---------- Constants ----------
# # # INTERACTION_LOG = "/tmp/user_interactions.csv"
# # # BASE_API_URL = "http://127.0.0.1:7860"
# # # FLOW_ID = "1a520f57-f6b0-46b3-aa50-9041f4d83e60"

# # # # ---------- Load Jobs Data ----------
# # # jobs_df = pd.read_csv("jobs.csv")
# # # available_skills = sorted(jobs_df["Job type"].dropna().unique().tolist())
# # # indian_states = [
# # #      'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
# # #      'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
# # #      'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
# # #      'Tripura', 'Uttarakhand', 'Uttar Pradesh', 'West Bengal', 'Andaman and Nicobar Islands', 'Chandigarh',
# # #      'Dadra and Nagar Haveli and Daman and Diu', 'Lakshadweep', 'Delhi', 'Puducherry'
# # #  ]

# # # # ---------- Session State Initialization ----------
# # # def init_session():
# # #     keys = {
# # #         'authenticated': False,
# # #         'page': 'login',
# # #         'login_trigger': 0,
# # #         'user_data': {},
# # #         'recommendations': None,
# # #         'interaction_trigger': 0,
# # #         'generated_otp': None,
# # #         'user_id': None,
# # #         'user_name': None,
# # #         'session_id': None,
# # #         'messages': []
# # #     }
# # #     for k, v in keys.items():
# # #         if k not in st.session_state:
# # #             st.session_state[k] = v

# # # init_session()

# # # # ---------- Chatbot Helper ----------
# # # def run_flow(user_message, session_id, user_name, tweaks=None, api_key=None):
# # #     api_url = f"{BASE_API_URL}/api/v1/run/{FLOW_ID}"
# # #     payload = {
# # #         "session_id": session_id,
# # #         "input_value": user_message,
# # #         "input_type": "chat",
# # #         "output_type": "chat",
# # #         "tweaks": tweaks or {}
# # #     }
# # #     payload["tweaks"].update({
# # #         "ChatInput-voquK": {"session_id": session_id},
# # #         "TextInput-s26mJ": {"input_value": user_name},
# # #         "Memory-rRETL": {"session_id": session_id},
# # #         "ChatOutput-GFauW": {"session_id": session_id}
# # #     })
# # #     headers = {"x-api-key": api_key} if api_key else None
# # #     response = requests.post(api_url, json=payload, headers=headers)
# # #     response.raise_for_status()
# # #     return response.json()["outputs"][0]["outputs"][0]["results"]["message"]["text"]

# # # # ---------- Routing ----------
# # # if st.session_state.page == 'login':
# # #     st.title("🔐 Login")
# # #     phone_number = st.text_input("Enter your phone number")
# # #     if phone_number and (not phone_number.isdigit() or len(phone_number) != 10):
# # #         st.error("Please enter a valid 10-digit phone number.")
# # #     valid_phone = phone_number.isdigit() and len(phone_number) == 10
# # #     if st.button("Send OTP", disabled=not valid_phone):
# # #         st.session_state.generated_otp = "123456"
# # #         st.success("OTP sent! Use 123456 for demo.")
# # #     if st.session_state.generated_otp:
# # #         entered_otp = st.text_input("Enter OTP", key="entered_otp")
# # #         if st.button("Verify OTP"):
# # #             if entered_otp == st.session_state.generated_otp:
# # #                 st.session_state.authenticated = True
# # #                 st.session_state.page = 'main'
# # #                 st.session_state.session_id = phone_number
# # #                 # Set user_name and clear chat history for new user session
# # #                 st.session_state.user_name = phone_number
# # #                 st.session_state.messages = []
# # #                 st.rerun()
# # #             else:
# # #                 st.error("Incorrect OTP")

# # # elif st.session_state.page == 'main' and st.session_state.authenticated:
# # #     st.title("🧠 AI Job Recommender")
# # #     if st.button("Logout"):
# # #         for k in ('authenticated', 'generated_otp', 'recommendations', 'user_data'):
# # #             st.session_state[k] = None
# # #         st.session_state.page = 'login'
# # #         st.rerun()
# # #     if st.button("💬 Go to Chatbot"):
# # #         st.session_state.page = 'chatbot'
# # #         st.rerun()

# # #     with st.form("user_input_form"):
# # #         name = st.text_input("Name")
# # #         age = st.number_input("Age", min_value=18, max_value=70, value=25)
# # #         location = st.selectbox("Select Your Location (State)", indian_states)
# # #         skills = st.multiselect("Select Skills (Job Types)", available_skills)
# # #         salary = st.number_input("Expected Monthly Salary (INR)", min_value=0)
# # #         top_n = st.slider("Number of Job Recommendations", 1, 10, 3)
# # #         submitted = st.form_submit_button("Get Recommendations")

# # #     if submitted:
# # #         if not name.strip():
# # #             st.error("Please enter your name.")
# # #         elif not skills:
# # #             st.error("Please select at least one skill.")
# # #         else:
# # #             with st.spinner('Generating recommendations...'):
# # #                 session_id = str(uuid.uuid4())
# # #                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# # #                 skills_str = ", ".join(skills)
# # #                 st.session_state.user_data = {
# # #                     "name": name, "age": age, "location": location,
# # #                     "skills": skills_str, "salary": salary, "top_n": top_n,
# # #                     "session_id": session_id, "timestamp": timestamp
# # #                 }
# # #                 st.session_state.recommendations = recommend_jobs(
# # #                     user_name=name, user_age=age, user_location=location,
# # #                     user_skills=skills_str, expected_salary=salary, top_n=top_n
# # #                 )

# # #     if st.session_state.recommendations is not None:
# # #         if not st.session_state.recommendations.empty:
# # #             st.write(f"🔍 Recommendations for **{st.session_state.user_data['name']}**:")
# # #             recs = st.session_state.recommendations[['Company', 'Job type', 'State', 'match_score']].sort_values(by="match_score", ascending=False)
# # #             for index, row in recs.iterrows():
# # #                 job_key = f"expander_{index}"
# # #                 with st.expander(f"📌 {row['Company']}"):
# # #                     st.write(f"**Job Type**: {row['Job type']}")
# # #                     st.write(f"**Location**: {row['State']}")
# # #                     st.write(f"**Match Score**: {row['match_score']}")
# # #                     if st.button(f"I'm interested in {row['Company']}", key=f"button_{job_key}"):
# # #                         interaction_data = {
# # #                             "Timestamp": st.session_state.user_data['timestamp'],
# # #                             "Session ID": st.session_state.user_data['session_id'],
# # #                             "Name": st.session_state.user_data['name'],
# # #                             "Age": st.session_state.user_data['age'],
# # #                             "Location": st.session_state.user_data['location'],
# # #                             "Skills": st.session_state.user_data['skills'],
# # #                             "Expected Salary": st.session_state.user_data['salary'],
# # #                             "Top N": st.session_state.user_data['top_n'],
# # #                             "Jobs Recommended": "|".join(st.session_state.recommendations["Company"].astype(str).tolist()),
# # #                             "Jobs Clicked": row['Company'],
# # #                             "Match Scores": "|".join(st.session_state.recommendations["match_score"].astype(str).tolist())
# # #                         }
# # #                         if os.path.exists(INTERACTION_LOG):
# # #                             df_log = pd.read_csv(INTERACTION_LOG)
# # #                             df_log = pd.concat([df_log, pd.DataFrame([interaction_data])], ignore_index=True)
# # #                         else:
# # #                             df_log = pd.DataFrame([interaction_data])
# # #                         df_log.to_csv(INTERACTION_LOG, index=False)
# # #                         st.success(f"Your interest in {row['Company']} has been logged!")
# # #         else:
# # #             st.warning("⚠️ No jobs found matching your profile.")

# # # elif st.session_state.page == 'chatbot':
# # #     st.set_page_config(page_title="InnoDatatics Chat", layout="wide")
# # #     st.title("💬 InnoDatatics Chat")

# # #     if st.button("🔙 Back to Recommender"):
# # #         st.session_state.page = 'main'
# # #         st.rerun()

# # #     for msg in st.session_state.messages:
# # #         with st.chat_message(msg["role"]):
# # #             st.markdown(msg["content"])

# # #     if prompt := st.chat_input("Type your message…"):
# # #         st.session_state.messages.append({"role": "user", "content": prompt})
# # #         with st.chat_message("assistant"):
# # #             reply = run_flow(
# # #                 user_message=prompt,
# # #                 session_id=st.session_state.session_id,
# # #                 user_name=st.session_state.user_name
# # #             )
# # #             st.markdown(reply)
# # #             st.session_state.messages.append({"role": "assistant", "content": reply})

# # import streamlit as st
# # import pandas as pd
# # import os
# # import time
# # from datetime import datetime
# # import uuid
# # import requests
# # from recommender import recommend_jobs


# # # 🔧 Set constant layout and page width for all pages
# # st.set_page_config(
# #     page_title="Job Recommender System",
# #     layout="centered",  # Use "wide" if you want full screen width
# #     initial_sidebar_state="auto"
# # )

# # # Optional: CSS for fixed width and spacing
# # st.markdown("""
# #     <style>
# #         .main {
# #             max-width: 720px;
# #             margin: 0 auto;
# #         }
# #         .block-container {
# #             padding-top: 2rem;
# #             padding-bottom: 2rem;
# #         }
# #     </style>
# # """, unsafe_allow_html=True)

# # # Full CSV file path for interaction logs
# # # need to set up google api - using google console to store interaction data on google sheets
# # INTERACTION_LOG = "/tmp/user_interactions.csv"

# # # LangFlow API constants
# # BASE_API_URL = "http://127.0.0.1:7860"
# # FLOW_ID = "1a520f57-f6b0-46b3-aa50-9041f4d83e60"

# # # Load jobs data
# # jobs_df = pd.read_csv("C://Users//Tanmay//Downloads//jobs.csv")

# # # Extract unique job types (skills) from 'Job type' column
# # available_skills = sorted(jobs_df["Job type"].dropna().unique().tolist())

# # # List of all Indian states
# # indian_states = [
# #     'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
# #     'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland',
# #     'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttarakhand', 'Uttar Pradesh', 'West Bengal',
# #     'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu', 'Lakshadweep', 'Delhi', 'Puducherry'
# # ]

# # # Initialize session variables
# # for key, value in {
# #     'authenticated': False,
# #     'page': 'login',
# #     'login_trigger': 0,
# #     'user_data': {},
# #     'recommendations': None,
# #     'interaction_trigger': 0,
# #     'generated_otp': None,
# #     'user_role': 'user',
# #     'messages': [],               # for chatbot history
# #     'session_id': str(uuid.uuid4())  # for logging user sessions
# # }.items():
# #     if key not in st.session_state:
# #         st.session_state[key] = value


# # # ---------- LangFlow chatbot integration ----------

# # def run_flow(user_message, session_id, user_name, tweaks=None, api_key=None):
# #     api_url = f"{BASE_API_URL}/api/v1/run/{FLOW_ID}"
# #     payload = {
# #         "session_id": session_id,
# #         "input_value": user_message,
# #         "input_type": "chat",
# #         "output_type": "chat",
# #         "tweaks": tweaks or {}
# #     }
# #     # Pass context nodes for LangFlow components
# #     payload["tweaks"].update({
# #         "ChatInput-voquK": {"session_id": session_id},
# #         "TextInput-s26mJ": {"input_value": user_name},
# #         "Memory-rRETL": {"session_id": session_id},
# #         "ChatOutput-GFauW": {"session_id": session_id}
# #     })
# #     headers = {"x-api-key": api_key} if api_key else {}
# #     response = requests.post(api_url, json=payload, headers=headers)
# #     response.raise_for_status()
# #     return response.json()["outputs"][0]["outputs"][0]["results"]["message"]["text"]


# # # -------------- LOGIN PAGE --------------
# # if st.session_state.page == 'login':
# #     # Regular user login form
# #     st.title("🔐 Login")
# #     phone_number = st.text_input("Enter your phone number")
# #     name_input = st.text_input("Enter your name")

# #     valid_phone = False

# #     # Validate phone number length and digits
# #     if phone_number:
# #         if not phone_number.isdigit() or len(phone_number) != 10:
# #             st.error("Please enter a valid 10-digit phone number.")
# #         else:
# #             valid_phone = True

# #     valid_name = bool(name_input and name_input.strip())  # True if name is not empty or whitespace only

# #     send_otp_button = st.button("Send OTP", disabled=not (valid_phone and valid_name))

# #     if send_otp_button:
# #         st.session_state.generated_otp = "123456"
# #         st.session_state.user_data["name"] = name_input.strip()
# #         st.success("OTP sent! Use 123456 for demo.")

# #     if st.session_state.generated_otp:
# #         entered_otp = st.text_input("Enter OTP", key="entered_otp")
# #         if st.button("Verify OTP"):
# #             if entered_otp == st.session_state.generated_otp:
# #                 st.session_state.authenticated = True
# #                 st.session_state.user_role = "user"
# #                 st.session_state.page = 'main'  # Switch to main app page
# #                 st.session_state.login_trigger += 1
# #                 st.rerun()
# #             else:
# #                 st.error("Incorrect OTP")
                
# #     st.markdown("---")

# #     st.subheader("Admin Access")
# #     admin_email = st.text_input("Enter your work email", key="admin_email_input")
# #     if st.button("Admin Access"):
# #         if admin_email.lower().endswith("@innodatatics.com"):
# #             st.session_state.authenticated = True
# #             st.session_state.user_role = "admin"
# #             st.session_state.page = "admin_view"  # Redirect to admin view page
# #             st.success("Admin access granted.")
# #             st.rerun()
# #         else:
# #             st.error("Access denied. Please use a valid email.")

# # # -------------- MAIN APP PAGE --------------
# # elif st.session_state.page == 'main' and st.session_state.authenticated:
# #     st.title("🧠 AI Job Recommender")

# #     if st.button("Logout"):
# #         for key in ['authenticated', 'generated_otp', 'recommendations', 'user_data']:
# #             st.session_state[key] = False if isinstance(st.session_state[key], bool) else {}
# #         st.session_state.page = 'login'
# #         st.rerun()

# #     if st.button("Chatbot Help"):
# #         st.session_state.page = "chatbot"
# #         st.rerun()

# #     # User input form
# #     with st.form("user_input_form"):
# #         name = st.text_input("Name", value=st.session_state.user_data.get("name", ""))
# #         age = st.number_input("Age", min_value=18, max_value=90, value=30)
# #         location = st.selectbox("Select Your Location (State)", indian_states)
# #         skills = st.multiselect("Select Skills (Job Types)", available_skills)
# #         salary = st.number_input("Expected Monthly Salary (INR)", min_value=0)
# #         model_choice = st.selectbox("Choose Recommendation Model", ["Rule-Based", "Unsupervised"])
# #         top_n = st.slider("Number of Job Recommendations", 1, 20, 3)
# #         submitted = st.form_submit_button("Get Recommendations")

# #     # Handle recommendation generation with basic validation
# #     if submitted:
# #         if not name.strip():
# #             st.error("Please enter your name.")
# #         elif not skills or len(skills) == 0:
# #             st.error("Please select at least one skill.")
# #         else:
# #             with st.spinner('Generating recommendations...'):
# #                 session_id = str(uuid.uuid4())
# #                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# #                 skills_str = ", ".join(skills) if isinstance(skills, list) else skills
# #                 st.session_state.user_data = {
# #                     "name": name,
# #                     "age": age,
# #                     "location": location,
# #                     "skills": skills_str,
# #                     "salary": salary,
# #                     "top_n": top_n,
# #                     "session_id": session_id,
# #                     "timestamp": timestamp
# #                 }

# #                 if model_choice == "Rule-Based":
# #                     st.session_state.recommendations = recommend_jobs(
# #                         user_name=name,
# #                         user_age=age,
# #                         user_location=location,
# #                         user_skills=skills_str,
# #                         expected_salary=salary,
# #                         top_n=top_n
# #                     )
# #                 else:
# #                     st.session_state.recommendations = recommend_unsupervised(
# #                         skills=skills_str,  # Convert list to comma-separated string
# #                         location=location,
# #                         expected_monthly_salary=salary,
# #                         top_n=top_n
# #                     )
                
                
# #     # Display recommendations if available
# #     if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
# #         st.write(f"🔍 Recommendations for **{st.session_state.user_data['name']}**:")
# #         recommendations_display = st.session_state.recommendations[['Company', 'Job type', 'State', 'match_score']].sort_values(by="match_score", ascending=False)

# #         for index, row in recommendations_display.iterrows():
# #             job_key = f"expander_{index}"
# #             is_open = st.session_state.get("last_clicked_job") == job_key
        
# #             with st.expander(f"📌 {row['Company']}", expanded=is_open):
# #                 st.write(f"**Job Type**: {row['Job type']}")
# #                 st.write(f"**Location**: {row['State']}")
# #                 st.write(f"**Match Score**: {row['match_score']}")
        
# #                 if st.button(f"I'm interested in {row['Company']}", key=f"button_{job_key}"):
# #                     st.session_state["last_clicked_job"] = job_key
# #                     st.session_state["clicked_job"] = job_key  #
        
# #                     # Log interaction
# #                     interaction_data = {
# #                         "Timestamp": st.session_state.user_data['timestamp'],
# #                         "Session ID": st.session_state.user_data['session_id'],
# #                         "Name": st.session_state.user_data['name'],
# #                         "Age": st.session_state.user_data['age'],
# #                         "Location": st.session_state.user_data['location'],
# #                         "Skills": st.session_state.user_data['skills'],
# #                         "Expected Salary": st.session_state.user_data['salary'],
# #                         "Top N": st.session_state.user_data['top_n'],
# #                         "Jobs Recommended": "|".join(st.session_state.recommendations["Company"].astype(str).tolist()),
# #                         "Jobs Clicked": row['Company'],
# #                         "Match Scores": "|".join(st.session_state.recommendations["match_score"].astype(str).tolist()),
# #                     }
        
# #                     # Save to CSV
# #                     if os.path.exists(INTERACTION_LOG):
# #                         df_log = pd.read_csv(INTERACTION_LOG)
# #                         df_log = pd.concat([df_log, pd.DataFrame([interaction_data])], ignore_index=True)
# #                     else:
# #                         df_log = pd.DataFrame([interaction_data])
        
# #                     df_log.to_csv(INTERACTION_LOG, index=False)
# #                     st.session_state['interaction_trigger'] = st.session_state.get('interaction_trigger', 0) + 1
# #                     st.rerun()  # 🔄 Trigger page refresh after saving
        
# #                 # After rerun, show success if this was the clicked job
# #                 if st.session_state.get('clicked_job') == job_key:
# #                     st.success(f"Your interest in {row['Company']} has been logged!")
        
# #     elif st.session_state.recommendations is not None and st.session_state.recommendations.empty:
# #         st.warning("⚠️ No jobs found matching your profile.")


# # # ---------- PAGE: CHATBOT ----------
# # elif st.session_state.page == 'chatbot':
# #     st.title("💬 InnoDatatics Chat")

# #     if st.button("🔙 Back to Recommender"):
# #         st.session_state.page = 'main'
# #         st.rerun()

# #     if "messages" not in st.session_state:
# #         st.session_state.messages = []

# #     # Show previous messages
# #     for msg in st.session_state.messages:
# #         with st.chat_message(msg["role"]):
# #             st.markdown(msg["content"])

# #     # Handle new user input
# #     if prompt := st.chat_input("Type your message…"):
# #         st.session_state.messages.append({"role": "user", "content": prompt})

# #         with st.chat_message("assistant"):
# #             with st.spinner("Thinking..."):
# #                 reply = run_flow(
# #                     user_message=prompt,
# #                     session_id=st.session_state.session_id,
# #                     user_name=st.session_state.user_data.get("name", "")
# #                 )
# #             st.markdown(reply)
# #             st.session_state.messages.append({"role": "assistant", "content": reply})
             
# # # ----------- ADMIN VIEW PAGE -----------
# # elif st.session_state.page == "admin_view" and st.session_state.authenticated and st.session_state.user_role == "admin":
# #     st.title("🛠 Admin View")

# #     if st.button("Logout"):
# #         st.session_state.authenticated = False
# #         st.session_state.page = 'login'
# #         st.session_state.user_role = 'user'
# #         st.rerun()

# #     selected_action = st.radio("Choose an action:", [
# #         "View Dashboard",
# #         "Download Interaction Data",
# #         "Append to jobs.csv"
# #     ])

# #     if selected_action == "View Dashboard":
# #         st.info("📊 Show some Streamlit charts or tables here.")


# #     elif selected_action == "Download Interaction Data":
# #         if os.path.exists(INTERACTION_LOG):
# #             interaction_df = pd.read_csv(INTERACTION_LOG)
# #             csv = interaction_df.to_csv(index=False).encode("utf-8")
# #             st.download_button("Download Interaction CSV", csv, "interactions.csv", "text/csv")
# #         else:
# #             st.warning("No interaction data found.")

# #     elif selected_action == "Append to jobs.csv":
# #         st.info("📥 Add your job appending logic here.")
# #         new_job = st.text_area("Paste new job data (CSV format)")
# #         if st.button("Append Job"):
# #             if new_job:
# #                 try:
# #                     from io import StringIO
# #                     new_df = pd.read_csv(StringIO(new_job))
# #                     existing_df = pd.read_csv("jobs.csv")
# #                     combined = pd.concat([existing_df, new_df], ignore_index=True)
# #                     combined.to_csv("jobs.csv", index=False)
# #                     st.success("New job(s) added successfully.")
# #                 except Exception as e:
# #                     st.error(f"Error: {e}")
# #             else:
# #                 st.warning("Please enter job data first.")


# import streamlit as st
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import warnings
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler
# from sentence_transformers import SentenceTransformer
# from geopy.distance import geodesic
# from geopy.geocoders import OpenCage
# from geopy.extra.rate_limiter import RateLimiter
# from typing import Optional, Tuple

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # Configuration
# PCA_COMPONENTS = 50
# GEOCODE_API_KEY = "e16212d2c51a4da288bf22c3dced407d"
# CACHE_FILE = Path("location_cache.csv")

# # App title
# st.set_page_config(page_title="Unsupervised Recommender", layout="centered")
# st.title("🤖 Unsupervised Job Recommendations")

# @st.cache_data
# def load_data(file_path: str) -> pd.DataFrame:
#     df = pd.read_excel(file_path, engine="openpyxl")
#     df["Avg_salary"] = (df["Min salary"] + df["Max salary"]) / 2
#     mean_sal = df.loc[df["Avg_salary"] != 0, "Avg_salary"].mean()
#     df["Avg_salary"] = df["Avg_salary"].replace(0, mean_sal)
#     return df

# @st.cache_resource
# def init_geocoder():
#     geolocator = OpenCage(api_key=GEOCODE_API_KEY)
#     return RateLimiter(geolocator.geocode, min_delay_seconds=1)

# @st.cache_resource
# def init_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_data
# def load_location_cache() -> dict:
#     if CACHE_FILE.exists():
#         return pd.read_csv(CACHE_FILE).set_index("location")[['lat','lon']].to_dict('index')
#     return {}

# @st.cache_data
# def get_coordinates(name: str, cache: dict) -> Optional[Tuple[float, float]]:
#     if pd.isna(name):
#         return None
#     if name in cache:
#         return (cache[name]['lat'], cache[name]['lon'])
#     loc = init_geocoder()(name)
#     if loc:
#         coords = (loc.latitude, loc.longitude)
#         cache[name] = {'lat': coords[0], 'lon': coords[1]}
#         pd.DataFrame.from_dict({name: coords}, orient='index', columns=['lat','lon']) \
#             .to_csv(CACHE_FILE, mode='a', header=not CACHE_FILE.exists(), index_label='location')
#         return coords
#     return None

# @st.cache_data
# def build_job_text(df: pd.DataFrame) -> pd.DataFrame:
#     df["job_text"] = (
#         df["Job type"]
#         + " role in "
#         + df["State"]
#         + ".\nAverage salary: ₹"
#         + df["Avg_salary"].astype(int).astype(str).str.replace(r"(\d)(?=(\d{3})+$)", r"\1,", regex=True)
#         + "."
#     )
#     return df

# @st.cache_data
# def generate_embeddings(texts: list, _model) -> np.ndarray:
#     return _model.encode(texts, show_progress_bar=False, batch_size=64)

# @st.cache_data
# def reduce_dimensions(emb: np.ndarray, n_components: int = PCA_COMPONENTS) -> np.ndarray:
#     scaler = MinMaxScaler()
#     scaled = scaler.fit_transform(emb)
#     pca = PCA(n_components=n_components, random_state=42)
#     return pca.fit_transform(scaled)

# # Sidebar input form
# st.sidebar.header("Worker Profile")
# w_name    = st.sidebar.text_input("Name", "John Doe")
# w_city    = st.sidebar.text_input("City", "Mumbai")
# w_state   = st.sidebar.text_input("State", "Maharashtra")
# w_skills  = st.sidebar.text_input("Skills (comma-separated)", "Plumber")
# w_salary  = st.sidebar.number_input("Monthly Wage (₹)", min_value=0, value=30000)
# top_n     = st.sidebar.slider("Top N Recommendations", 1, 20, 5)
# run_btn   = st.sidebar.button("Recommend Jobs")

# if run_btn:
#     jobs_df = load_data("C://Users//Tanmay//Downloads//Data_Innodatatics1-Data_Innodatatics1.xlsx.xlsx")
#     jobs_df = build_job_text(jobs_df)
#     cache = load_location_cache()
#     jobs_df["coords"] = jobs_df["State"].apply(lambda s: get_coordinates(s, cache))

#     model = init_embedding_model()
#     job_emb = generate_embeddings(jobs_df["job_text"].tolist(), model)

#     skill_texts = [f"{sk.strip()} seeking role in {w_city}" for sk in w_skills.split(",")]
#     w_emb = generate_embeddings(skill_texts, model).mean(axis=0).reshape(1, -1)

#     sims = cosine_similarity(w_emb, job_emb).flatten()
#     compatible = jobs_df[(jobs_df["Min salary"] <= w_salary) & (jobs_df["Max salary"] >= w_salary)].copy()

#     w_coord = get_coordinates(w_city, cache)
#     results = []
#     for idx, row in compatible.iterrows():
#         dist = (geodesic(row['coords'], w_coord).km if row['coords'] and w_coord else np.inf)
#         results.append((idx, sims[idx], dist))

#     if not results:
#         st.warning("No matching jobs found for the given salary range.")
#     else:
#         arr = np.array([[r[1], r[2]] for r in results])
#         norm_sim = MinMaxScaler().fit_transform(arr[:, 0].reshape(-1, 1))
#         norm_dist = MinMaxScaler().fit_transform(arr[:, 1].reshape(-1, 1))
#         scores = 0.7 * norm_sim - 0.3 * norm_dist

#         top = sorted(
#             [(res[0], res[1], res[2], float(norm_sim[i]), float(norm_dist[i]), float(scores[i]))
#              for i, res in enumerate(results)], key=lambda x: -x[5]
#         )[:top_n]

#         out = []
#         for idx, sim, dist, _, _, score in top:
#             row = jobs_df.loc[idx]
#             out.append({
#                 "Job Description": row["job_text"],
#                 "Expected Salary": f"{int(row['Min salary'])} - {int(row['Max salary'])}₹",
#                 "Distance (km)": round(dist, 1),
#                 "Match Score": round(score, 3),
#                 "Details": f"Sim: {sim:.2f}, Dist: {dist:.1f} km"
#             })
#         st.subheader(f"Top {top_n} Job Recommendations for {w_name}")
#         st.dataframe(pd.DataFrame(out))

# # Footer
# st.markdown("---")
# st.markdown("Powered by Streamlit & Sentence Transformers")



# import streamlit as st
# import pandas as pd
# import os
# import time
# from datetime import datetime
# import uuid
# import requests
# import numpy as np
# from pathlib import Path
# import warnings
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler
# from sentence_transformers import SentenceTransformer
# from geopy.distance import geodesic
# from geopy.geocoders import OpenCage
# from geopy.extra.rate_limiter import RateLimiter
# from typing import Optional, Tuple
# from recommender import recommend_jobs  # Your rule-based model

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # --- Constants ---
# INTERACTION_LOG = "/tmp/user_interactions.csv"
# BASE_API_URL = "http://127.0.0.1:7860"
# FLOW_ID = "1a520f57-f6b0-46b3-aa50-9041f4d83e60"
# GEOCODE_API_KEY = "e16212d2c51a4da288bf22c3dced407d"
# CACHE_FILE = Path("location_cache.csv")
# PCA_COMPONENTS = 50

# # Load job data
# jobs_df = pd.read_csv("C://Users//Tanmay//Downloads//jobs.csv")
# available_skills = sorted(jobs_df["Job type"].dropna().unique().tolist())

# indian_states = [
#     'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
#     'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
#     'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
#     'Telangana', 'Tripura', 'Uttarakhand', 'Uttar Pradesh', 'West Bengal', 'Andaman and Nicobar Islands',
#     'Chandigarh', 'Dadra and Nagar Haveli and Daman and Diu', 'Lakshadweep', 'Delhi', 'Puducherry'
# ]

# # --- Session Variables ---
# for key, value in {
#     'authenticated': False,
#     'page': 'login',
#     'login_trigger': 0,
#     'user_data': {},
#     'recommendations': None,
#     'interaction_trigger': 0,
#     'generated_otp': None,
#     'user_role': 'user',
#     'messages': [],
#     'session_id': str(uuid.uuid4())
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = value

# # --- Helper Functions ---
# def run_flow(user_message, session_id, user_name, tweaks=None, api_key=None):
#     api_url = f"{BASE_API_URL}/api/v1/run/{FLOW_ID}"
#     payload = {
#         "session_id": session_id,
#         "input_value": user_message,
#         "input_type": "chat",
#         "output_type": "chat",
#         "tweaks": tweaks or {}
#     }
#     payload["tweaks"].update({
#         "ChatInput-voquK": {"session_id": session_id},
#         "TextInput-s26mJ": {"input_value": user_name},
#         "Memory-rRETL": {"session_id": session_id},
#         "ChatOutput-GFauW": {"session_id": session_id}
#     })
#     headers = {"x-api-key": api_key} if api_key else {}
#     response = requests.post(api_url, json=payload, headers=headers)
#     response.raise_for_status()
#     return response.json()["outputs"][0]["outputs"][0]["results"]["message"]["text"]

# @st.cache_resource
# def init_geocoder():
#     return RateLimiter(OpenCage(api_key=GEOCODE_API_KEY).geocode, min_delay_seconds=1)

# @st.cache_resource
# def init_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_data
# def load_location_cache() -> dict:
#     if CACHE_FILE.exists():
#         return pd.read_csv(CACHE_FILE).set_index("location")[['lat','lon']].to_dict('index')
#     return {}

# @st.cache_data
# def get_coordinates(name: str, cache: dict) -> Optional[Tuple[float, float]]:
#     if pd.isna(name): return None
#     if name in cache:
#         return (cache[name]['lat'], cache[name]['lon'])
#     loc = init_geocoder()(name)
#     if loc:
#         coords = (loc.latitude, loc.longitude)
#         cache[name] = {'lat': coords[0], 'lon': coords[1]}
#         pd.DataFrame.from_dict({name: coords}, orient='index', columns=['lat','lon']).to_csv(CACHE_FILE, mode='a', header=not CACHE_FILE.exists(), index_label='location')
#         return coords
#     return None

# # --- Pages ---

# ## LOGIN
# if st.session_state.page == 'login':
#     st.title("🔐 Login")
#     phone = st.text_input("Phone")
#     name = st.text_input("Name")
#     if st.button("Send OTP", disabled=not (phone.isdigit() and len(phone) == 10 and name.strip())):
#         st.session_state.generated_otp = "123456"
#         st.session_state.user_data["name"] = name.strip()
#         st.success("OTP sent! Use 123456 for demo.")
#     if st.session_state.generated_otp:
#         otp = st.text_input("Enter OTP")
#         if st.button("Verify OTP"):
#             if otp == st.session_state.generated_otp:
#                 st.session_state.authenticated = True
#                 st.session_state.page = "main"
#                 st.rerun()
#             else:
#                 st.error("Incorrect OTP")
#     st.subheader("Admin Access")
#     admin_email = st.text_input("Work Email")
#     if st.button("Admin Access"):
#         if admin_email.endswith("@innodatatics.com"):
#             st.session_state.authenticated = True
#             st.session_state.user_role = "admin"
#             st.session_state.page = "admin_view"
#             st.success("Access granted")
#             st.rerun()
#         else:
#             st.error("Access denied.")

# ## MAIN APP
# elif st.session_state.page == 'main' and st.session_state.authenticated:
#     st.title("🧠 AI Job Recommender")
#     col1, col2, col3 = st.columns(3)
#     if col1.button("Logout"):
#         for key in ['authenticated', 'generated_otp', 'recommendations', 'user_data']:
#             st.session_state[key] = False if isinstance(st.session_state[key], bool) else {}
#         st.session_state.page = 'login'
#         st.rerun()
#     if col2.button("Chatbot Help"):
#         st.session_state.page = "chatbot"
#         st.rerun()
#     if col3.button("Unsupervised Recommendation"):
#         st.session_state.page = "unsupervised"
#         st.rerun()

#     with st.form("form"):
#         name = st.text_input("Name", value=st.session_state.user_data.get("name", ""))
#         age = st.number_input("Age", 18, 90, 30)
#         location = st.selectbox("Location", indian_states)
#         skills = st.multiselect("Skills", available_skills)
#         salary = st.number_input("Expected Salary", 0)
#         model_choice = st.selectbox("Model", ["Rule-Based", "Unsupervised"])
#         top_n = st.slider("Top N", 1, 20, 3)
#         submitted = st.form_submit_button("Get Recommendations")

#     if submitted and name.strip() and skills:
#         st.session_state.user_data = {
#             "name": name,
#             "age": age,
#             "location": location,
#             "skills": ", ".join(skills),
#             "salary": salary,
#             "top_n": top_n,
#             "session_id": str(uuid.uuid4()),
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#         if model_choice == "Rule-Based":
#             st.session_state.recommendations = recommend_jobs(
#                 user_name=name, user_age=age, user_location=location,
#                 user_skills=st.session_state.user_data["skills"],
#                 expected_salary=salary, top_n=top_n
#             )
#         else:
#             st.warning("Use the Unsupervised button above to access this model.")

#     if st.session_state.recommendations is not None:
#         df = st.session_state.recommendations
#         if not df.empty:
#             for i, row in df.iterrows():
#                 with st.expander(f"{row['Company']} - {row['Job type']}"):
#                     st.write(f"📍 {row['State']}")
#                     st.write(f"💰 Match Score: {row['match_score']}")
#                     if st.button(f"Interested in {row['Company']}", key=f"{i}"):
#                         st.success("Interest logged!")

# ## CHATBOT
# elif st.session_state.page == 'chatbot':
#     st.title("💬 InnoDatatics Chat")
#     if st.button("🔙 Back"):
#         st.session_state.page = "main"
#         st.rerun()
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]): st.markdown(msg["content"])
#     if prompt := st.chat_input("Type your message…"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 reply = run_flow(prompt, st.session_state.session_id, st.session_state.user_data.get("name", ""))
#             st.markdown(reply)
#             st.session_state.messages.append({"role": "assistant", "content": reply})

# ## UNSUPERVISED
# elif st.session_state.page == "unsupervised":
#     st.title("🤖 Unsupervised Job Recommendation")
#     if st.button("🔙 Back to Recommender"):
#         st.session_state.page = "main"
#         st.rerun()

#     st.sidebar.header("Worker Profile")
#     w_name = st.sidebar.text_input("Name", "John Doe")
#     w_city = st.sidebar.text_input("City", "Mumbai")
#     w_skills = st.sidebar.text_input("Skills (comma-separated)", "Plumber")
#     w_salary = st.sidebar.number_input("Monthly Wage (₹)", min_value=0, value=30000)
#     top_n = st.sidebar.slider("Top N", 1, 20, 5)
#     run_btn = st.sidebar.button("Recommend Jobs")

#     if run_btn:
#         jobs_path = "C://Users//Tanmay//Downloads//Data_Innodatatics1-Data_Innodatatics1.xlsx.xlsx"
#         df_uns = pd.read_excel(jobs_path, engine="openpyxl")
#         df_uns["Avg_salary"] = (df_uns["Min salary"] + df_uns["Max salary"]) / 2
#         mean_sal = df_uns.loc[df_uns["Avg_salary"] != 0, "Avg_salary"].mean()
#         df_uns["Avg_salary"] = df_uns["Avg_salary"].replace(0, mean_sal)
#         df_uns["job_text"] = (
#             df_uns["Job type"] + " role in " + df_uns["State"] +
#             ". Average salary: ₹" + df_uns["Avg_salary"].astype(int).astype(str)
#         )
#         cache = load_location_cache()
#         df_uns["coords"] = df_uns["State"].apply(lambda s: get_coordinates(s, cache))
#         model = init_model()
#         job_emb = model.encode(df_uns["job_text"].tolist(), show_progress_bar=False)
#         skill_texts = [f"{s.strip()} seeking role in {w_city}" for s in w_skills.split(",")]
#         w_emb = model.encode(skill_texts, show_progress_bar=False).mean(axis=0).reshape(1, -1)
#         sims = cosine_similarity(w_emb, job_emb).flatten()
#         df_uns["similarity"] = sims
#         df_filtered = df_uns[(df_uns["Min salary"] <= w_salary) & (df_uns["Max salary"] >= w_salary)].copy()
#         user_coord = get_coordinates(w_city, cache)
#         df_filtered["distance"] = df_filtered["coords"].apply(lambda c: geodesic(c, user_coord).km if c and user_coord else np.inf)
#         arr = np.array([[s, d] for s, d in zip(df_filtered["similarity"], df_filtered["distance"])])
#         norm_sim = MinMaxScaler().fit_transform(arr[:, 0].reshape(-1, 1))
#         norm_dist = MinMaxScaler().fit_transform(arr[:, 1].reshape(-1, 1))
#         scores = 0.7 * norm_sim - 0.3 * norm_dist
#         df_filtered["score"] = scores.flatten()
#         top_df = df_filtered.sort_values("score", ascending=False).head(top_n)
#         st.dataframe(top_df[["Company", "Job type", "State", "Avg_salary", "score"]])

# ## ADMIN VIEW
# elif st.session_state.page == "admin_view":
#     st.title("🛠 Admin Panel")
#     if st.button("Logout"):
#         st.session_state.page = 'login'
#         st.session_state.authenticated = False
#         st.session_state.user_role = "user"
#         st.rerun()
#     action = st.radio("Choose action", ["Download Logs", "Append Jobs"])
#     if action == "Download Logs":
#         if os.path.exists(INTERACTION_LOG):
#             logs = pd.read_csv(INTERACTION_LOG)
#             st.download_button("Download CSV", logs.to_csv(index=False), "interactions.csv", "text/csv")
#         else:
#             st.warning("No interaction data found.")
#     elif action == "Append Jobs":
#         csv_data = st.text_area("Paste CSV data")
#         if st.button("Append"):
#             if csv_data:
#                 try:
#                     from io import StringIO
#                     new_df = pd.read_csv(StringIO(csv_data))
#                     current = pd.read_csv("jobs.csv")
#                     combined = pd.concat([current, new_df], ignore_index=True)
#                     combined.to_csv("jobs.csv", index=False)
#                     st.success("Jobs appended.")
#                 except Exception as e:
#                     st.error(str(e))
#             else:
#                 st.warning("Enter CSV job data.")


import streamlit as st
import pandas as pd
import os
from datetime import datetime
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
BASE_API_URL      = "http://52.205.254.228:7860"
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

# --- LangFlow Chatbot Helper ---
def run_flow(user_message, session_id, user_name, tweaks=None, api_key=None):
    api_url = f"{BASE_API_URL}/api/v1/run/{FLOW_ID}"
    payload = {
        "session_id": session_id,
        "input_value": user_message,
        "input_type": "chat",
        "output_type": "chat",
        "tweaks": tweaks or {}
    }
    payload["tweaks"].update({
        "ChatInput-aAzUo": {"session_id": session_id},
        "TextInput-LnzCN": {"input_value": user_name},
        "Memory-YVR39": {"session_id": session_id},
        "ChatOutput-8QykV": {"session_id": session_id}
    })
    headers = {"x-api-key": api_key} if api_key else {}
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["outputs"][0]["outputs"][0]["results"]["message"]["text"]

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

    can_send = phone.isdigit() and len(phone)==10 and name.strip()
    if st.button("Send OTP", disabled=not can_send):
        st.session_state.generated_otp = "123456"
        st.session_state.user_data["name"] = name.strip()
        st.success("OTP sent! Use 123456 for demo.")

    if st.session_state.generated_otp:
        otp = st.text_input("Enter OTP", key="otp_input")
        if st.button("Verify OTP"):
            if otp == st.session_state.generated_otp:
                # Clear previous chat
                st.session_state['messages']   = []
                st.session_state['session_id'] = str(uuid.uuid4())

                st.session_state.authenticated  = True
                st.session_state.user_role      = 'user'
                st.session_state.page           = 'main'
                st.session_state.login_trigger += 1
                st.rerun()
            else:
                st.error("Incorrect OTP")

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
            st.error("Access denied. Use a valid @innodatatics.com email.")

# --- PAGE: MAIN APP ---
elif st.session_state.page == 'main' and st.session_state.authenticated:
    st.title("🧠 AI Job Recommender")

    col1, col2, col3 = st.columns(3)
    if col1.button("Logout"):
        for k in ['authenticated','generated_otp','recommendations','user_data']:
            st.session_state[k] = False if isinstance(st.session_state[k], bool) else {}
        st.session_state['messages']   = []
        st.session_state['session_id'] = str(uuid.uuid4())
        st.session_state.page = 'login'
        st.rerun()
    if col2.button("Chatbot Help"):
        st.session_state.page = 'chatbot'
        st.rerun()
    if col3.button("Unsupervised Recommendation"):
        st.session_state.page = 'unsupervised'
        st.rerun()

    with st.form("user_form"):
        name         = st.text_input("Name", value=st.session_state.user_data.get("name",""))
        age          = st.number_input("Age", 18, 90, 30)
        location     = st.selectbox("Location (State)", indian_states)
        skills       = st.multiselect("Skills (Job Types)", available_skills)
        salary       = st.number_input("Expected Monthly Salary (INR)", min_value=0)
        model_choice = st.selectbox("Choose Model", ["Rule-Based", "Unsupervised"])
        top_n        = st.slider("Number of Recommendations", 1, 20, 3)
        submitted    = st.form_submit_button("Get Recommendations")

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
            if model_choice == "Rule-Based":
                st.session_state.recommendations = recommend_jobs(
                    user_name=name,
                    user_age=age,
                    user_location=location,
                    user_skills=st.session_state.user_data["skills"],
                    expected_salary=salary,
                    top_n=top_n
                )
            else:
                st.warning("Please use the 'Unsupervised Recommendation' button above.")

    # Display rule-based results
    recs = st.session_state.recommendations
    if recs is not None:
        if not recs.empty:
            for idx, row in recs.iterrows():
                with st.expander(f"📌 {row['Company']}"):
                    st.write(f"**Job type:** {row['Job type']}")
                    st.write(f"**State:** {row['State']}")
                    st.write(f"**Match score:** {row['match_score']}")
                    if st.button(f"Interested in {row['Company']}", key=f"int_{idx}"):
                        st.success("Your interest has been logged!")
        else:
            st.warning("No jobs found matching your profile.")

# --- PAGE: CHATBOT ---
elif st.session_state.page == 'chatbot':
    st.title("💬 InnoDatatics Chat")
    if st.button("🔙 Back to Recommender"):
        st.session_state.page = 'main'
        st.rerun()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Type your message…"):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("assistant"):
            reply = run_flow(prompt, st.session_state.session_id,
                             st.session_state.user_data.get("name",""))
            st.markdown(reply)
            st.session_state.messages.append({"role":"assistant","content":reply})

# --- PAGE: UNSUPERVISED ---

# elif st.session_state.page == "unsupervised":
#     st.title("🤖 Unsupervised Job Recommendation")
#     if st.button("🔙 Back"):
#         st.session_state.page = "main"; st.rerun()
 
#     # Sidebar for unsupervised inputs
#     st.sidebar.header("Worker Profile")
#     w_nm    = st.sidebar.text_input("Name", "John Doe")
#     w_city  = st.sidebar.text_input("City", "Mumbai")
#     w_skill = st.sidebar.text_input("Skills (comma-separated)", "Plumber")
#     w_sal   = st.sidebar.number_input("Monthly Wage (₹)", 0, value=30000)
#     top_n   = st.sidebar.slider("Top N", 1, 20, 5)
#     run_btn = st.sidebar.button("Run Unsupervised")
 
#     if run_btn:
#         # Prep text
#         df_uns = jobs_df.copy()
#         df_uns["Avg_salary"] = (df_uns["Min salary"] + df_uns["Max salary"])/2
#         mean_sal = df_uns.loc[df_uns["Avg_salary"]!=0,"Avg_salary"].mean()
#         df_uns["Avg_salary"].replace(0, mean_sal, inplace=True)
#         df_uns["job_text"] = (df_uns["Job type"] + " role in " + df_uns["State"]
#                               + ". Avg ₹" + df_uns["Avg_salary"].astype(int).astype(str))
 
#         # Embedding + PCA
#         mdl   = init_model()
#         emb   = mdl.encode(df_uns["job_text"].tolist(), show_progress_bar=False)
#         scaler= MinMaxScaler().fit(emb)
#         emb_s = scaler.transform(emb)
#         pca  = PCA(n_components=PCA_COMPONENTS, random_state=42)
#         job_pca = pca.fit_transform(emb_s)
 
#         # Cluster
#         best_name, labels = run_tuned_clustering(job_pca)
#         df_uns["cluster"] = labels
#         st.success(f"Best algorithm: {best_name}")
 
#         # Worker embed + transform
#         skill_texts = [f"{sk.strip()} seeking role in {w_city}" for sk in w_skill.split(",")]
#         skill_emb   = mdl.encode(skill_texts, show_progress_bar=False)
#         emb_w_s     = scaler.transform(skill_emb)
#         w_pca_full  = pca.transform(emb_w_s)
#         w_pca       = w_pca_full.mean(axis=0).reshape(1,-1)
 
#         # Assign to cluster by nearest job
#         dists      = np.linalg.norm(job_pca - w_pca, axis=1)
#         worker_cl  = int(df_uns.loc[dists.argmin(),"cluster"])
#         st.write(f"Worker assigned to cluster **{worker_cl}**")
 
#         # Compute semantic similarity
#         worker_emb = skill_emb.mean(axis=0).reshape(1,-1)
#         sims       = cosine_similarity(worker_emb, emb).flatten()
#         df_uns["sim"] = sims
 
#         # Show top-N in that cluster
#         subset = df_uns[df_uns["cluster"]==worker_cl]
#         top_jobs = subset.nlargest(top_n, "sim")
#         st.subheader(f"Top {top_n} jobs in cluster {worker_cl}")
#         for _, row in top_jobs.iterrows():
#             st.markdown(f"**{row['Company']}**  \n"
#                         f"{row['Job type']} — {row['State']}  \n"
#                         f"Similarity: {row['sim']:.2f}")

elif st.session_state.page == 'unsupervised':

 
    # Define and ensure model directory exists
    PCA_MODEL_PATH = Path("models/pca_model.pkl")
    CLUSTER_MODEL_PATH = Path("models/cluster_model.pkl")
    PCA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    st.title("🤖 Unsupervised Job Recommendation")
    if st.button("🔙 Back"):
        st.session_state.page = 'main'; st.rerun()
 
    st.sidebar.header("Worker Profile")
    w_nm = st.sidebar.text_input("Name", "John Doe")
    w_city = st.sidebar.text_input("City", "Mumbai")
    w_skill = st.sidebar.text_input("Skills (comma-separated)", "Plumber")
    w_sal = st.sidebar.number_input("Monthly Wage (₹)", 0, value=30000)
    top_n = st.sidebar.slider("Top N", 1, 20, 5)
    run_btn = st.sidebar.button("Run Unsupervised")
 
    if run_btn:
        df_uns = jobs_df.copy()
        df_uns['Avg_salary'] = (df_uns['Min salary'] + df_uns['Max salary']) / 2
        mean_sal = df_uns.loc[df_uns['Avg_salary'] != 0, 'Avg_salary'].mean()
        df_uns['Avg_salary'].replace(0, mean_sal, inplace=True)
        df_uns['job_text'] = df_uns['Job type'] + " role in " + df_uns['State'] + ". Avg ₹" + df_uns['Avg_salary'].astype(int).astype(str)
 
        mdl = init_model()
        emb = mdl.encode(df_uns['job_text'].tolist(), show_progress_bar=False)
        scaler = MinMaxScaler().fit(emb)
        emb_s = scaler.transform(emb)
 
        # Load or fit PCA
        if PCA_MODEL_PATH.exists():
            with open(PCA_MODEL_PATH, 'rb') as f:
                pca = pickle.load(f)
        else:
            pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
            pca.fit(emb_s)
            with open(PCA_MODEL_PATH, 'wb') as f:
                pickle.dump(pca, f)
 
        job_pca = pca.transform(emb_s)
 
        # Load or fit clustering model
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
        st.subheader(f"Top {top_n} jobs in cluster {worker_cl}")
        for _, row in top_jobs.iterrows():
            st.markdown(f"**{row['Company']}**  \n{row['Job type']} — {row['State']}  \nSimilarity: {row['sim']:.2f}")

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
        if os.path.exists(INTERACTION_LOG):
            df_log = pd.read_csv(INTERACTION_LOG)
            st.download_button("Download CSV", df_log.to_csv(index=False), "interactions.csv", "text/csv")
        else:
            st.warning("No interaction data found.")
    elif action == "Append to jobs.csv":
        new_data = st.text_area("Paste new job data (CSV format)")
        if st.button("Append Job"):
            if new_data:
                try:
                    from io import StringIO
                    new_df   = pd.read_csv(StringIO(new_data))
                    existing = pd.read_csv("jobs.csv")
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined.to_csv("jobs.csv", index=False)
                    st.success("New job(s) added successfully.")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter job data first.")
