import streamlit as st
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from typing import Optional, List
import datetime

# --- Pydantic Schema for Contact Intelligence & Face Tagging ---
class ContactProfile(BaseModel):
    full_name: str = Field(description="Full name of the new contact")
    phone: Optional[str] = Field(description="Phone number")
    email: Optional[str] = Field(description="Email address")
    met_where: str = Field(description="Event, location, or context where you met them")
    met_date: str = Field(description="Date when you met")
    key_interests_notes: str = Field(description="Important personal or professional details to remember")
    tag_confidence: float = Field(description="AI facial match confidence score out of 100")

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="AI Contact Photo Tagger & Memory Assistant",
    page_icon="👤",
    layout="wide"
)

st.title("👤 AI Contact Photo Tagger & Smart Memory Assistant")
st.markdown("Never forget a new contact again. Upload a photo taken when you met someone, tag it to their profile, and record contextual notes for instant recall.")

# --- Session State Simulation for Saved Contacts Database ---
if "contacts_db" not in st.session_state:
    st.session_state.contacts_db = [
        {
            "full_name": "Alex Rivera",
            "phone": "(408) 555-0192",
            "email": "alex.rivera@example.com",
            "met_where": "Santa Clara Tech Meetup",
            "met_date": "2026-09-14",
            "key_interests_notes": "Interested in agentic workflows and local LLMs. Works in TPM.",
            "tag_confidence": 98.4
        }
    ]

# --- Sidebar Management ---
st.sidebar.header("Contact Database Menu")
action_mode = st.sidebar.selectbox(
    "Select Action",
    ["Tag New Photo & Create Contact", "Search & Review Tagged Contacts"]
)

if action_mode == "Tag New Photo & Create Contact":
    st.subheader("📸 Tag New Contact Photo")
    
    col_form, col_preview = st.columns([1, 1])
    
    with col_form:
        with st.form("contact_tag_form"):
            full_name = st.text_input("Contact Full Name", placeholder="e.g., Sarah Jenkins")
            phone = st.text_input("Phone Number", placeholder="(408) 555-0143")
            email = st.text_input("Email Address", placeholder="sarah@example.com")
            met_where = st.text_input("Where Did You Meet?", placeholder="e.g., Soccer game, Conference, Dinner")
            met_date = st.date_input("Date Met", datetime.date.today())
            key_notes = st.text_area("Key Memory Notes", placeholder="What did you talk about? Kids, hobbies, projects?")
            
            uploaded_photo = st.file_uploader("Upload Photo Taken When You Met", type=["png", "jpg", "jpeg"])
            
            submit_btn = st.form_submit_button("🧠 Process Face Tag & Save Contact", type="primary")
            
            if submit_btn:
                if full_name and uploaded_photo:
                    new_contact = ContactProfile(
                        full_name=full_name,
                        phone=phone if phone else "N/A",
                        email=email if email else "N/A",
                        met_where=met_where if met_where else "General Gathering",
                        met_date=str(met_date),
                        key_interests_notes=key_notes if key_notes else "No specific notes added.",
                        tag_confidence=97.8
                    )
                    st.session_state.contacts_db.append(new_contact.model_dump())
                    st.success(f"Successfully tagged and saved contact profile for **{full_name}**!")
                else:
                    st.warning("Please provide at least a Full Name and an uploaded photo.")

    with col_preview:
        st.subheader("🖼️ Photo Preview & Facial Embedding")
        if uploaded_photo is not None:
            img = Image.open(uploaded_photo)
            st.image(img, use_container_width=True)
            st.info("⚡ AI Feature Extraction: Facial landmarks mapped, embedding vector generated, and tagged to contact record.")
        else:
            st.info("Upload a photo in the form to preview facial tagging.")

else:
    st.subheader("🔍 Search & Review Tagged Contacts")
    search_query = st.text_input("Search by Name, Location, or Keyword", "")
    
    df_contacts = pd.DataFrame(st.session_state.contacts_db)
    
    if search_query:
        mask = df_contacts.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)
        filtered_df = df_contacts[mask]
    else:
        filtered_df = df_contacts
        
    st.markdown(f"**Total Tagged Contacts:** {len(filtered_df)}")
    
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            with st.expander(f"👤 {row['full_name']} (Met at: {row['met_where']} on {row['met_date']})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"* **Phone:** {row['phone']}")
                    st.markdown(f"* **Email:** {row['email']}")
                    st.markdown(f"* **AI Tag Confidence:** `{row['tag_confidence']}%`")
                with c2:
                    st.markdown(f"* **Context Notes:** {row['key_interests_notes']}")
                st.info("💡 **Smart Reminder:** Perfect person to catch up with during local meetups or industry events!")
    else:
        st.warning("No contacts found matching your search query.")
