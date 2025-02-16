import streamlit as st
import nltk
import spacy
nltk.download('stopwords')
spacy.load('en_core_web_sm')
import re
from pdfminer.high_level import extract_text
import google.generativeai as genai
from spacy.matcher import Matcher
import pandas as pd
import base64, random
import time, datetime
import google
import io, random
from streamlit_tags import st_tags
from bs4 import BeautifulSoup
from PIL import Image
import pymysql
import plotly.express as px
from io import BytesIO
import requests
import uuid 

st.sidebar.title("HireBot")
page = st.sidebar.selectbox("Select Page", ["Resume Analyzer", "Job Search"])

def extract_text_from_pdf(pdf_file):
    """Extract text content from a PDF file."""
    pdf_bytes = pdf_file.read()  # Read the file as bytes
    pdf_io = BytesIO(pdf_bytes)  # Convert bytes to a file-like object
    return extract_text(pdf_io)  # Extract text from the file-like object

def extract_contact_number_from_resume(text):
    """Extract a contact number from the resume text using regex."""
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None

def extract_email_from_resume(text):
    """Extract an email address from the resume text using regex."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None

def extract_name(resume_text):
    """Extract the full name from the resume text using spaCy and Matcher."""
    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First, Middle, and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]  # First, Middle, Middle, and Last name
    ]
    for pattern in patterns:
        matcher.add('NAME', patterns=[pattern])
    doc = nlp(resume_text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        return span.text
    return None

def extract_degree_name(resume_text):
    """Extract degree name from the resume text using Google Generative AI."""
    prompt = f"Based on the following resume text, identify the degree name. If there are multiple degree, separate them with commas and ensure the current one appears first. Only provide the degree name. Sample degree name will be like Bachelor of Enginnering in Computer Science, Master of Technology Data Science and Engineering, etc.. Resume text: {resume_text}"
    response = generate_response(prompt)
    return response

def extract_college_name(resume_text):
    """Extract college name from the resume text using Google Generative AI."""
    prompt = f"Based on the following resume text, identify the college name.If there are multiple college name, separate them with commas and ensure the current one appears first. Only provide the college name. Resume text: {resume_text}"
    response = generate_response(prompt)
    return response

def extract_company_name(resume_text):
    """Extract company name from the resume text using Google Generative AI."""
    prompt = f"""
            Based on the following resume text, identify the company name(s) with the years of experience.
            If there are multiple company names, separate them with commas and ensure the current one appears first.
            Format: Company Name | Years of Experience (in decimal format like X.Y years). 

            Only provide the company name(s) and years of experience in that company. 

            Resume text: {resume_text}
            """
    response = generate_response(prompt)
    return response

def extract_total_experience(resume_text):
    """Extract company name from the resume text using Google Generative AI."""
    prompt = f"""
            Based on the following resume text, Calculate the total experience.
            If there are multiple company names, sum up the years of experience.

            Only provide the years of experience. 
            Examples: 7 years, 2.8 years, 5.1 years

            Resume text: {resume_text}
            """
    response = generate_response(prompt)
    return response

def extract_strength(resume_text):
    """Identify 4 key points from the resume text using Google Generative AI."""
    prompt = f"Based on the following resume text, identify the 4 key Strength. Provide short, concise and specific points. Resume text: {resume_text}"
    response = generate_response(prompt)
    return response

def extract_weakness(resume_text):
    """Identify 4 key points from the resume text using Google Generative AI."""
    prompt = f"Based on the following resume text, identify the 4 key Weakness. Provide short, concise and specific points. Resume text: {resume_text}"
    response = generate_response(prompt)
    return response

def extract_skills(resume_text):
    """Extract skills from the resume text using Google Generative AI and remove duplicates."""
    prompt = f"Based on the following resume text, list the top 20 skills related to the job designation, including technical skills. Separate them by commas. Only give skills. Resume text: {resume_text}"
    response = generate_response(prompt)
    skills = response.split(',')
    skills = [skill.strip() for skill in skills]  # Strip any extra spaces
    return list(set(skills))  # Remove duplicates by converting to a set and back to a list

def extract_designation(resume_text):
    """Extract the designation from the resume text using Google Generative AI."""
    prompt = f"Based on the following resume text, identify the job designation(s). If there are multiple designations, separate them with commas and ensure the current one appears first. Just give name of designations. Resume text: {resume_text}"
    response = generate_response(prompt)
    designations = response.split(',')
    designations = [designation.strip() for designation in designations]  # Strip any extra spaces
    return designations

def extract_recommended_skills(designations):
    """Extract the designation from the resume text using Google Generative AI."""
    prompt = f"Based on the following designations text, recommend the top 15 skills. Separate them with commas. Just name the skills. Resume text: {designations}"
    response = generate_response(prompt)
    recommended_skills = response.split(',')
    recommended_skills = [recommended_skill.strip() for recommended_skill in recommended_skills]  # Strip any extra spaces
    return recommended_skills

def extract_recommended_job_position(designations):
    """Extract the designation from the resume text using Google Generative AI."""
    prompt = f"Based on the following designations text, recommend the 10 job position. Separate them with commas. Just name the position. Resume text: {designations}"
    response = generate_response(prompt)
    recommended_job_positions = response.split(',')
    recommended_job_positions = [recommended_job_position.strip() for recommended_job_position in recommended_job_positions]  # Strip any extra spaces
    return recommended_job_positions

def extract_social(resume_text):
    pattern = r"(https?:\/\/)?(www\.)?(linkedin\.com\/in\/[a-zA-Z0-9-]+|github\.com\/[a-zA-Z0-9-]+)"
    matches = re.findall(pattern, resume_text)
    links = ["".join(match) for match in matches]
    return links

def generate_response(prompt):
    """Generate a response from the generative AI model with retry logic."""
    genai.configure(api_key="") # Add your api key over here
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65536,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        generation_config=generation_config,
    )
    chat_session = model.start_chat(history=[])
    
    # Retry logic with exponential backoff
    for attempt in range(5):  # Retry 5 times
        try:
            response = chat_session.send_message(prompt)
            return response.text.replace("**", "").strip()
        except google.api_core.exceptions.InternalServerError as e:
            if attempt < 4:  # Don't wait after the last attempt
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
            else:
                st.error(f"Error generating response: {e}")
                return "Failed to generate response. Please try again later."

def extract_resume_information(pdf_file):
    """Extract all relevant information from a resume PDF."""
    # Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_file)

    # Extract contact info, name, degree, college, company, skills, and designation
    social = extract_social(resume_text)
    contact_number = extract_contact_number_from_resume(resume_text)
    email = extract_email_from_resume(resume_text)
    name = extract_name(resume_text)
    degree_name = extract_degree_name(resume_text)
    college_name = extract_college_name(resume_text)
    company_name = extract_company_name(resume_text)
    skills = extract_skills(resume_text)
    designations = extract_designation(resume_text)
    recommended_skills = extract_recommended_skills(designations)
    recommended_job_positions = extract_recommended_job_position(designations)
    weakness = extract_weakness(resume_text)
    strength = extract_strength(resume_text)
    total_experience = extract_total_experience(resume_text)

    return {
        'name': name,
        'contact_number': contact_number,
        'email': email,
        'degree_name': degree_name,
        'college_name': college_name,
        'company_name': company_name,
        'skills': skills,
        'designations': designations,
        'recommended_skills': recommended_skills,
        'linkedin': social[0] if len(social) > 0 else "NA",  # Check if LinkedIn is available
        'github': social[1] if len(social) > 1 else "NA",  # Check if GitHub is available
        'recommended_job_positions': recommended_job_positions,
        'weakness': weakness,
        'strength': strength,
        'total_experience': total_experience
}

def extract_resume_information_job_search(pdf_file):
    """Extract all relevant information from a resume PDF."""
    # Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_file)

    # Extract contact info, name, degree, college, company, skills, and designation
    name = extract_name(resume_text)
    skills = extract_skills(resume_text)
    designations = extract_designation(resume_text)

    return {
        'name': name,
        'skills': skills,
        'designations': designations
}

def display_static_tags_edu(title, items):
    st.write(f"**{title}:**")
    if isinstance(items, list):  # Check if items is already a list
        items_list = items
    elif isinstance(items, str):  # If it's a string, split it
        items_list = [item.strip() for item in items.split(",")]
    else:
        items_list = []

    if items_list:
        items_display = " ".join(
            [f"<div style='display: inline-block; background-color: #ffff99; color: black; "
             f"padding: 8px 12px; border-radius: 5px; font-size: 17px; margin: 2px;'>{item}</div>"
             for item in items_list]
        )
        st.markdown(f"<div style='display: flex; flex-wrap: wrap;'>{items_display}</div>", unsafe_allow_html=True)
    else:
        st.write("Not Found")

def display_static_tags_exp(title, items):
    st.write(f"**{title}:**")
    if isinstance(items, list):  # Check if items is already a list
        items_list = items
    elif isinstance(items, str):  # If it's a string, split it
        items_list = [item.strip() for item in items.split(",")]
    else:
        items_list = []

    if items_list:
        items_display = " ".join(
            [f"<div style='display: inline-block; background-color: #6699ff; color: black; "
             f"padding: 8px 12px; border-radius: 5px; font-size: 17px; margin: 2px;'>{item}</div>"
             for item in items_list]
        )
        st.markdown(f"<div style='display: flex; flex-wrap: wrap;'>{items_display}</div>", unsafe_allow_html=True)
    else:
        st.write("Not Found")

def display_static_tags_skills(title, items):
    st.write(f"**{title}**")
    if isinstance(items, list):  # Check if items is already a list
        items_list = items
    elif isinstance(items, str):  # If it's a string, split it
        items_list = [item.strip() for item in items.split(",")]
    else:
        items_list = []

    if items_list:
        items_display = " ".join(
            [f"<div style='display: inline-block; background-color: #2BC5B4; color: black; "
             f"padding: 8px 12px; border-radius: 5px; font-size: 16px; margin: 2px;'>{item}</div>"
             for item in items_list]
        )
        st.markdown(f"<div style='display: flex; flex-wrap: wrap;'>{items_display}</div>", unsafe_allow_html=True)
    else:
        st.write("Not Found")


def run():
    if page == "Resume Analyzer":
        st.title("HireBot")
        pdf_file = st.file_uploader("Upload a resume", type=["pdf"])

        if pdf_file is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Extracting text from PDF...")
            progress_bar.progress(20)
            time.sleep(5)

            status_text.text("Analyzing Resume Information...")
            progress_bar.progress(50)
            resume_info = extract_resume_information(pdf_file)
            time.sleep(2)
            
            status_text.text("Generating output...")
            progress_bar.progress(80)
            time.sleep(5)
            
            status_text.text("Finalizing output...")
            progress_bar.progress(100)
            time.sleep(3)
            
            progress_bar.empty()
            status_text.empty()

            st.success(f"Hello, {resume_info['name']} 👋")
            st.subheader('**🧑 Basic Info**')
            try:
                st.write('😀 Name: ' + (resume_info['name'] or "Not Found"))
                st.write('📧 Email: ' + (resume_info['email'] or "Not Found"))
                st.write('📞 Contact: ' + (resume_info['contact_number'] or "Not Found"))
                st.write('#️⃣ LinkedIn: ' + (resume_info['linkedin'] or "NA"))
                st.write('#️⃣ Github: ' + (resume_info['github'] or "NA"))
                st.write('🗞️ Total Years of Experience: ' + (resume_info['total_experience'] or "NA"))
            except:
                pass
            
            st.markdown("---")  

            st.subheader('**🎓 Education Details**')
            try:
                # st.write("**Degree Name:**", resume_info.get('degree_name', "Not Found"))
                display_static_tags_edu("📜 Degree", resume_info.get('degree_name', "Not Found"))
                display_static_tags_edu("🏫 College/University", resume_info.get('college_name', "Not Found"))
            except:
                pass
            
            st.markdown("---")  

            st.subheader('**💼 Professional Details**')
            try:
                display_static_tags_exp("🏢 Company Name", resume_info.get('company_name', "Not Found"))  # Fixed key
                # st.write("**Designations:**", ", ".join(resume_info.get('designations', [])))
                display_static_tags_exp("🧑‍💻 Designations ",resume_info.get('designations', "Not Found"))  # Fixed key
                display_static_tags_exp("🧑‍⚖️ Recommended Job Roles",resume_info.get('recommended_job_positions', "Not Found"))  # Fixed key
            except:
                pass
                
            st.markdown("---")  

            st.subheader('**🪛 Skills**')
            # keywords_skills = st_tags(label='### Skills that you have', text='See your skills', value=resume_info['skills'], key='skills')
            display_static_tags_skills("Your Skills", resume_info.get('skills', "Not Found"))  # Fixed key
            # keywords_recommended_skills = st_tags(label='### Skills that we recommend', text='See our skills recommendation', value=resume_info['recommended_skills'], key='recommended_skills')
            display_static_tags_skills("Recommended Skills", resume_info.get('recommended_skills', "Not Found"))  # Fixed key    

            st.markdown("---")

            st.subheader('**Strength and Weakness**')
            st.write('**✅ Strength**')
            st.success(resume_info.get('strength'))
            st.write('**❌ Weakness**')
            st.error(resume_info.get('weakness'))

        else:
            st.warning("Please upload a PDF file.")

    elif page == 'Job Search':
        st.title("HireBot")
        pdf_file = st.file_uploader("Upload a resume", type=["pdf"])
        
        if pdf_file is not None:
            resume_info = extract_resume_information_job_search(pdf_file)

            col1, col2, col3 = st.columns(3)
            job_role = st.text_input("Enter Job Role:", value=resume_info['designations'][0] if isinstance(resume_info['designations'], list) and resume_info['designations'] else "", key="job_role")
            location_preference = col1.text_input("Enter Location:", key="location")
            employment_options = ["Full-time", "Part-time", "Intern", "Contractor"]
            employment_type_preference = col2.selectbox("Select Employment Type:", employment_options, key="employment_type")
            dated_posted_options = ['Month', 'Week', 'Today']
            dated_posted_preference = col3.selectbox('Select Date Posted: ', dated_posted_options, key="dated_posted")
            keywords_skills = st_tags(label='### Skills', text='See your skills', value=resume_info['skills'], key='skills')
            st.markdown("---")  

            if st.button("Submit ➜"):
                url = "https://jobs-api14.p.rapidapi.com/v2/list"

                employment_type_mapping = {
                    "Full-time": "fulltime",
                    "Part-time": "parttime",
                    "Intern": "intern",
                    "Contractor": "contractor"
                }

                dated_posted_mapping = {
                    "Month": "month",
                    "Week": "week",
                    "Today": "today"
                }
                
                all_jobs = []

                # API Request Parameters
                querystring = {
                    "query": job_role,
                    "location": location_preference,
                    "autoTranslateLocation": "true",
                    "remoteOnly": "false",
                    "employmentTypes": employment_type_mapping[employment_type_preference],
                    "datePosted": dated_posted_mapping[dated_posted_preference]
                }

                headers = {
                    "x-rapidapi-key": "", # Add your creds
                    "x-rapidapi-host": ""
                }

                # API Call
                response = requests.get(url, headers=headers, params=querystring)

                # Display Results
                if response.status_code == 200:
                    data = response.json()
                    st.write("### Job Results:")

                    jobs = data.get("jobs", [])
                    next_page_token = data.get("nextPage", None)

                    for job in jobs:
                        description = job.get("description", "")
                        cleaned_desc = BeautifulSoup(description, "html.parser").get_text()  # Removes HTML tags
                        truncated_desc = cleaned_desc[:250]  # Limit to 250 characters

                        all_jobs.append({
                            "Title": job.get("title", ""),
                            "Company": job.get("company", ""),
                            "Description": truncated_desc,
                            "Full Description": cleaned_desc,  # Store full description for "Read More"
                            "Location": job.get("location", ""),
                            "Employment Type": job.get("employmentType", ""),
                            "Date Posted": job.get("datePosted", ""),
                            "Provider": job.get("jobProviders", [{}])[0].get("jobProvider", ""),
                            "Link": job.get("jobProviders", [{}])[0].get("url", ""),
                        })

                    # **Display Jobs in Card Format**
                    if all_jobs:
                        for job in all_jobs:
                            read_more_id = str(uuid.uuid4())
                            full_description_id = str(uuid.uuid4())
                            link_provider = job.get('Provider', 'N/A')
                            link_url = job.get('Link', '#')

                            html_card = f"""
                            <div style="border: 2px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 10px; background-color: #1e1e1e; color: white;">
                                <h3 style="margin-bottom: 3px;">{job.get("Title", "N/A")}</h3>
                                <h4 style="color: #ccc; margin-bottom: 3px;">{job.get("Company", "N/A")}</h4>
                                <h5>📍 {job.get("Location", "N/A")} &nbsp;|&nbsp; 🏢 {job.get("Employment Type", "N/A")} &nbsp;|&nbsp; 📅 {job.get("Date Posted", "N/A")}</h5>

                                <p id="{read_more_id}">{job.get("Description", "N/A")}...</p>

                                <div id="{full_description_id}" style="display: none;">{job.get('Full Description', '')}</div>

                                <a href="javascript:void(0);" onclick="  // Key change here
                                    document.getElementById('{read_more_id}').style.display = 'none';
                                    document.getElementById('{full_description_id}').style.display = 'block';
                                    this.style.display = 'none';"
                                    style="color: #4CAF50; text-decoration: none;">Read More</a>

                                <a href="javascript:void(0);" onclick=" // And here
                                    document.getElementById('{full_description_id}').style.display = 'none';
                                    document.getElementById('{read_more_id}').style.display = 'block';
                                    this.style.display = 'none';"
                                    style="color: #4CAF50; text-decoration: none; display:none;">Read Less</a>

                                <h5>🔗 {link_provider}: <a href="{link_url}" target="_blank" style="color: #4CAF50; text-decoration: none;">Apply Now!</a></h5>
                            </div>
                            """

                            st.components.v1.html(html_card, height=500, scrolling=True)
                    else:
                        st.write("No jobs found for the given criteria.")
                else:
                    st.error("Failed to fetch jobs. Please try again.")

run()



