# HireBot

### Resume Parsing Functions:

- **extract\_text\_from\_pdf(pdf\_file):** Extracts text content from a given PDF file by reading its bytes and using a text extraction function.
- **extract\_contact\_number\_from\_resume(text):** Uses regex to identify and extract a contact number from the resume text.
- **extract\_email\_from\_resume(text):** Uses regex to identify and extract an email address from the resume text.
- **extract\_name(resume\_text):** Uses the spaCy NLP library and pattern matching to extract a full name (first, middle, and last names) from the resume text.
- **extract\_degree\_name(resume\_text):** Uses a generative AI model to extract the degree name from the resume text, ensuring that the current degree appears first.
- **extract\_college\_name(resume\_text):** Uses a generative AI model to extract the college/university name from the resume text.
- **extract\_company\_name(resume\_text):** Uses a generative AI model to identify company names and their corresponding years of experience from the resume text.
- **extract\_total\_experience(resume\_text):** Uses a generative AI model to calculate the total years of experience by summing up the experience across multiple companies.
- **extract\_strength(resume\_text):** Identifies four key strengths from the resume text using Google Generative AI.
- **extract\_weakness(resume\_text):** Identifies four key weaknesses from the resume text using Google Generative AI.
- **extract\_skills(resume\_text):** Extracts and returns the top 20 relevant skills from the resume text, ensuring uniqueness.
- **extract\_designation(resume\_text):** Identifies job designations mentioned in the resume, prioritizing the most recent one.
- **extract\_recommended\_skills(designations):** Suggests 15 relevant skills based on the extracted job designations.
- **extract\_recommended\_job\_position(designations):** Recommends 10 job positions based on the extracted designations.
- **extract\_social(resume\_text):** Extracts LinkedIn and GitHub profile links from the resume text using regex.

### Match Percentage Calculation:

- **extract\_match\_percentage(resume\_text, job\_description):** Uses a weighted scoring system to determine how well a resume matches a job description based on four criteria: skills, experience, education, and keywords.

#### Weighted Function Breakdown:

Match Percentage is computed as:

```
Match Percentage = (Skills Score × 0.40) + (Experience Score × 0.30) + (Education Score × 0.15) + (Keywords Score × 0.15)
```

- **Skills (40%)**: Exact skill matches score highest, related skills score lower.
- **Experience (30%)**: More years of experience get higher scores.
- **Education (15%)**: Full match gets full points; related degrees get partial points.
- **Keywords (15%)**: Important job description keywords are matched against the resume.

This ensures that skills and experience have the most weight, while education and keyword relevance contribute to the final match percentage.

## Resume Analyzer Features

A Streamlit-based resume analysis tool that allows users to upload a PDF resume and extract key information, providing insights into the candidate’s personal details, education, professional experience, skills, strengths, and weaknesses.

### How It Works:

1. **User Uploads a Resume:**

   - The user selects the "Resume Analyzer" page and uploads a PDF file.
   - If no file is uploaded, a warning message prompts the user.

2. **Resume Processing with a Progress Bar:**

   - Progress stages: Text extraction → Analysis → Output generation → Final results.

3. **Extracting Resume Information:**

   - The function `extract_resume_information(pdf_file)` processes the file and extracts:
     - **Basic Information:** Name, email, contact number, LinkedIn, GitHub, years of experience.
     - **Education Details:** Degree name and university.
     - **Professional Details:** Previous company, designations, and recommended job roles.
     - **Skills:** Extracted and recommended skills.
     - **Strengths & Weaknesses:** Key strengths and weaknesses identified from the resume.

### Key Features:

✔️ PDF Upload Support\
✔️ Real-time Progress Updates\
✔️ Automatic Resume Parsing\
✔️ Skill & Job Recommendations\
✔️ Strength & Weakness Analysis

This interactive resume analyzer streamlines resume processing and provides insights for recruiters and job seekers. 🚀

## Job Search Feature

A Streamlit web app feature that allows users to upload a resume (PDF), extract job-related details (role, experience, skills), and search for matching job postings via an API.

### How It Works:

1. **Resume Upload & Extraction:**

   - User uploads a PDF resume.
   - The function `extract_resume_information_job_search()` extracts job-related details.

2. **User Input Fields:**

   - Extracted job role is auto-filled in a text box.
   - Users enter preferred location and select employment type (Full-time, Part-time, Intern, Contractor).
   - Job postings can be filtered by date posted (Today, Week, Month).
   - Skills are extracted from the resume and displayed using `st_tags`.

3. **Job Search via API:**

   - Upon submission, an API request is sent to `jobs-api14.p.rapidapi.com`.
   - Filters include job role, location, employment type, and date posted.
   - The response is parsed, extracting job details (title, company, location, employment type, date posted, job provider).

4. **Displaying Job Listings:**

   - Results are shown as interactive job cards using HTML and CSS.
   - Each job listing includes a **match percentage** (calculated via `extract_match_percentage()`).
   - Users can expand/collapse the job description.
   - "Apply Now" button directs users to the job application link.

5. **Error Handling & API Limitations:**

   - Displays a message if no jobs are found.
   - A 30-second delay (`time.sleep(30)`) prevents API rate limiting.

### Key Features:

✅ Resume parsing for automated job search input\
✅ API-based job retrieval with filtering options\
✅ Match percentage calculation for better job relevance\
✅ Interactive job cards with "Read More" functionality\
✅ Streamlit UI integration for a seamless user experience

This feature simplifies the job search process by leveraging resume insights and real-time job postings. 🚀

## Installation & Usage

### Prerequisites

- Python 3.8+
- Install dependencies using:

```
pip install -r requirements.txt
```

### Running the Application

To start the Streamlit app, run:

```
streamlit run app.py
```

### Screenshot
<img width="907" alt="app" src="https://github.com/zahidshaikh10101/Hirebot/blob/1b5a6c27d1b204c5675296cc0a7091569210b442/images/RP%20-%201.png"> 
<img width="907" alt="app" src="https://github.com/zahidshaikh10101/Hirebot/blob/1b5a6c27d1b204c5675296cc0a7091569210b442/images/RP%20-%202.png"> 
<img width="907" alt="app" src="https://github.com/zahidshaikh10101/Hirebot/blob/1b5a6c27d1b204c5675296cc0a7091569210b442/images/RP%20-%203.png"> 
<img width="907" alt="app" src="https://github.com/zahidshaikh10101/Hirebot/blob/1b5a6c27d1b204c5675296cc0a7091569210b442/images/RS%20-1.png"> 
<img width="907" alt="app" src="https://github.com/zahidshaikh10101/Hirebot/blob/1b5a6c27d1b204c5675296cc0a7091569210b442/images/JS%20-2.png"> 
<img width="907" alt="app" src="https://github.com/zahidshaikh10101/Hirebot/blob/1b5a6c27d1b204c5675296cc0a7091569210b442/images/JS%20-%203.png"> 

### API Keys Setup

- Ensure you have API keys configured for job search functionalities.
- Add them in a `.env` file or directly in the script.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.


