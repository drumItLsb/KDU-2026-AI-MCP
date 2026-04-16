"""
LangChain utilities for resume processing
"""

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# OpenRouter exposes an OpenAI-compatible API surface, so we can keep using
# LangChain's OpenAI integrations while pointing them at the OpenRouter base URL.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# Initialize LangChain components
def init_langchain_components(api_key, base_url=OPENROUTER_BASE_URL,
                              chat_model=DEFAULT_CHAT_MODEL,
                              embedding_model=DEFAULT_EMBEDDING_MODEL):
    """Initialize LangChain components.
    
    Args:
        api_key: OpenRouter API key
        base_url: OpenRouter API base URL
        chat_model: OpenRouter chat model name
        embedding_model: Embedding model name exposed through OpenRouter's
            OpenAI-compatible endpoint
        
    Returns:
        tuple: (embeddings, llm) or (None, None) if error
    """
    if not api_key:
        return None, None

    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model=embedding_model,
        openai_api_base=base_url,
    )
    llm = ChatOpenAI(
        temperature=0,
        model=chat_model,
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    return embeddings, llm

def prepare_resume_documents(resume_text, filename):
    """
    Split resume text into chunks and wrap them as LangChain Document objects.
    
    Args:
        resume_text: Raw resume text
        filename: Name of the resume file
    
    Returns:
        dict: Contains original text and chunked Document list
    """
    # Step 1: Chunk the resume
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(resume_text)

    # Step 2: Wrap each chunk in a Document with metadata
    documents = [
        Document(page_content=chunk, metadata={"source": filename, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    return {
        "text": resume_text,
        "chunks": documents
    }

def find_relevant_sections(processed_resume, job_description, embeddings):
    """
    Use FAISS vector store to find top 3 resume chunks most relevant to a job description.
    
    Args:
        processed_resume: Output of process_resume_with_langchain (includes chunks)
        job_description: Job description string
        embeddings: OpenAI embeddings object
    
    Returns:
        List of (chunk_text, similarity_score) tuples
    """
    if not embeddings:
        return []

    # Build FAISS index from processed chunks
    vectorstore = FAISS.from_documents(processed_resume["chunks"], embeddings)

    # Perform semantic search
    results = vectorstore.similarity_search_with_score(job_description, k=3)

    # Return list of (text, score)
    return [(doc.page_content, score) for doc, score in results]


def extract_skills_with_langchain(resume_text, llm):
    """Extract skills from resume text using LangChain.
    
    Args:
        resume_text: Resume text content
        llm: LangChain language model
        
    Returns:
        str: Extracted skills or error message
    """
    if not llm:
        return "LangChain LLM not available for skill extraction."
    
    try:
        # Create a skill extraction chain
        prompt = PromptTemplate.from_template(
            """
            Extract the skills from the following resume. 
            Organize them into categories like:
            - Technical Skills
            - Soft Skills
            - Languages
            - Tools & Platforms
            
            Resume:
            {resume_text}
            
            Extracted Skills:
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        
        # Run the chain
        skills = chain.invoke({"resume_text": resume_text})
        return skills
        
    except Exception as e:
        return f"Error extracting skills: {str(e)}"


def extract_candidate_snapshot(resume_text, llm):
    """Build a recruiter-friendly candidate snapshot from resume text."""
    if not llm:
        return "LangChain LLM not available for candidate snapshot."

    try:
        prompt = PromptTemplate.from_template(
            """
            You are helping a recruiter quickly understand a candidate from a resume.

            Resume:
            {resume_text}

            Create a concise candidate snapshot with these sections:
            1. Candidate Name
            2. Current Status
            3. Target Roles
            4. Experience Level
            5. Top Skills
            6. Education
            7. One-Line Recruiter Summary

            Rules:
            - If a field is unclear, say "Not clearly stated".
            - Keep each section short and recruiter-friendly.
            """
        )

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"resume_text": resume_text})

    except Exception as e:
        return f"Error creating candidate snapshot: {str(e)}"


def extract_candidate_projects(resume_text, llm):
    """Extract and summarize candidate projects from resume text."""
    if not llm:
        return "LangChain LLM not available for project extraction."

    try:
        prompt = PromptTemplate.from_template(
            """
            You are helping a recruiter identify the most relevant projects from a candidate's resume.

            Resume:
            {resume_text}

            Extract the candidate's projects and format each one with:
            1. Project Name
            2. What the project does
            3. Technologies used
            4. Candidate contribution or impact

            Rules:
            - If project names are not explicit, provide a short descriptive name.
            - If only limited details are available, state "Not clearly stated".
            - Keep each project summary concise and recruiter-friendly.
            """
        )

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"resume_text": resume_text})

    except Exception as e:
        return f"Error extracting candidate projects: {str(e)}"


def generate_interview_focus_areas(resume_text, job_description, llm):
    """Generate practical interview focus areas for a candidate and job."""
    if not llm:
        return "LangChain LLM not available for interview focus generation."

    try:
        prompt = PromptTemplate.from_template(
            """
            You are a hiring manager preparing an interview plan.

            Resume:
            {resume_text}

            Job Description:
            {job_description}

            Provide:
            1. Top 3 areas to validate in the interview
            2. 5 tailored interview questions
            3. 3 possible risk areas or follow-up concerns

            Keep the output practical and specific to the candidate.
            """
        )

        chain = prompt | llm | StrOutputParser()
        return chain.invoke(
            {
                "resume_text": resume_text,
                "job_description": job_description,
            }
        )

    except Exception as e:
        return f"Error generating interview focus areas: {str(e)}"

def assess_resume_for_job(resume_text, job_description, llm):
    """Assess how well a resume matches a job description.
    
    Args:
        resume_text: Resume text content
        job_description: Job description text
        llm: LangChain language model
        
    Returns:
        str: Assessment or error message
    """
    if not llm:
        return "LangChain LLM not available for resume assessment."
    
    try:
        # Create an assessment chain
        prompt = PromptTemplate.from_template(
            """
            You are a skilled recruiter. Evaluate how well the following resume matches the job description.
            
            Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            
            Provide an assessment with the following sections:
            1. Match Score (0-100)
            2. Matching Skills & Qualifications
            3. Missing Skills & Qualifications
            4. Overall Assessment
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        
        # Run the chain
        assessment = chain.invoke({
            "resume_text": resume_text,
            "job_description": job_description
        })
        return assessment
        
    except Exception as e:
        return f"Error assessing resume: {str(e)}"
