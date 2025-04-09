"""
GraphRAG
Implementing a simple GraphRAG pipeline using Neo4j and a qdrant vector database.
This will be the file to handle the preprocess of the data.
"""

import os
import PyPDF2
from dotenv import load_dotenv
import re
from datetime import datetime, timedelta
import time
import google.generativeai as genai
import json

load_dotenv()


PDF_DIR = "ML-DS-TEXTBOOKS"

# Load the pdf files

def gather_data():
    "Gather only the first pdf file in the ML-DS-TEXTBOOKS directory"
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            return file
        
def preprocess_data(file_path):
    """
    Preprocess the data from the pdf file.
    """
    text = ""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            print(f"Processing {total_pages} pages...")
            for page_num in range(total_pages):
                if page_num % 10 == 0:
                    print(f"Processing page {page_num}/{total_pages}")
                page_text = pdf_reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text + "\n\n"  # Preserve page breaks
        print("Text extraction complete")
        return text
    except Exception as e:
        print(f"Error processing file: {e}")
        return ""

def extract_entities_relationships(text):
    """Divide the text by chapters and extract entities and relationships from each chapter using google gemini 1.5 pro"""
    # defining rate limit parameters
    requests_per_minute = 2
    requests_per_day = 50
    tokens_per_minute = 32000

    #tracking usage
    requests_timestamps = []
    tokens_usage = []
    daily_requests_count = 0
    day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # First dividing text by chapters
    chapters = identify_chapters(text)
    results = []

    print(f"Found {len(chapters)} chapters. Beginning processing...")

    for i, (chapter_num, chapter_title, chapter_text) in enumerate(chapters):
        # Check daily rate limit
        current_time = datetime.now()
        if current_time - day_start > timedelta(days=1):
            day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            daily_requests_count = 0

        if daily_requests_count >= requests_per_day:
            print(f"Reached daily request limit ({requests_per_day}). Waiting until tomorrow.")
            # calculating time until midnight
            tomorrow = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            sleep_seconds = (tomorrow - current_time).total_seconds()
            time.sleep(sleep_seconds)
            daily_requests_count = 0
            day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

        # Check request rate limit
        current_time = datetime.now()
        # remove timestamps older than one minute
        requests_timestamps = [ts for ts in requests_timestamps if current_time - ts <= timedelta(minutes=1)]
        if len(requests_timestamps) >= requests_per_minute:
            # calculate how long to wait
            oldest_timestamp = min(requests_timestamps)
            wait_time = 60 - (current_time - oldest_timestamp).total_seconds()
            if wait_time > 0:
                print(f"Rate limit approaching. Waiting {wait_time:.1f} seconds before next request.")
                time.sleep(wait_time)

        # Check token rate limit
        current_time = datetime.now()
        # remove token usage older than one minute
        tokens_usage = [(ts, tokens) for ts, tokens in tokens_usage if current_time - ts < timedelta(minutes=1)]

        total_tokens_last_minute = sum(tokens for _, tokens in tokens_usage)
        #Estimating tokens for this request (rough approximation)
        estimated_tokens = len(chapter_text) / 4
        
        if total_tokens_last_minute + estimated_tokens > tokens_per_minute:
            wait_time = 60
            print(f"Token rate limit approaching. Waiting {wait_time} seconds before next request.")
            time.sleep(wait_time)
            # Clearing token usage after waiting
            token_usage = []

        # Processing the chapter
        print(f"Processing Chapter {chapter_num}: {chapter_title}")
        # formatting the prompt
        prompt = f"""
        You are a specialized data science and machine learning concepts extractor. 
        Analyze this chapter from a data science textbook and extract all data science and machine learning concepts, algorithms, techniques, and frameworks mentioned.

        Chapter: {chapter_num}
        Title: {chapter_title}

        Return ONLY a well-formatted list of data science concepts, algorithms, and techniques with no explanation or commentary.
        Format each item as a separate item (one per line).

        Here is the chapter:
        {chapter_text}
        
        """

        try:
            #calling gemini api
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            
            #updating rate limit tracking
            current_time = datetime.now()
            requests_timestamps.append(current_time)
            # rough estimation of tokens used
            approx_tokens = len(prompt) / 4 + len(response.text) / 4
            tokens_usage.append((current_time, approx_tokens))
            daily_requests_count += 1

            #processing the response to get clean list of concepts
            concepts_text = response.text

            #cleaning the response to get a list of concepts
            concepts = []
            for line in concepts_text.split("\n"):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line)):
                    concept = re.sub(r'^[-*\d.\s]+', '', line).strip()
                    if concept:
                        concepts.append(concept)

                # if not in list format, trying to capture standalone terms
                elif line and not line.startswith('#') and not re.match(r'^(chapter|here|the following)', line.lower()):
                    concepts.append(line)


            # formatting the result. I want to store the chapter number, title, and concepts as a list of dictionaries
            results.append({
                "chapter": chapter_num,
                "title": chapter_title,
                "concepts": concepts
            })

            print(f"Completed Chapter {chapter_num}. Found {len(concepts)} concepts.")

            # adding small delay to avoid rate limit
            if i < len(chapters) - 1:  # Skip delay after the last chapter
                time.sleep(1)

        except Exception as e:
            print(f"Error processing Chapter {chapter_num}: {e}")
            # Add to results with error note
            results.append(f"Chapter {chapter_num}, {chapter_title} - Error: Could not process chapter")

    return results

def save_results(results, output_file):
    """Save the results to a json file"""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")


def identify_chapters(text):
    """Identify chapter boundaries in the text"""
    chapter_pattern = r'(?m)^(?:CHAPTER|Chapter)\s+(\d+)\s*$\s*^([^\r\n]+)'
    chapter_matches = list(re.finditer(chapter_pattern, text))
    chapters = []
    current_chapter = None

    if chapter_matches:
        for i, match in enumerate(chapter_matches):
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            start_pos = match.start()
            end_pos = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            chapter_content = text[start_pos:end_pos]
            chapters.append((chapter_num, chapter_title, chapter_content))
    else:
        lines = text.split("\n")
        current_chapter = []
        current_chapter_num = None
        current_chapter_title = None

        for i, line in enumerate(lines):
            match = re.search(r'(?:CHAPTER|Chapter)\s+(\d+)[.\s]*(.+)?', line)
            if match:
                if current_chapter:
                    chapter_content = "\n".join(current_chapter)
                    chapters.append((current_chapter_num, current_chapter_title, chapter_content))
                current_chapter = [line]
                current_chapter_num = match.group(1)
                current_chapter_title = match.group(2).strip() if match.group(2) else ""
            else:
                if current_chapter_num is None:
                    chapter_match = re.search(r'(?:CHAPTER|Chapter)\s+(\d+)', line)
                    if chapter_match:
                        current_chapter_num = chapter_match.group(1)
                        current_chapter_title = ""
                if current_chapter is not None:
                    current_chapter.append(line)

        if current_chapter:
            chapter_content = "\n".join(current_chapter)
            chapters.append((current_chapter_num, current_chapter_title, chapter_content))

        if not chapters:
            print("Warning: No chapters explicitly identified. Processing entire text as one chapter.")
            chapters = [("1", "Complete Text", text)]

    return chapters 


































