# 8 sources with streamlit + sementic search and improved one
import spacy
import feedparser
import requests
import re
from xml.etree import ElementTree as ET
from datetime import datetime
import streamlit as st
from urllib.parse import quote
from scholarly import scholarly
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_fixed
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="Research Paper Finder", page_icon="📚", layout="wide")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load sentence transformer for semantic search
semantic_model = SentenceTransformer('./models/all-MiniLM-L6-v2')


# API Keys (replace with your own or use environment variables)
CORE_API_KEY = st.secrets["CORE_API_KEY"]  # https://core.ac.uk/services/api

def extract_keywords_and_year(query):
    doc = nlp(query)
    keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop]
    
    year_constraint = None
    # Handle "before/after/in <year>"
    year_match = re.search(r"(before|after|in)\s+(\d{4})", query, re.IGNORECASE)
    if year_match:
        condition, year = year_match.groups()
        year_constraint = {"condition": condition.lower(), "year": int(year)}
    # Handle "between <year> and <year>"
    range_match = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", query, re.IGNORECASE)
    if range_match:
        start_year, end_year = map(int, range_match.groups())
        year_constraint = {"condition": "range", "start_year": start_year, "end_year": end_year}
    # Handle "last <N> years"
    last_match = re.search(r"last\s+(\d+)\s+years?", query, re.IGNORECASE)
    if last_match:
        n_years = int(last_match.group(1))
        year_constraint = {"condition": "range", "start_year": datetime.now().year - n_years, "end_year": datetime.now().year}
    
    return keywords, year_constraint

def standardize_result(source, title, summary, link, pub_year=None):
    return {
        "source": source,
        "title": title or "No title available",
        "summary": (summary or "No summary available")[:300] + "...",
        "link": link or f"https://{source.lower().replace(' ', '')}.org",
        "pub_year": pub_year
    }

# 🔍 ArXiv
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def search_arxiv(keywords, year_constraint=None, max_results=10):
    try:
        query = "+AND+".join(keywords)
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
        feed = feedparser.parse(url)
        results = []
        for entry in feed.entries:
            pub_year = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ").year
            if year_constraint and not check_year(pub_year, year_constraint):
                continue
            results.append(standardize_result(
                "arXiv",
                entry.title,
                entry.summary,
                entry.link,
                pub_year
            ))
        time.sleep(1)  # Rate limiting
        return results
    except Exception as e:
        st.warning(f"arXiv search failed: {e}")
        return []

# 🔍 Semantic Scholar
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def search_semantic_scholar(keywords, year_constraint=None, max_results=10):
    try:
        query = " ".join(keywords)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote(query)}&limit={max_results}&fields=title,abstract,url,year"
        response = requests.get(url)
        results = []
        if response.status_code == 200:
            data = response.json()
            for paper in data.get("data", []):
                pub_year = paper.get("year")
                if year_constraint and pub_year and not check_year(pub_year, year_constraint):
                    continue
                results.append(standardize_result(
                    "Semantic Scholar",
                    paper.get("title"),
                    paper.get("abstract"),
                    paper.get("url"),
                    pub_year
                ))
        time.sleep(1)
        return results
    except Exception as e:
        st.warning(f"Semantic Scholar search failed: {e}")
        return []

# 🔍 PubMed
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def search_pubmed(keywords, year_constraint=None, max_results=10):
    try:
        query = "+".join(keywords)
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax={max_results}&term={quote(query)}"
        search_response = requests.get(search_url).json()
        ids = ",".join(search_response.get("esearchresult", {}).get("idlist", []))
        if not ids:
            return []
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
        fetch_response = requests.get(fetch_url)
        tree = ET.fromstring(fetch_response.content)
        results = []
        for article in tree.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle")
            abstract = article.findtext(".//AbstractText")
            pub_year = article.findtext(".//PubDate/Year")
            pub_year = int(pub_year) if pub_year and pub_year.isdigit() else None
            if year_constraint and pub_year and not check_year(pub_year, year_constraint):
                continue
            results.append(standardize_result(
                "PubMed",
                title,
                abstract,
                f"https://pubmed.ncbi.nlm.nih.gov/{article.findtext('.//PMID')}",
                pub_year
            ))
        time.sleep(1)
        return results
    except Exception as e:
        st.warning(f"PubMed search failed: {e}")
        return []

# 🔍 CrossRef
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def search_crossref(keywords, year_constraint=None, max_results=10):
    try:
        query = "+".join(keywords)
        url = f"https://api.crossref.org/works?query={quote(query)}&rows={max_results}"
        response = requests.get(url)
        results = []
        if response.status_code == 200:
            data = response.json()
            for item in data["message"]["items"]:
                title = item.get("title", ["No title"])[0]
                abstract = item.get("abstract", "No summary available").replace("\n", " ")
                link = item.get("URL", "https://www.crossref.org/")
                pub_year = None
                if item.get("published-print") or item.get("published-online"):
                    date_parts = (item.get("published-print") or item.get("published-online")).get("date-parts", [[None]])[0]
                    pub_year = date_parts[0]
                if year_constraint and pub_year and not check_year(pub_year, year_constraint):
                    continue
                results.append(standardize_result(
                    "CrossRef",
                    title,
                    abstract,
                    link,
                    pub_year
                ))
        time.sleep(1)
        return results
    except Exception as e:
        st.warning(f"CrossRef search failed: {e}")
        return []

# 🔍 CORE
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def search_core(keywords, year_constraint=None, max_results=10):
    try:
        query = " ".join(keywords)
        url = f"https://api.core.ac.uk/v3/search/works?q={quote(query)}&limit={max_results}&api_key={CORE_API_KEY}"
        response = requests.get(url)
        results = []
        if response.status_code == 200:
            data = response.json()
            for item in data.get("results", []):
                pub_year = item.get("yearPublished")
                if year_constraint and pub_year and not check_year(pub_year, year_constraint):
                    continue
                results.append(standardize_result(
                    "CORE",
                    item.get("title"),
                    item.get("abstract"),
                    item.get("downloadUrl") or item.get("fullTextUrl") or "https://core.ac.uk",
                    pub_year
                ))
        time.sleep(1)
        return results
    except Exception as e:
        st.warning(f"CORE search failed: {e}")
        return []

# 🔍 DOAJ
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def search_doaj(keywords, year_constraint=None, max_results=10):
    try:
        query = " ".join(keywords)
        url = f"https://doaj.org/api/search/articles/{quote(query)}?page=1&limit={max_results}"
        response = requests.get(url)
        results = []
        if response.status_code == 200:
            data = response.json()
            for item in data.get("results", []):
                pub_year = item.get("bibjson", {}).get("year")
                pub_year = int(pub_year) if pub_year and pub_year.isdigit() else None
                if year_constraint and pub_year and not check_year(pub_year, year_constraint):
                    continue
                results.append(standardize_result(
                    "DOAJ",
                    item.get("bibjson", {}).get("title"),
                    item.get("bibjson", {}).get("abstract"),
                    item.get("bibjson", {}).get("link", [{}])[0].get("url", "https://doaj.org"),
                    pub_year
                ))
        time.sleep(1)
        return results
    except Exception as e:
        st.warning(f"DOAJ search failed: {e}")
        return []

# 🔍 Google Scholar
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def search_google_scholar(keywords, year_constraint=None, max_results=10):
    try:
        query = " ".join(keywords)
        search_query = scholarly.search_pouts(query)
        results = []
        for i, pub in enumerate(search_query):
            if i >= max_results:
                break
            pub_year = pub.get("bib", {}).get("pub_year")
            pub_year = int(pub_year) if pub_year and str(pub_year).isdigit() else None
            if year_constraint and pub_year and not check_year(pub_year, year_constraint):
                continue
            results.append(standardize_result(
                "Google Scholar",
                pub.get("bib", {}).get("title"),
                pub.get("bib", {}).get("abstract"),
                pub.get("eprint_url") or pub.get("pub_url") or "https://scholar.google.com",
                pub_year
            ))
        time.sleep(1)
        return results
    except Exception as e:
        st.warning(f"Google Scholar search failed: {e}")
        return []

# 🔍 ERIC
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def search_eric(keywords, year_constraint=None, max_results=10):
    try:
        query = "+".join(keywords)
        search_url = f"https://api.ies.ed.gov/eric/?search={quote(query)}&format=json&rows={max_results}"
        response = requests.get(search_url)
        results = []
        if response.status_code == 200:
            data = response.json()
            for item in data.get("response", {}).get("docs", []):
                pub_year = item.get("publicationdate")
                pub_year = int(pub_year[:4]) if pub_year and pub_year[:4].isdigit() else None
                if year_constraint and pub_year and not check_year(pub_year, year_constraint):
                    continue
                results.append(standardize_result(
                    "ERIC",
                    item.get("title"),
                    item.get("description"),
                    item.get("url", "https://eric.ed.gov"),
                    pub_year
                ))
        time.sleep(1)
        return results
    except Exception as e:
        st.warning(f"ERIC search failed: {e}")
        return []

# Helper function to check year constraint
def check_year(pub_year, year_constraint):
    condition = year_constraint["condition"]
    if condition == "before":
        return pub_year < year_constraint["year"]
    elif condition == "after":
        return pub_year > year_constraint["year"]
    elif condition == "in":
        return pub_year == year_constraint["year"]
    elif condition == "range":
        return year_constraint["start_year"] <= pub_year <= year_constraint["end_year"]
    return True

# Semantic search scoring
def compute_semantic_score(query, results):
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    for result in results:
        text = f"{result['title']} {result['summary']}"
        text_embedding = semantic_model.encode(text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, text_embedding).item()
        result["semantic_score"] = score
    return results

# Keyword-based ranking
def rank_results(results, keywords):
    for result in results:
        score = sum(1 for keyword in keywords if keyword.lower() in result["title"].lower() or keyword.lower() in result["summary"].lower())
        result["keyword_score"] = score
    return sorted(results, key=lambda x: (x.get("semantic_score", 0) + x.get("keyword_score", 0)), reverse=True)

# Deduplicate results by title
def deduplicate_results(results):
    seen_titles = set()
    unique_results = []
    for result in results:
        title = result["title"].lower().strip()
        if title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(result)
    return unique_results

# Citation export
def export_to_csv(results):
    df = pd.DataFrame(results)
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "papers.csv", "text/csv")

# 🔧 Fetch Papers
def fetch_all_papers(user_input, max_papers):
    if max_papers <= 0:
        return [], "Please specify a positive number of papers."

    keywords, year_constraint = extract_keywords_and_year(user_input)
    search_info = f"Searching for: {', '.join(keywords)}  \n"
    if year_constraint:
        if year_constraint["condition"] == "range":
            search_info += f"Filtering by year: between {year_constraint['start_year']} and {year_constraint['end_year']}"
        else:
            search_info += f"Filtering by year: {year_constraint['condition']} {year_constraint['year']}"
    else:
        search_info += "No year constraint specified."

    per_source_limit = max_papers * 2
    search_functions = [
        (search_arxiv, keywords, year_constraint, per_source_limit),
        (search_semantic_scholar, keywords, year_constraint, per_source_limit),
        (search_pubmed, keywords, year_constraint, per_source_limit),
        (search_crossref, keywords, year_constraint, per_source_limit),
        (search_core, keywords, year_constraint, per_source_limit),
        (search_doaj, keywords, year_constraint, per_source_limit),
        (search_google_scholar, keywords, year_constraint, per_source_limit),
        (search_eric, keywords, year_constraint, per_source_limit),
    ]

    all_results = []
    progress_bar = st.progress(0)
    with ThreadPoolExecutor() as executor:
        for i, result in enumerate(executor.map(lambda f: f[0](f[1], f[2], f[3]), search_functions)):
            all_results.extend(result)
            progress_bar.progress((i + 1) / len(search_functions))

    all_results = deduplicate_results(all_results)
    all_results = compute_semantic_score(user_input, all_results)
    all_results = rank_results(all_results, keywords)
    all_results = all_results[:max_papers]
    progress_bar.empty()

    return all_results, search_info

# Streamlit UI
def main():
    st.title("📚 Research Paper Finder")
    st.write("Search for academic papers across 8 sources: arXiv, Semantic Scholar, PubMed, CrossRef, CORE, DOAJ, Google Scholar, and ERIC.")

    # Input form
    with st.form(key="search_form"):
        user_query = st.text_input("Enter your Research Topic or keywords related to required research paper", "")
        max_papers = st.number_input("Number of Papers", min_value=1, max_value=50, value=5)
        submit_button = st.form_submit_button(label="Search")

    if submit_button and user_query:
        with st.spinner("Searching for papers..."):
            results, search_info = fetch_all_papers(user_query, int(max_papers))
        
        st.markdown("### Search Details")
        st.markdown(search_info)

        if not results:
            st.error("No papers found matching your criteria.")
        else:
            # Export options
            st.markdown("### Export your result here")
            export_to_csv(results)

            # Display results
            st.markdown(f"### Top {min(max_papers, len(results))} Research Papers")
            for i, result in enumerate(results):
                st.markdown(f"**{i+1}. [{result['source']}] {result['title']}**")
                st.markdown(f"**Summary**: {result['summary']}")
                st.markdown(f"**Link**: [{result['link']}]({result['link']})")
                st.markdown("---")

if __name__ == "__main__":
    main()