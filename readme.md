AI Powered Research Paper Finder -- (Your ultimate Research Paper Finder)

Welcome to AI Powered Research Paper Finder, a powerful Streamlit web application designed to streamline academic research by searching for papers across eight reputable sources: arXiv, Semantic Scholar, PubMed, CrossRef, CORE, DOAJ, Google Scholar, and ERIC. This tool leverages natural language processing (NLP) and semantic search to deliver relevant, high-quality results tailored to your research needs.
📖 About
Research Paper Finder simplifies the process of discovering academic papers by combining keyword extraction, semantic ranking, and year-based filtering. Whether you're a student, researcher, or professional, this app helps you find papers efficiently with features like:

Multi-Source Search: Query eight academic databases in one go.
Semantic Ranking: Uses the all-MiniLM-L6-v2 model to rank papers based on relevance to your query.
Smart Keyword Extraction: Extracts key terms from your query using SpaCy for precise searches.
Flexible Year Filters: Narrow results with constraints like "after 2020", "before 2015", "between 2010 and 2020", or "last 5 years".
Deduplication: Ensures unique results by removing duplicate titles.
CSV Export: Download your search results for easy reference.

🚀 Features

Intuitive Interface: Built with Streamlit for a seamless user experience.
Fast and Concurrent Searches: Uses ThreadPoolExecutor for parallel API queries.
Robust Error Handling: Implements retries with tenacity to manage API failures.
Customizable Results: Choose how many papers to retrieve (1–50).
Open-Source: Fully customizable and deployable on Streamlit Community Cloud.


🎯 Usage

Open the app (local or deployed).
Enter a research topic (e.g., "artificial intelligence in education last 5 years").
Select the number of papers to retrieve.
Click Search to view results with titles, summaries, links, and years.
Export results as a CSV file for further analysis.


🤝 Contributing
Contributions are welcome! To contribute:


Report bugs or suggest features via GitHub Issues.
📜 License
This project is licensed under the MIT License.
🙌 Acknowledgments

Built with Streamlit for an awesome web framework.
Powered by SpaCy and Sentence Transformers for NLP and semantic search.
Thanks to the open APIs of arXiv, Semantic Scholar, PubMed, CrossRef, CORE, DOAJ, Google Scholar, and ERIC.


Happy researching! 📚
