💻 eBay Policy Knowledge Graph Chatbot

An intelligent policy assistant that leverages a Knowledge Graph built from the eBay User Agreement PDF. It uses Natural Language Processing (NLP), Neo4j for graph storage, and an LLM (LLaMA-3.3-70B via Groq API) to answer user questions grounded in policy facts.

⚖️ End-to-End Architecture

PDF File --> SpaCy (NER & Dependency Parsing)
         --> Triples Extraction (Subject, Relation, Object)
         --> Neo4j Graph Insertion (Cypher)
         --> LLM Prompting via LangChain + Groq API
         --> Streamlit UI + Pyvis Graph Visualizer

🏛️ How the Knowledge Graph is Built and Stored

PDF Loading: The eBay User Agreement is loaded using PyPDFLoader from LangChain.

Text Cleaning: Headers/footers are stripped using regex.

Triple Extraction: SpaCy's dependency parser identifies subjects, verbs, and objects to create (subject, relation, object) triples.

Neo4j Insertion:

MERGE ensures nodes and edges are unique

Relationship verbs are converted to uppercase with underscores (e.g. PROVIDE, AGREE_TO)

A descriptive sentence is stored as a property on each edge

🔍 How Questions Are Translated into Graph Queries

User Input is processed using SpaCy.

Entity & Verb Extraction is used to generate a Cypher MATCH query dynamically.

Cypher Example:

MATCH (a)-[r]->(b)
WHERE toLower(a.name) CONTAINS 'ebay' AND toLower(type(r)) CONTAINS 'provide'
RETURN a.name AS subject, type(r) AS relation, b.name AS object

Results are formatted into bullet point context for the LLM prompt.

🔍 Prompting Strategy

Prompts are grounded and constrained:

You are an eBay policy expert. Use ONLY these facts:

- EBAY PROVIDE services
- YOU AGREE user

Question: What does eBay provide?

Guidelines:
1. If the answer isn't in the facts, say "The policy doesn't specify"
2. For policy questions, cite specific sections if available
3. Be precise about conditions and requirements
4. Never invent information

Answer:

This ensures factual, policy-bound answers.

🌟 How to Run the Chatbot

Requirements

Python 3.10+

Install dependencies:

pip install -r requirements.txt

Run the App

streamlit run app.py

On First Run

The PDF is parsed and triples are inserted into Neo4j

The knowledge graph is visualized with Pyvis

UI Features

✅ Chat Input

✅ Dynamic LLM Responses

✅ Knowledge Context Viewer

✅ Graph Visualization Toggle

Screenshot

https://github.com/1009rishit/knowledge_graph_chatbot/blob/ab8b35dbe0f066b5e219f51f80612479d54d05ee/Screenshot%202025-06-22%20161206.png
![Screenshot 2025-06-22 161225](https://github.com/user-attachments/assets/303a0e8e-1810-438f-9150-81df09ffb299)


🤖 Model Details & Streaming

Model: LLaMA-3.3-70B (Groq-hosted)

LangChain Wrapper: ChatGroq

Streaming: Currently disabled; synchronous call via .invoke()

Why LLaMA-3.3-70B? Fast inference via Groq and accurate grounding in factual prompts

📁 Project Structure

.
├── app.py              # Streamlit UI & logic
├── extracted_triples.txt (optional)
├── knowledge_graph.html  # Pyvis graph output
├── README.md
├── requirements.txt
