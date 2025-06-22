# app.py
import spacy
import re
import os
from neo4j import GraphDatabase
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict, Tuple
from pyvis.network import Network
import streamlit.components.v1 as components

# Configuration - MUST be first Streamlit command
st.set_page_config(page_title="eBay Policy Assistant", layout="wide")

# Constants
PDF_PATH = r"C:\Users\rishi\OneDrive\Desktop\New folder (3)\Ebay user agreement.pdf"

# --- Core Functions ---
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_lg")
    except Exception as e:
        st.error(f"Error loading NLP model: {e}")
        return None

@st.cache_resource
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(
            "neo4j+s://1c1278ac.databases.neo4j.io",
            auth=("neo4j", "c57deRLWLEuNtqhbic4LofMdE-UtGMiZ_Nh0MZvdFAo")
        )
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

@st.cache_resource
def get_llm():
    try:
        return ChatGroq(
            temperature=0,
            model="llama-3.3-70b-versatile",
            api_key="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"
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

# --- Knowledge Graph Visualization ---
def visualize_knowledge_graph(driver):
    """Create interactive graph visualization using pyvis"""
    if not driver:
        st.warning("Cannot visualize graph - Neo4j connection unavailable")
        return
    
    try:
        with driver.session() as session:
            result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN a.name AS source, 
                   type(r) AS relation, 
                   b.name AS target,
                   r.sentence AS description
            LIMIT 100
            """)
            
            data = [dict(record) for record in result]
            
            if not data:
                st.warning("No graph data found to visualize")
                return
            
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
            
            nodes = set()
            for row in data:
                source = row['source']
                target = row['target']
                relation = row['relation']
                description = row.get('description', '')
                
                if source not in nodes:
                    net.add_node(source, title=source, color="#97c2fc")
                    nodes.add(source)
                if target not in nodes:
                    net.add_node(target, title=target, color="#97c2fc")
                    nodes.add(target)
                
                net.add_edge(source, target, title=f"{relation}\n\n{description}", label=relation)
            
            net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200)
            net.save_graph("knowledge_graph.html")
            
            with open("knowledge_graph.html", "r", encoding="utf-8") as f:
                html = f.read()
            
            components.html(html, height=600, scrolling=True)
            
    except Exception as e:
        st.error(f"Graph visualization failed: {e}")

# --- Knowledge Graph Operations ---
def process_pdf_to_graph(nlp, driver, file_path: str):
    """Process PDF and insert triples into Neo4j"""
    if not os.path.exists(file_path):
        st.error(f"PDF file not found at: {file_path}")
        return False

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        all_triples = []
        for doc in documents:
            text = re.sub(r'=+ Page \d+ =+\n', '', doc.page_content)
            text = re.sub(r'^.*(eBay|User Agreement).*\n', '', text, flags=re.MULTILINE)
            text = re.sub(r'\s+', ' ', text).strip()
            
            doc = nlp(text)
            for sent in doc.sents:
                if len(sent) < 5: continue
                
                for verb in [t for t in sent if t.pos_ == "VERB" and t.dep_ in ("ROOT", "acl", "relcl")]:
                    subj = next((t.text for t in verb.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
                    obj = find_object(verb)
                    if subj and obj:
                        all_triples.append({
                            "subject": subj,
                            "relation": verb.lemma_.lower(),
                            "object": obj,
                            "sentence": sent.text
                        })

        if not all_triples:
            st.warning("No triples extracted from document")
            return False

        with driver.session() as session:
            for triple in all_triples:
                session.run("""
                MERGE (a:Entity {name: $subject})
                MERGE (b:Entity {name: $object})
                MERGE (a)-[r:%s]->(b)
                SET r.sentence = $sentence
                """ % triple["relation"].upper().replace(" ", "_"),
                subject=triple["subject"],
                object=triple["object"],
                sentence=triple["sentence"])

        st.success(f"Inserted {len(all_triples)} triples into knowledge graph")
        return True

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

def find_object(verb):
    for token in verb.rights:
        if token.dep_ in ("dobj", "attr"):
            return token.text
        if token.dep_ == "prep":
            return next((t.text for t in token.children if t.dep_ == "pobj"), None)
    return None

# --- Query Processing ---
def generate_cypher(nlp, question: str) -> str:
    """Generate Cypher query from natural language question"""
    if not nlp:
        return None

    try:
        doc = nlp(question.lower())
        
        # Extract key components
        entities = [ent.text for ent in doc.ents]
        keywords = [token.text for token in doc 
                   if token.pos_ in ("NOUN", "PROPN") and token.text not in ("policy", "agreement")]
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        
        # Build query conditions
        conditions = []
        if entities:
            conditions.append(f"(toLower(a.name) CONTAINS '{entities[0]}' OR toLower(b.name) CONTAINS '{entities[0]}')")
        if keywords:
            conditions.append(f"(toLower(a.name) CONTAINS '{keywords[0]}' OR toLower(b.name) CONTAINS '{keywords[0]}')")
        if verbs:
            conditions.append(f"toLower(type(r)) CONTAINS '{verbs[0]}'")
        if "policy" in question.lower():
            conditions.append("toLower(type(r)) CONTAINS 'policy'")
        
        if not conditions:
            return None
            
        return f"""
        MATCH (a)-[r]->(b)
        WHERE {' OR '.join(conditions)}
        RETURN a.name AS subject, type(r) AS relation, b.name AS object
        LIMIT 10
        """
    except Exception as e:
        st.error(f"Query generation error: {e}")
        return None

def query_graph(driver, query: str):
    """Execute query with better error handling"""
    if not driver:
        return None
        
    try:
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None

# --- Response Generation ---
def generate_response(llm, context: str, question: str) -> str:
    """Generate more contextual responses"""
    if not llm:
        return "LLM service unavailable"
    
    prompt = f"""You are an eBay policy expert. Use ONLY these facts:
    
{context}

Question: {question}

Guidelines:
1. If the answer isn't in the facts, say "The policy doesn't specify"
2. For policy questions, cite specific sections if available
3. Be precise about conditions and requirements
4. Never invent information

Answer:"""
    
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --- Main Application ---
def main():
    # Initialize services
    nlp = load_nlp_model()
    driver = get_neo4j_driver()
    llm = get_llm()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.show_graph = False
        if os.path.exists(PDF_PATH):
            with st.spinner("Building knowledge graph..."):
                if process_pdf_to_graph(nlp, driver, PDF_PATH):
                    st.session_state.graph_loaded = True
    
    # Sidebar
    with st.sidebar:
        st.header("eBay Policy Assistant")
        st.write("Query the knowledge graph")
        if st.button("Visualize Knowledge Graph"):
            st.session_state.show_graph = True
        if st.button("Hide Graph"):
            st.session_state.show_graph = False
        if st.button("Clear Chat"):
            st.session_state.messages = []
    
    # Main interface
    st.title("eBay Policy Knowledge Base")
    
    # Show knowledge graph visualization if toggled
    if st.session_state.get("show_graph", False):
        st.header("Knowledge Graph Visualization")
        visualize_knowledge_graph(driver)
        st.markdown("---")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask about eBay policies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not driver or not llm:
                response = "Backend services unavailable"
            else:
                # Generate and execute query
                cypher = generate_cypher(nlp, prompt)
                facts = query_graph(driver, cypher) if cypher else None
                
                # Fallback for policy questions
                if not facts and "policy" in prompt.lower():
                    facts = query_graph(driver, """
                        MATCH (a)-[r]->(b)
                        WHERE toLower(type(r)) CONTAINS 'policy'
                        RETURN a.name AS subject, type(r) AS relation, b.name AS object
                        LIMIT 5""")
                
                # Format context
                context = "No relevant facts found" if not facts else "\n".join(
                    f"- {f['subject']} {f['relation']} {f['object']}" for f in facts)
                
                # Generate response
                response = generate_response(llm, context, prompt)
                
                # Show context if available
                if facts:
                    with st.expander("View knowledge used"):
                        st.write(context)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()