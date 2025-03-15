import os
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from arango import ArangoClient
from langchain_community.llms import OpenAI
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.graphs import ArangoGraph

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

arango_schema = """
{'Graph Schema': [{'graph_name': 'health_graph',
   'edge_definitions': [{'edge_collection': 'health_graph_node_to_health_graph_node',
     'from_vertex_collections': ['health_graph_node'],
     'to_vertex_collections': ['health_graph_node']}]}],
 'Collection Schema': [{'collection_name': 'health_graph_node',
   'collection_type': 'document',
   'document_properties': [{'name': '_key', 'type': 'str'},
    {'name': '_id', 'type': 'str'},
    {'name': '_rev', 'type': 'str'},
    {'name': 'node_type', 'type': 'str'},
    {'name': 'name', 'type': 'str'}],
   'example_document': {'_key': '0',
    '_id': 'health_graph_node/0',
    '_rev': '_jVKqzx----',
    'node_type': 'Gender',
    'name': 'Male'}}]}
    """
sample_aql = """
    FOR patient IN health_graph_node
        FILTER patient.node_type == "Patient"
        FOR patient_encounter_edge IN health_graph_node_to_health_graph_node
            FILTER patient_encounter_edge._from == patient._id
            LET encounter = DOCUMENT(patient_encounter_edge._to)
            FILTER encounter.node_type == "Encounter"
            FOR encounter_condition_edge IN health_graph_node_to_health_graph_node
                FILTER encounter_condition_edge._from == encounter._id
                LET condition = DOCUMENT(encounter_condition_edge._to)
                FILTER condition.node_type == "Condition"
                FOR encounter_medication_edge IN health_graph_node_to_health_graph_node
                    FILTER encounter_medication_edge._from == encounter._id
                    LET medication = DOCUMENT(encounter_medication_edge._to)
                    FILTER medication.node_type == "Medication"
                    RETURN {
                        patient_name: patient.name,
                        patient_age: patient.age,
                        encounter_id: encounter._id,
                        encounter_name: encounter.name,
                        encounter_type: encounter.type,  
                        condition_name: condition.name,
                        medication_name: medication.name
                    }
    """
schema_details = """
Please note following key points :
GRAPH STRUCTURE:
    - Node Collection: health_graph_node
    - Edge Collection: health_graph_node_to_health_graph_node
Node Types: Patient, Encounter, Condition, Medication
Node Properties: 
- Patient: name
- Encounter: name, type
- Condition: name
- Medication: name
Nodes collection: health_graph_node
Edges collection: health_graph_node_to_health_graph_node

ELATIONSHIP PATTERNS:
    Patient -(HAS_ENCOUNTER)-> Encounter
    Encounter -(HAS_CONDITION|HAS_MEDICATION)-> Condition|Medication

Node Type connections:
Patient -> Encounter -> Condition  -> Medication


Please travese the graph from Patient to Medication to answer questions.
"""
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found! Make sure it's set in the .env file.")
    st.stop()

# Define ArangoDB connection details
ARANGO_URL = 'https://42b1588ff4aa.arangodb.cloud:8529'
DB_NAME = 'health'
USERNAME = 'root'
PASSWORD = 'bK0xJ38665DC62twrwB2'

# Streamlit Database Connection
@st.cache_resource
def connect_to_db():
    try:
        client = ArangoClient(hosts=ARANGO_URL)
        db = client.db(DB_NAME, username=USERNAME, password=PASSWORD, verify=True)
        return db
    except Exception as e:
        st.error(f"Failed to connect to ArangoDB: {e}")
        return None

# Query function to fetch patient data
def fetch_patient_data(db):
    query = """
    FOR patient IN health_graph_node
        FILTER patient.node_type == "Patient"
        FOR patient_encounter_edge IN health_graph_node_to_health_graph_node
            FILTER patient_encounter_edge._from == patient._id
            LET encounter = DOCUMENT(patient_encounter_edge._to)
            FILTER encounter.node_type == "Encounter"
            FOR encounter_condition_edge IN health_graph_node_to_health_graph_node
                FILTER encounter_condition_edge._from == encounter._id
                LET condition = DOCUMENT(encounter_condition_edge._to)
                FILTER condition.node_type == "Condition"
                FOR encounter_medication_edge IN health_graph_node_to_health_graph_node
                    FILTER encounter_medication_edge._from == encounter._id
                    LET medication = DOCUMENT(encounter_medication_edge._to)
                    FILTER medication.node_type == "Medication"
                    RETURN {
                        patient_name: patient.name,
                        patient_age: patient.age,
                        encounter_id: encounter._id,
                        encounter_name: encounter.name,
                        encounter_type: encounter.type,  
                        condition_name: condition.name,
                        medication_name: medication.name
                    }
    """
    return list(db.aql.execute(query)) if db else []

# Cache patient data for faster queries
@st.cache_data
def get_patient_data(_db):
    """Fetch patient data from ArangoDB."""
    return fetch_patient_data(_db)

# Function to convert text to AQL using LLM
def text_to_aql(query: str):
    """Converts natural language query to AQL using LLM"""
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)

    # ✅ Initialize ArangoClient properly
    client = ArangoClient(hosts=ARANGO_URL)
    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD, verify=True)

    # ✅ Use `db` instance directly in `ArangoGraph`
    graph = ArangoGraph(db=db)  # Removed `graph_name` argument

    # ✅ Use `graph` in ArangoGraphQAChain
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )

    query = f"Convert the following query to AQL: {query}. {schema_details} "

    result = chain.invoke(query)
    return str(result["result"])
# Function to process user queries using LLM
def process_query(query: str):
    llm = OpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY)
    tools = [
        Tool(
            name="Search ArangoDB",
            func=text_to_aql,
            description="Converts natural language queries into AQL and fetches results."
        )
    ]
    agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    response = agent.run(query)
    return response

# Sidebar Layout
with st.sidebar:
    st.image("Logo.png", use_container_width=True, output_format="PNG")
    st.header("Search Query")
    query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if query:
            response = process_query(query)
            st.write(f"Query Response: {response}")
        else:
            st.write("Please enter a query to search.")

# **Main UI Layout**
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>HealthWeave</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #7F8C8D;'>Connecting Data, Empowering Care.</h3>", unsafe_allow_html=True)

# Connect to ArangoDB
db = connect_to_db()

if db:
    st.header("Patients with Encounters, Conditions, and Medications")
    patient_data = get_patient_data(db)

    if len(patient_data) == 0:
        st.text("No data available for the selected criteria.")
    else:
        patient_df = pd.DataFrame(patient_data)
        st.write(patient_df)

        # Add filter options
        patient_names = list(patient_df['patient_name'].unique())
        selected_patient = st.selectbox("Select Patient", options=["All Patients"] + patient_names)

        condition_names = list(patient_df['condition_name'].unique())
        selected_condition = st.selectbox("Select Condition", options=["All Conditions"] + condition_names)

        medication_names = list(patient_df['medication_name'].unique())
        selected_medication = st.selectbox("Select Medication", options=["All Medications"] + medication_names)

        encounter_types = list(patient_df['encounter_type'].unique())
        selected_encounter_type = st.selectbox("Select Encounter Type", options=["All Encounter Types"] + encounter_types)

        # Filter patient data
        filtered_data = patient_data
        if selected_patient != "All Patients":
            filtered_data = [item for item in filtered_data if item['patient_name'] == selected_patient]
        if selected_condition != "All Conditions":
            filtered_data = [item for item in filtered_data if item['condition_name'] == selected_condition]
        if selected_medication != "All Medications":
            filtered_data = [item for item in filtered_data if item['medication_name'] == selected_medication]
        if selected_encounter_type != "All Encounter Types":
            filtered_data = [item for item in filtered_data if item['encounter_type'] == selected_encounter_type]

        # Create a graph visualization
        G = nx.Graph()
        color_map = []
        node_colors = {'Patient': 'skyblue', 'Encounter': 'orange', 'Condition': 'green', 'Medication': 'purple'}

        for patient in filtered_data:
            G.add_node(patient['patient_name'], type="Patient")
            G.add_node(patient['encounter_name'], type="Encounter")
            G.add_edge(patient['patient_name'], patient['encounter_name'])
            G.add_node(patient['condition_name'], type="Condition")
            G.add_edge(patient['encounter_name'], patient['condition_name'])
            G.add_node(patient['medication_name'], type="Medication")
            G.add_edge(patient['condition_name'], patient['medication_name'])

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, seed=42)

        for node, data in G.nodes(data=True):
            color_map.append(node_colors.get(data['type'], 'gray'))

        nx.draw(G, pos, with_labels=True, node_size=2000, node_color=color_map, edge_color='gray', font_size=10)

        st.subheader(f"Visualized Relationships ({selected_patient} - {selected_condition} - {selected_medication} - {selected_encounter_type})")
        st.pyplot(plt)

else:
    st.error("Could not connect to the database. Please check your credentials or database availability.")
st.write(ArangoGraph(db=db))
