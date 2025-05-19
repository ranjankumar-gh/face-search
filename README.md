# Face Search
Fast face search using vector db.<br>
## Libs
1. `pip install face-recognition`
2. `pip install faiss`
3. `pip install pickle`
4. `pip install streamlit`
## Steps
1. Ingestion of face embeddings to vector db (faiss) <br>
`python data_ingestion_2_vector_db.py`
2. Search Interface (Web) <br>
`streamlit run WebApp.py`
## Screenshot of the application
![](screenshot-face-search.png)