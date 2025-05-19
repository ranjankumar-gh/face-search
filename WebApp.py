import streamlit as st
import face_recognition
import uuid
import time
import os
import faiss
import numpy as np
import pickle

is_dataset_loaded = False

# Load the face embedding from the saved face_representations.txt file 
def get_data():   
    with st.spinner("Wait for the dataset to load...", show_time=True): 
        representations = None
        with open ('face_representations.txt', 'rb') as fp:
            representations = pickle.load(fp)
        print(representations)

         # Load the index
        face_index = faiss.read_index("face_index.bin")

        return representations, face_index

# Load the face embedding at the startup and store in session
if st.button('Rerun'):
    st.session_state.representations, st.session_state.index = get_data()
if 'index' not in st.session_state:
    st.session_state.representations, st.session_state.index = get_data()
index = st.session_state.index
representations = st.session_state.representations

# Search web interface
with st.form("search-form"):
    uploaded_face_image = st.file_uploader("Choose face image for search", key="search_face_image_uploader")
    if uploaded_face_image is not None:
        tic = time.time()
        st.text("Saving the query image...")
        print("Saving the query image in the directory: " + "query-images")
        random_query_image_name = uuid.uuid4().hex
        query_image_full_path = "query-images/" + random_query_image_name + ".jpg"
        with open(query_image_full_path, "wb") as binary_file:
            binary_file.write(uploaded_face_image.getvalue())

        st.image(uploaded_face_image, caption="Image uploaded for search")

        query_image = face_recognition.load_image_file(query_image_full_path)
        query_image_embedding = face_recognition.face_encodings(query_image)
        if len(query_image_embedding) > 0:
            query_image_embedding = query_image_embedding[0]
        query_image_embedding = np.expand_dims(query_image_embedding, axis = 0)

        # Search
        st.text("Searching the images...")
        k = 1
        distances, neighbours = index.search(query_image_embedding, k)
        #print(neighbours)
        #print(distances)
        i = 0
        is_image_found = False
        for distance in distances[0]:
            if distance < 0.3:
                st.text("Found the image.")
                st.text("Similarity: " + str(distance))
                image_file_name = representations[neighbours[0][i]][0]
                image_path = "lfw-deepfunneled/" + image_file_name[:-9] + "/" + image_file_name
                st.image(image_path)
                is_image_found = True
            i = i + 1
        if is_image_found == False:
            st.text("Cound not found the image.")
        
        toc = time.time()
        st.text("Total time taken: " + str(toc - tic) + " seconds")

    st.form_submit_button('Submit')