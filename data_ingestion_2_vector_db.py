import os
import face_recognition
import faiss
import numpy as np
import pickle

# Load the face images -> get the face embeddings
# -> save face embeddings in vector db -> serialize the db
representations = []
path_dataset = "lfw-deepfunneled"
dirs = os.listdir(path_dataset)
dirs.sort()
count = 1
for dir in dirs:
    file_names = os.listdir(path_dataset + "/" + dir)
    for file_name in file_names:
        
        full_path_of_image = os.path.join(path_dataset, dir, file_name)
        print(f"Count: {count}, Image path: {full_path_of_image}")
        loaded_image = face_recognition.load_image_file(full_path_of_image)
        image_embedding = face_recognition.face_encodings(loaded_image)
        if len(image_embedding) > 0:
            image_embedding = image_embedding[0]
            if len(image_embedding) > 0:
                representations.append([file_name, image_embedding])
        count = count + 1

embeddings = []
for key, value in representations:
    embeddings.append(value)

print("Size of total embeddings: " + str(len(embeddings)))

# Initialize vector store  
print("Storing embeddings in faiss.") 
index = faiss.IndexFlatL2(128) 
index.add(np.array(embeddings, dtype = "f"))

# Save the index
faiss.write_index(index, "face_index.bin")

# Save the representations
with open('face_representations.txt', 'wb') as fp:
    pickle.dump(representations, fp)
print("Done")