import face_recognition
import os
import pickle

KNOWN_FACES_DIR = "known_faces"
MODEL = "hog" # or "cnn" for more accuracy
ENCODINGS_PATH = "trusted_faces.pkl"

print("Starting enrollment process...")
known_faces_encodings = []
known_faces_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name)):
        print(f"Processing images for: {name}")
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
            image_path = f"{KNOWN_FACES_DIR}/{name}/{filename}"
            image = face_recognition.load_image_file(image_path)

            # Find face locations
            face_locations = face_recognition.face_locations(image, model=MODEL)

            if len(face_locations) == 1:
                # Encode the face
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_faces_encodings.append(encoding)
                known_faces_names.append(name)
                print(f"  - Encoded {filename} for {name}")
            else:
                print(f"  - Warning: Found {len(face_locations)} faces in {filename}. Skipping.")

# Save the encodings
data = {"encodings": known_faces_encodings, "names": known_faces_names}
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"\nEnrollment complete. Data saved to {ENCODINGS_PATH}")
