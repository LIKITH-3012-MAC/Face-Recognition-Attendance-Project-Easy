import face_recognition
import chromadb
from pathlib import Path


# --------------------------------
# 1. Project paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "Training_images" / "likith.jpeg"

VECTOR_DB_PATH = BASE_DIR / "face_vector_db"


# --------------------------------
# 2. Connect to persistent Vector DB
# --------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client.get_or_create_collection(
    name="students"
)


# --------------------------------
# 3. Load student's image
# --------------------------------

print("Loading image:", IMAGE_PATH)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image = face_recognition.load_image_file(
    str(IMAGE_PATH)
)


# --------------------------------
# 4. Generate face embedding
# --------------------------------

encodings = face_recognition.face_encodings(image)

if len(encodings) == 0:
    raise Exception("No face found in the image")

if len(encodings) > 1:
    raise Exception("Multiple faces found in the image")

embedding = encodings[0]


print("\nEmbedding generated successfully!")
print("Dimensions:", len(embedding))
print("First 5 values:", embedding[:5])


# --------------------------------
# 5. Store embedding permanently
# --------------------------------

collection.upsert(
    ids=["23CS001"],

    embeddings=[
        embedding.tolist()
    ],

    metadatas=[
        {
            "student_id": "23CS001",
            "name": "Likith Naidu",
            "department": "CSE-AI"
        }
    ]
)


print("\nEmbedding stored successfully!")

print("Student ID:", "23CS001")
print("Collection:", collection.name)
print("Vector DB:", VECTOR_DB_PATH)


# --------------------------------
# 6. Verify that it was stored
# --------------------------------

result = collection.get(
    ids=["23CS001"]
)
print(result)
print("\nVerification:")
print("Stored IDs:", result["ids"])
print("Stored metadata:", result["metadatas"])

if result["ids"]:
    print("✅ 128-D face embedding is permanently stored!")