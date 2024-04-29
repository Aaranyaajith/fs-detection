import cv2
import numpy as np
from PIL import Image
import keras_ocr
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from roboflow import Roboflow

# Function to load the model
def load_model():
    rf = Roboflow(api_key="KcuIz5gAZ7qNkVfBw1hZ")
    project = rf.workspace().project("number-plate-detection-fcguc")
    model = project.version(2).model
    image_path = r"D:\fs_detection\myven\mamtha (1).jpg"
    return model

# Function to perform inference and save the image with bounding boxes
def perform_inference(model, image_path, output_path, confidence=40, overlap=30):
    results = model.predict(image_path, confidence=confidence, overlap=overlap)
    results.save(output_path)
    print(results)
    return results

# Function to extract ROI (Region of Interest) from an image
def extract_roi(image, prediction):
    x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
    half_width, half_height = width // 2, height // 2
    top_left_x, top_left_y = x - half_width, y - half_height
    roi = image[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
    return roi

# Function for image preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)
    cv2.imwrite('processed.jpg', img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

# Function for text recognition
def recognize_text(image_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    image = keras_ocr.tools.read(image_path)
    prediction_group = pipeline.recognize([image])
    data = prediction_group
    print(data)
    combined_string = ""
    for element in data:
        key, value = element
        if isinstance(key, str):
            key = key.upper()
        combined_string += f"{key}_{value[0]}"
    start_index = combined_string.find("'")
    end_index = combined_string.find("'", start_index + 1)
    key = combined_string[start_index + 1:end_index]
    uppercase_text = key.upper()
    print(uppercase_text)
    return uppercase_text

# Function to search data in Firestore
def search_data(collection_name, registration_number, vehicle_type=None):
  """
  Searches for a vehicle in Firestore and checks for a match in stolen vehicles.

  Args:
      collection_name: The name of the collection to search (e.g., "vehicles").
      registration_number: The registration number of the vehicle to search for.
      vehicle_type: (Optional) The vehicle type to filter by.

  Returns:
      A dictionary containing vehicle information if found, or None if not found.
      Prints a message indicating "stolen vehicle" or "fake number plate" based on checks.
  """

  cred = credentials.Certificate(r"D:\fs_detection\myven\fsds-dea3a-firebase-adminsdk-h3aaw-8f22b53efc.json")
  firebase_admin.initialize_app(cred)
  db = firestore.client()

  collection_ref = db.collection(collection_name)
  query = collection_ref.where('registrationNumber', '==', registration_number)
  if vehicle_type:
      query = query.where('vehicleType', '==', vehicle_type)

  try:
      documents = query.stream()
      results = [doc.to_dict() for doc in documents]

      if results:  # Found a match in the first search
          vehicle_data = results[0]  # Assuming only one match (modify if needed)
          stolen_vehicles_ref = db.collection("stolenVehicles")
          stolen_query = stolen_vehicles_ref.where('registrationNumber', '==', registration_number)
          stolen_docs = stolen_query.stream()
          is_stolen = len(list(stolen_docs)) > 0

          if is_stolen:
              print(f"Stolen vehicle found! Details: {vehicle_data}")
          else:
              print(f"Vehicle found, but not reported stolen. Details: {vehicle_data}")
          return vehicle_data

      else:  # No match in the first search
          print("Fake number plate detected.")
          return None

  except Exception as e:
      print(f"Error searching data: {e}")
      return None


# Main function
def main():
    image_path = r"D:\fs_detection\myven\mamtha (1).jpg"
    output_path = "prediction.jpg"

    model = load_model()
    results = perform_inference(model, image_path, output_path)

    image = cv2.imread(output_path)
    image1 = cv2.imread(image_path)
    prediction = results.predictions[0]
    roi = extract_roi(image1, prediction)
    cv2.imwrite("extracted_number_plate.jpg", roi)

    processed_img = preprocess_image("extracted_number_plate.jpg")
    uppercase_text = recognize_text("processed.jpg")

    collection_name = "vehicles"
    owner_name = input("Enter the owner name (optional, leave blank to skip): ")
    results = search_data(collection_name, uppercase_text, owner_name)

    if results:
        for result in results:
            print(result)
    else:
        print("No documents found matching the search criteria.")

# Call the main function if the script is executed
if __name__ == "__main__":
    main()
