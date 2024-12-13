from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests

app = Flask(__name__)

# Load the pre-trained model
model_path = r"C:\Users\Mayuri\Desktop\Medical plant identification\medicinal_plant_classifier.h5"
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the correct class labels
class_labels = [
    'Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avacado', 'Bamboo', 
    'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 
    'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 
    'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya', 
    'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel'
]

# Google Custom Search API
api_key = 'AIzaSyC2VscsBglCj0gVc5aRkTxnsKJhusebyvY'  # Replace with your API key
cx = 'c61fc0dc5172542e1'  # Custom Search Engine ID

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded image
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    # Perform Google Custom Search API for more information
    search_query = f"Information about {predicted_class_label}"
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={search_query}"
    response = requests.get(url)

    search_results = []
    if response.status_code == 200:
        data = response.json()
        items = data.get("items", [])
        if items:
            for item in items[:3]:  # Limit to first 3 results
                title = item.get("title")
                link = item.get("link")
                search_results.append({'title': title, 'link': link})
        else:
            search_results.append({"title": "No search results found.", "link": ""})
    else:
        search_results.append({"title": "Error performing search.", "link": ""})

    return render_template('result.html', predicted_class=predicted_class_label, search_results=search_results)

@app.route('/cancel')
def cancel():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
