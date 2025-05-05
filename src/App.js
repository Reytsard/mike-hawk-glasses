import React, { useState } from "react";
import * as ort from "onnxruntime-web";
import "./App.css";

function App() {
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);

  const generateRecommendation = async () => {
    try {
      setLoading(true);

      // Load the ONNX model
      const session = await ort.InferenceSession.create('head_shape_model.onnx');
      console.log("Model loaded successfully!");

      // Get the uploaded image file from the input
      const inputElement = document.getElementById('imageInput');
      const file = inputElement.files[0];

      if (!file) {
        console.error('No file uploaded!');
        setOutput('Please upload an image.');
        setLoading(false);
        return;
      }

      // Process the image
      const image = await loadImage(file);
      const tensor = preprocessImage(image);

      // Feed the input tensor into the model
      const feeds = { 'input': tensor };
      const results = await session.run(feeds);

      // Get the output and display it
      const output = results['output'];
      setOutput(`Model Output: ${output.data}`);
    } catch (error) {
      console.error("Error loading or running model:", error);
      setOutput('Error processing the image.');
    } finally {
      setLoading(false);
    }
  };

  // Helper function to load the image from file
  const loadImage = (file) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const reader = new FileReader();

      reader.onload = (e) => {
        img.onload = () => resolve(img);
        img.src = e.target.result;
      };

      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  // Helper function to preprocess the image (resize and normalize)
  const preprocessImage = (image) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Resize image to 28x28 (input size for the model)
    canvas.width = 28;
    canvas.height = 28;

    // Draw the image onto the canvas
    ctx.drawImage(image, 0, 0, 28, 28);

    // Get the image data (RGB values)
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const data = imageData.data;

    // Normalize and convert the image data into a Float32Array
    const inputData = new Float32Array(28 * 28 * 3);
    let index = 0;
    for (let i = 0; i < data.length; i += 4) {
      // Normalize each RGB value to the range [0, 1]
      inputData[index++] = data[i] / 255.0;    // Red
      inputData[index++] = data[i + 1] / 255.0; // Green
      inputData[index++] = data[i + 2] / 255.0; // Blue
    }

    // Return the tensor with shape [1, 28, 28, 3]
    return new ort.Tensor('float32', inputData, [1, 28, 28, 3]);
  };

  return (
    <div className="App">
      <h1>Upload an Image for Head Shape Prediction</h1>

      {/* Image upload input */}
      <input type="file" id="imageInput" accept="image/*" />
      <br /><br />

      {/* Run model button */}
      <button onClick={generateRecommendation} disabled={loading}>
        {loading ? "Running..." : "Run Model"}
      </button>

      {/* Display the output */}
      <div id="output" style={{ marginTop: '20px', fontSize: '20px' }}>
        {output ? output : "Model output will be displayed here."}
      </div>
    </div>
  );
}

export default App;
