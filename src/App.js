import React, { useEffect, useMemo, useState } from "react";
import * as ort from "onnxruntime-web";
import "./App.css";

function App() {
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [glassesFrameShapes, setGlassesFrameShapes] = useState([]);
  const [faceShape, setFaceShape] = useState(null);
  const [frameData, setFrameData] = useState([]);
  const [oval, setOval] = useState();
  const [heart, setHeart] = useState();
  const [oblong, setOblong] = useState();
  const [square, setSquare] = useState();
  const [round, setRound] = useState();

  useEffect(() => {
    fetchDataFrame();
  }, []);

  function fetchDataFrame() {
    fetch("./res/clustered_frames.json")
      .then((response) => response.json())
      .then((data) => {
        setFrameData(data);
      })
      .catch((error) => console.error("Error loading JSON:", error));
  }

  function getPredictedFace(value) {
    let labels = ["Oval", "Heart", "Oblong", "Square", "Round"];
    let highestIndex = 0;
    for (let i = 0; i < labels.length; i++) {
      if (value[i] >= value[highestIndex]) {
        highestIndex = i;
      }
    }
    return labels[highestIndex];
  }

  function getRecommendedStylesForFace(faceShapeLabel) {
    if (!frameData || frameData.length === 0) return [];

    const matches = frameData.filter(
      (item) =>
        item.ClusterFaceShape.toLowerCase() === faceShapeLabel.toLowerCase()
    );

    const topStyles = {};

    matches.forEach((item) => {
      const style = item.PreferredFrameStyle;
      topStyles[style] = (topStyles[style] || 0) + 1;
    });

    const sortedStyles = Object.entries(topStyles)
      .sort((a, b) => b[1] - a[1]) // sort by frequency
      .slice(0, 5) // top 5
      .map(([style]) => style);

    return sortedStyles;
  }

  const generateRecommendation = async () => {
    try {
      setLoading(true);

      const session = await ort.InferenceSession.create(
        "res/new_head_shape_model.onnx"
      );

      const inputElement = document.getElementById("imageInput");
      const file = inputElement.files[0];

      if (!file) {
        setOutput("Please upload an image.");
        setLoading(false);
        return;
      }

      const image = await loadImage(file);
      const tensor = preprocessImage(image);

      const feeds = { input: tensor };
      const results = await session.run(feeds);
      const data = results["sequential_1"].data;
      setOval(data[0] * 100);
      setHeart(data[1] * 100);
      setOblong(data[2] * 100);
      setSquare(data[3] * 100);
      setRound(data[4] * 100);
      const resultdata = getPredictedFace(results["sequential_1"].data);

      setOutput(resultdata);
      setFaceShape(resultdata);

      setfaceShapeAndRecommend(resultdata);
    } catch (error) {
      console.error("Error loading or running model:", error);
      setOutput("Error processing the image.");
    } finally {
      setLoading(false);
    }
  };

  function setfaceShapeAndRecommend(resultdata) {
    console.log(resultdata);
    const recommendedGlasses = getRecommendedStylesForFace(resultdata);
    setGlassesFrameShapes(recommendedGlasses);
  }

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

  const preprocessImage = (image) => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = 86;
    canvas.height = 86;
    ctx.drawImage(image, 0, 0, 86, 86);

    const imageData = ctx.getImageData(0, 0, 86, 86);
    const data = imageData.data;

    const inputData = new Float32Array(86 * 86 * 3);
    let index = 0;
    for (let i = 0; i < data.length; i += 4) {
      inputData[index++] = data[i] / 255.0; // R
      inputData[index++] = data[i + 1] / 255.0; // G
      inputData[index++] = data[i + 2] / 255.0; // B
    }

    return new ort.Tensor("float32", inputData, [1, 86, 86, 3]);
  };

  const glassesMemo = useMemo(() => {
    return glassesFrameShapes.map((shape, index) => (
      <div key={index}>• {shape}</div>
    ));
  }, [glassesFrameShapes]);

  return (
    <div className="App">
      <h1>Upload an Image for Head Shape Prediction</h1>
      <h3>Please upload an image with a picture ratio of 1:1</h3>

      <input type="file" id="imageInput" accept="image/*" />
      <br />
      <br />

      <button onClick={generateRecommendation} disabled={loading}>
        {loading ? "Running..." : "Run Model"}
      </button>

      <h4>Results:</h4>
      <div id="output" style={{ marginTop: "20px", fontSize: "20px" }}>
        {output ? <h2>{output}</h2> : "Model output will be displayed here."}
        {oval && <div>Oval: {oval}%</div>}
        {heart && <div>Heart: {heart}%</div>}
        {oblong && <div>Oblong: {oblong}%</div>}
        {square && <div>Square: {square}%</div>}
        {round && <div>Round: {round}%</div>}
        {output && (
          <>
            <h4>Recommended Glasses Styles:</h4>

            <p>
              You have a face shape of <strong>{output}</strong>. The
              recommended eyeglasses styles are:
            </p>
            {faceShape == "Oval" ? (
              <div>
                <h3>
                  All glasses frames are suitable for you. Just find what you
                  fancy the most.
                </h3>
              </div>
            ) : (
              glassesMemo
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
