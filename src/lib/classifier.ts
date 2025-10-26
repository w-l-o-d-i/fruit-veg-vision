// Type declarations for TensorFlow.js libraries loaded from CDN
declare const tf: any;
declare const tflite: any;

// Global interpreter instance (similar to your Colab code)
let interpreter: any | null = null;
let inputDetails: any | null = null;
let outputDetails: any | null = null;

const LABELS = [
  "Tomato 4", "Apple Red Delicious", "Tomato 3", "Huckleberry", "Blueberry",
  "Pear Red", "Banana Lady Finger", "Melon Piel de Sapo", "Pear", "Cherry 1",
  "Strawberry", "Nut Forest", "Avocado", "Tomato 2", "Pomegranate",
  "Dates", "Carambula", "Potato Red Washed", "Granadilla", "Kohlrabi",
  "Tamarillo", "Pepper Red", "Fig", "Ginger Root", "Kiwi",
  "Cherry Wax Yellow", "Lemon", "Guava", "Apple Golden 2", "Pear Stone",
  "Apple Red 1", "Cauliflower", "Mandarine", "Quince", "Strawberry Wedge",
  "Pear Monster", "Raspberry", "Pitahaya Red", "Nut Pecan", "Apple Golden 3",
  "Redcurrant", "Apple Red Yellow 1", "Pepper Yellow", "Grape Pink", "Banana Red",
  "Cucumber Ripe 2", "Physalis", "Cherry Rainier", "Maracuja", "Chestnut",
  "Plum", "Potato Sweet", "Cucumber Ripe", "Hazelnut", "Nectarine",
  "Cherry Wax Black", "Cantaloupe 2", "Lychee", "Pepper Orange", "Clementine",
  "Watermelon", "Pear Kaiser", "Mangostan", "Cherry 2", "Pineapple Mini",
  "Rambutan", "Grape White", "Tomato Yellow", "Apple Braeburn", "Tomato Maroon",
  "Onion White", "Onion Red Peeled", "Mango", "Potato White", "Apple Crimson Snow",
  "Potato Red", "Corn Husk", "Cocos", "Mulberry", "Avocado ripe",
  "Tomato 1", "Passion Fruit", "Apple Granny Smith", "Beetroot", "Kumquats",
  "Grape White 2", "Apricot", "Eggplant", "Limes", "Corn",
  "Grape White 4", "Grape White 3", "Tomato Heart", "Apple Pink Lady", "Plum 3",
  "Pear Williams", "Tomato not Ripened", "Peach 2", "Pomelo Sweetie", "Salak",
  "Grapefruit Pink", "Apple Golden 1", "Banana", "Apple Red 2", "Onion Red",
  "Physalis with Husk", "Apple Red Yellow 2", "Grape Blue", "Lemon Meyer", "Plum 2",
  "Pepino", "Tangelo", "Cactus fruit", "Papaya", "Apple Red 3",
  "Walnut", "Pear Abate", "Pear 2", "Pear Forelle", "Pineapple",
  "Tomato Cherry Red", "Cherry Wax Red", "Mango Red", "Orange", "Nectarine Flat",
  "Kaki", "Pepper Green", "Grapefruit White", "Peach", "Cantaloupe 1",
  "Peach Flat"
];

/**
 * Load and initialize the TFLite model
 * This function follows the pattern from your working Colab code
 */
export async function loadModel(): Promise<void> {
  if (interpreter) {
    console.log("Model already loaded");
    return;
  }

  try {
    console.log("Starting model loading...");
    
    // Wait for TensorFlow.js to be ready
    await tf.ready();
    console.log("TensorFlow.js ready");
    
    // Set the correct WASM path for TFLite
    // This ensures the WASM files are loaded from the CDN
    const wasmPath = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/dist/';
    tflite.setWasmPath(wasmPath);
    console.log("WASM path set to:", wasmPath);
    
    // Load the TFLite model file
    const modelPath = "/models/mobile_fruit_classifier.tflite";
    console.log("Loading model from:", modelPath);
    
    // Fetch the model file
    const modelResponse = await fetch(modelPath);
    if (!modelResponse.ok) {
      throw new Error(`Failed to fetch model: ${modelResponse.status} ${modelResponse.statusText}`);
    }
    
    const modelArrayBuffer = await modelResponse.arrayBuffer();
    console.log("Model file loaded, size:", modelArrayBuffer.byteLength, "bytes");
    
    // Create TFLite interpreter (similar to tf.lite.Interpreter in Python)
    interpreter = await tflite.loadTFLiteModel(modelPath);
    console.log("TFLite interpreter created");
    
    // Get input and output tensor details (similar to your Colab code)
    // Note: With tfjs-tflite, we don't have direct access to input_details
    // but we know the model expects [1, 224, 224, 3] float32 input
    inputDetails = {
      shape: [1, 224, 224, 3],
      dtype: 'float32'
    };
    
    outputDetails = {
      shape: [1, LABELS.length]
    };
    
    console.log("Input shape:", inputDetails.shape);
    console.log("Output shape:", outputDetails.shape);
    
    // Warmup run to verify the model works
    console.log("Performing warmup inference...");
    const warmupTensor = tf.zeros([1, 224, 224, 3], 'float32');
    const warmupOutput = interpreter.predict(warmupTensor);
    
    // Wait for the result
    await warmupOutput.data();
    
    // Clean up
    warmupTensor.dispose();
    warmupOutput.dispose();
    
    console.log("Model loaded and warmup successful!");
    
  } catch (error) {
    console.error("Error loading model:", error);
    interpreter = null;
    inputDetails = null;
    outputDetails = null;
    throw error;
  }
}

/**
 * Classify an image using the loaded TFLite model
 * This follows the same preprocessing as your Colab code:
 * - Resize to 224x224
 * - Scale to [0, 1] by dividing by 255
 * - Add batch dimension
 * - Convert to float32
 */
export async function classifyImage(imageData: string): Promise<string> {
  if (!interpreter) {
    throw new Error("Model not loaded. Call loadModel() first.");
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    
    img.onload = async () => {
      try {
        console.log("Processing image for classification...");
        
        // Preprocess image similar to your Colab code
        const tensor = tf.tidy(() => {
          // Load image from pixels
          const imageTensor = tf.browser.fromPixels(img);
          
          // Resize to 224x224 (model input size)
          const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
          
          // Convert to float32 and scale to [0, 1]
          const normalized = resized.toFloat().div(tf.scalar(255.0));
          
          // Add batch dimension: shape becomes [1, 224, 224, 3]
          const batched = normalized.expandDims(0);
          
          return batched;
        });
        
        console.log("Input tensor shape:", tensor.shape);
        console.log("Input tensor dtype:", tensor.dtype);
        
        // Run inference (similar to interpreter.invoke() in your Colab code)
        const predictions = interpreter.predict(tensor);
        
        // Get the output data
        const predictionsData = await predictions.data();
        const probabilities = Array.from(predictionsData);
        
        console.log("Predictions shape:", predictions.shape);
        console.log("Number of classes:", probabilities.length);
        
        // Find the class with maximum probability
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        const maxProbability = probabilities[maxIndex];
        const predictedLabel = LABELS[maxIndex] || "Unknown";
        
        console.log(`Predicted: ${predictedLabel} (index: ${maxIndex}, confidence: ${(maxProbability * 100).toFixed(2)}%)`);
        
        // Clean up tensors
        tensor.dispose();
        predictions.dispose();
        
        resolve(predictedLabel);
        
      } catch (error) {
        console.error("Classification error:", error);
        reject(error);
      }
    };
    
    img.onerror = (error) => {
      console.error("Image loading error:", error);
      reject(new Error("Failed to load image"));
    };
    
    // Set the image source to trigger loading
    img.src = imageData;
  });
}

/**
 * Check if the model is loaded
 */
export function isModelLoaded(): boolean {
  return interpreter !== null;
}

/**
 * Unload the model and free resources
 */
export function unloadModel(): void {
  if (interpreter) {
    // TFLite models don't have explicit disposal in tfjs-tflite
    // but we can clear our references
    interpreter = null;
    inputDetails = null;
    outputDetails = null;
    console.log("Model unloaded");
  }
}
