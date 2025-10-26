declare const tf: any;
declare const tflite: any;

let model: any | null = null;

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

export async function loadModel(): Promise<void> {
  if (model) return;

  try {
    // Initialize TensorFlow.js
    await tf.ready();
    
    // Set WASM path for TFLite (from CDN)
    tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/dist/');
    
    // Prefer WebGL backend for performance
    if (tf.getBackend() !== 'webgl') {
      try { await tf.setBackend('webgl'); } catch { await tf.setBackend('cpu'); }
    }
    
    model = await tflite.loadTFLiteModel("/models/mobile_fruit_classifier.tflite");
    // Warmup to verify runtime + model work end-to-end
    try {
      const warmup = tf.zeros([1, 224, 224, 3], 'float32');
      const out = model.predict(warmup);
      await out.data?.();
      out.dispose?.();
      warmup.dispose?.();
      console.log("Model loaded and warmup successful");
    } catch (e) {
      console.error("Model warmup failed", e);
      throw e;
    }
  } catch (error) {
    console.error("Error loading model:", error);
    throw error;
  }
}

export async function classifyImage(imageData: string): Promise<string> {
  if (!model) {
    throw new Error("Model not loaded");
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = async () => {
      try {
        const tensor = tf.tidy(() => {
          const input = tf.browser.fromPixels(img)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims(0);
          return input;
        });

        const predictions = model.predict(tensor);
        const raw = (await predictions.data()) as Float32Array | number[];
        const probs = Array.from(raw as any) as number[];
        const maxIndex = probs.indexOf(Math.max(...probs));
        const predictedLabel = LABELS[maxIndex] || "Unknown";

        tensor.dispose();
        predictions.dispose?.();

        resolve(predictedLabel);
      } catch (error) {
        reject(error);
      }
    };
    img.onerror = reject;
    img.src = imageData;
  });
}
