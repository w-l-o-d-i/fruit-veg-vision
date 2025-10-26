import * as tf from "@tensorflow/tfjs";
import { loadTFLiteModel, TFLiteModel } from "@tensorflow/tfjs-tflite";

let model: TFLiteModel | null = null;

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
    await tf.ready();
    model = await loadTFLiteModel("/models/mobile_fruit_classifier.tflite");
    console.log("Model loaded successfully");
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
        // Preprocess image
        const tensor = tf.browser
          .fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .div(255.0)
          .expandDims(0);

        // Run prediction
        const predictions = model!.predict(tensor) as tf.Tensor;
        const probabilities = await predictions.data();
        
        // Get top prediction
        const maxIndex = probabilities.indexOf(Math.max(...Array.from(probabilities)));
        const predictedLabel = LABELS[maxIndex] || "Unknown";

        // Clean up
        tensor.dispose();
        predictions.dispose();

        resolve(predictedLabel);
      } catch (error) {
        reject(error);
      }
    };
    img.onerror = reject;
    img.src = imageData;
  });
}
