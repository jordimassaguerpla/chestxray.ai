async function run() {
  // Import model
  image = document.getElementById("example_image");
  const model = await tf.loadLayersModel('http://localhost/ml/saved_model.tfjs/model.json');
  // Infere
  canvas = document.getElementById("canvas_1");
  // Normalize?
  training_avg = 126.0662;
  training_stdev = 63.458977;
  tensorImg = tf.browser.fromPixels(image).resizeNearestNeighbor([320, 320]).toFloat().sub(tf.scalar(training_avg)).div(tf.scalar(training_stdev));
  // tf.browser.toPixels(tensorImg, canvas);
  const prediction = model.predict(tensorImg.expandDims());
  prediction_data = prediction.dataSync();
  document.getElementById("prediction").innerText = " Cardiomegaly: " + prediction_data[1] + "\n Mass: " + prediction_data[9] + "\n Pneumotorax: " + prediction_data[12] + "\n Edema: " + prediction_data[3];
  console.log("X");
  console.log(tensorImg.expandDims().dataSync());
  console.log("prediction");
  console.log(prediction_data);
}

run();
