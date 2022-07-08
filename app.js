function load_image() {
  const image_input = document.querySelector("#image-input");
  image_input.addEventListener("change", function(event) {
    var image = document.getElementById('example_image');
    image.src = URL.createObjectURL(event.target.files[0]);
    document.getElementById('prediction').innerText = "Calculating prediction...";
    run_prediction();
  });
}

async function run_prediction() {
  // Import model
  image = document.getElementById("example_image");
  const model = await tf.loadLayersModel('http://localhost/ml/saved_model.tfjs/model.json');
  // Infere
  canvas = document.getElementById("canvas_1");
  // Normalize
  training_avg = 126.0662;
  training_stdev = 63.458977;
  // resizeBilinear given this is the default for Keras https://keras.io/api/preprocessing/image/
  tensorImg = tf.browser.fromPixels(image).resizeBilinear([320, 320]).toFloat().sub(tf.scalar(training_avg)).div(tf.scalar(training_stdev));
  const prediction = model.predict(tensorImg.expandDims());
  prediction_data = prediction.dataSync();
  document.getElementById("prediction").innerText = " Cardiomegaly: " + Math.round(prediction_data[0]*100.0) + "%\n Mass: " + Math.round(prediction_data[5]*100.0) + "%\n Pneumotorax: " + Math.round(prediction_data[8]*100.0) + "%\n Edema: " + Math.round(prediction_data[12]*100)+"%";
}

load_image();
run_prediction();
