// 64 x 3 RGB colormap.
// This is used to convert a 1-channel (grayscale) image into a color
// (RGB) one. The color map is based on the output of the "parula" colormap
// command in MATLAB.
const RGB_COLORMAP = [
  0.2422,   0.1504,  0.6603,   0.25039,   0.165,    0.70761,  0.25777,
  0.18178,  0.75114, 0.26473,  0.19776,   0.79521,  0.27065,  0.21468,
  0.83637,  0.27511, 0.23424,  0.87099,   0.2783,   0.25587,  0.89907,
  0.28033,  0.27823, 0.9221,   0.28134,   0.3006,   0.94138,  0.28101,
  0.32276,  0.95789, 0.27947,  0.34467,   0.97168,  0.27597,  0.36668,
  0.9829,   0.26991, 0.3892,   0.9906,    0.26024,  0.41233,  0.99516,
  0.24403,  0.43583, 0.99883,  0.22064,   0.46026,  0.99729,  0.19633,
  0.48472,  0.98915, 0.1834,   0.50737,   0.9798,   0.17864,  0.52886,
  0.96816,  0.17644, 0.5499,   0.95202,   0.16874,  0.57026,  0.93587,
  0.154,    0.5902,  0.9218,   0.14603,   0.60912,  0.90786,  0.13802,
  0.62763,  0.89729, 0.12481,  0.64593,   0.88834,  0.11125,  0.6635,
  0.87631,  0.09521, 0.67983,  0.85978,   0.068871, 0.69477,  0.83936,
  0.029667, 0.70817, 0.81633,  0.0035714, 0.72027,  0.7917,   0.0066571,
  0.73121,  0.76601, 0.043329, 0.7411,    0.73941,  0.096395, 0.75,
  0.71204,  0.14077, 0.7584,   0.68416,   0.1717,   0.76696,  0.65544,
  0.19377,  0.77577, 0.6251,   0.21609,   0.7843,   0.5923,   0.24696,
  0.7918,   0.55674, 0.29061,  0.79729,   0.51883,  0.34064,  0.8008,
  0.47886,  0.3909,  0.80287,  0.43545,   0.44563,  0.80242,  0.39092,
  0.5044,   0.7993,  0.348,    0.56156,   0.79423,  0.30448,  0.6174,
  0.78762,  0.26124, 0.67199,  0.77927,   0.2227,   0.7242,   0.76984,
  0.19103,  0.77383, 0.7598,   0.16461,   0.82031,  0.74981,  0.15353,
  0.86343,  0.7406,  0.15963,  0.90354,   0.73303,  0.17741,  0.93926,
  0.72879,  0.20996, 0.97276,  0.72977,   0.23944,  0.99565,  0.74337,
  0.23715,  0.99699, 0.76586,  0.21994,   0.9952,   0.78925,  0.20276,
  0.9892,   0.81357, 0.18853,  0.97863,   0.83863,  0.17656,  0.96765,
  0.8639,   0.16429, 0.96101,  0.88902,   0.15368,  0.95967,  0.91346,
  0.14226,  0.9628,  0.93734,  0.12651,   0.96911,  0.96063,  0.10636,
  0.9769,   0.9839,  0.0805
];

/**
 * Convert an input monocolor image to color by applying a color map.
 * 
 * @param {tf.Tensor4d} x Input monocolor image, assumed to be of shape
 *   `[1, height, width, 1]`.
 * @returns Color image, of shape `[1, height, width, 3]`.
 */
function applyColorMap(x) {
  tf.util.assert(
      x.rank === 4, `Expected rank-4 tensor input, got rank ${x.rank}`);
  tf.util.assert(
      x.shape[0] === 1,
      `Expected exactly one example, but got ${x.shape[0]} examples`);
  tf.util.assert(
      x.shape[3] === 1,
      `Expected exactly one channel, but got ${x.shape[3]} channels`);

  return tf.tidy(() => {
    // Get normalized x.
    const EPSILON = 1e-5;
    const xRange = x.max().sub(x.min());
    const xNorm = x.sub(x.min()).div(xRange.add(EPSILON));
    const xNormData = xNorm.dataSync();

    const h = x.shape[1];
    const w = x.shape[2];
    const buffer = tf.buffer([1, h, w, 3]);

    const colorMapSize = RGB_COLORMAP.length / 3;
    for (let i = 0; i < h; ++i) {
      for (let j = 0; j < w; ++j) {
        const pixelValue = xNormData[i * w + j];
        const row = Math.floor(pixelValue * colorMapSize);
        buffer.set(RGB_COLORMAP[3 * row], 0, i, j, 0);
        buffer.set(RGB_COLORMAP[3 * row + 1], 0, i, j, 1);
        buffer.set(RGB_COLORMAP[3 * row + 2], 0, i, j, 2);
      }
    }
    return buffer.toTensor();
  });
}


/**
 * Calculate class activation map (CAM) and overlay it on input image.
 *
 * This function automatically finds the last convolutional layer, get its
 * output (activation) under the input image, weights its filters by the
 * gradient of the class output with respect to them, and then collapses along
 * the filter dimension.
 *
 * @param {tf.Sequential} model A TensorFlow.js sequential model, assumed to
 *   contain at least one convolutional layer.
 * @param {number} classIndex Index to class in the model's final classification
 *   output.
 * @param {tf.Tensor4d} x Input image, assumed to have shape
 *   `[1, height, width, 3]`.
 * @param {number} overlayFactor Optional overlay factor.
 * @returns The input image with a heat-map representation of the class
 *   activation map overlaid on top of it, as float32-type `tf.Tensor4d` of
 *   shape `[1, height, width, 3]`.
 *   If x is an integer dtype, the resulting image 
 */
function gradClassActivationMap(model, classIndex, x, layerClassName = 'Conv', overlayFactor = 2.0) {
  // Try to locate the last BN layer of the model.
  let layerIndex = model.layers.length - 1;
  while (layerIndex >= 0) {
    if (model.layers[layerIndex].getClassName().startsWith(layerClassName)) {
      break;
    }
    layerIndex--;
  }
  tf.util.assert(
      layerIndex >= 0, `Failed to find a convolutional layer in model`);

  const lastConvLayer = model.layers[layerIndex];
  console.log(
      `Located last convolutional layer of the model at ` +
      `index ${layerIndex}: layer type = ${lastConvLayer.getClassName()}; ` +
      `layer name = ${lastConvLayer.name}`);

  // Get "sub-model 1", which goes from the original input to the output
  // of the last convolutional layer.
  const lastConvLayerOutput = lastConvLayer.output;
  const subModel1 =
      tf.model({inputs: model.inputs, outputs: lastConvLayerOutput});

  // Get "sub-model 2", which goes from the output of the last convolutional
  // layer to the original output.
  const newInput = tf.input({shape: lastConvLayerOutput.shape.slice(1)});
  layerIndex++;
  let y = newInput;
  while (layerIndex < model.layers.length) {
    y = model.layers[layerIndex++].apply(y);
  }
  const subModel2 = tf.model({inputs: newInput, outputs: y});

  return tf.tidy(() => {
    // This function runs sub-model 2 and extracts the slice of the probability
    // output that corresponds to the desired class.
    const convOutput2ClassOutput = (input) =>
        subModel2.apply(input, {training: true}).gather([classIndex], 1);
    // This is the gradient function of the output corresponding to the desired
    // class with respect to its input (i.e., the output of the last
    // convolutional layer of the original model).
    const gradFunction = tf.grad(convOutput2ClassOutput);

    // Calculate the values of the last conv layer's output.
    const lastConvLayerOutputValues = subModel1.apply(x);
    // Calculate the values of gradients of the class output w.r.t. the output
    // of the last convolutional layer.
    const gradValues = gradFunction(lastConvLayerOutputValues);

    // Pool the gradient values within each filter of the last convolutional
    // layer, resulting in a tensor of shape [numFilters].
    const pooledGradValues = tf.mean(gradValues, [0, 1, 2]);
    // Scale the convlutional layer's output by the pooled gradients, using
    // broadcasting.
    const scaledConvOutputValues =
        lastConvLayerOutputValues.mul(pooledGradValues);

    // Create heat map by averaging and collapsing over all filters.
    let heatMap = scaledConvOutputValues.mean(-1);

    // Discard negative values from the heat map and normalize it to the [0, 1]
    // interval.
    heatMap = heatMap.relu();
    heatMap = heatMap.div(heatMap.max()).expandDims(-1);

    // Up-sample the heat map to the size of the input image.
    heatMap = tf.image.resizeBilinear(heatMap, [x.shape[1], x.shape[2]]);

    // Apply an RGB colormap on the heatMap. This step is necessary because
    // the heatMap is a 1-channel (grayscale) image. It needs to be converted
    // into a color (RGB) one through this function call.
    heatMap = applyColorMap(heatMap);
    // To form the final output, overlay the color heat map on the input image.
    // If image is an integer, let's return an image between 0 an 255 
    if (x.dtype == 'int8' || x.dtype == 'int16' || x.dtype == 'int32' || x.dtype == 'int64') {
      heatMap = heatMap.mul(overlayFactor).add(x.div(255));
      // return an image that goes between 0 and 255
      heatMap = heatMap.sub(heatMap.min());
      return heatMap.div(heatMap.max()).mul(255);
    }
    else {
      heatMap = heatMap.mul(overlayFactor).add(x);
      // return an image that goes between 0 and 1
      heatMap = heatMap.sub(heatMap.min());
      return heatMap.div(heatMap.max());
    }
  });
}

function clear_canvases() {
  let canvas = document.getElementById("canvas_result_image_mass");
  let context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  canvas = document.getElementById("canvas_result_image_cardiomegaly");
  context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  canvas = document.getElementById("canvas_result_image_pneumotorax");
  context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  canvas = document.getElementById("canvas_result_image_edema");
  context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("cardiomegaly_div").style.borderColor="grey";
  document.getElementById("mass_div").style.borderColor="grey";
  document.getElementById("pneumotorax_div").style.borderColor="grey";
  document.getElementById("edema_div").style.borderColor="grey";
  document.getElementById("cardiomegaly_div_prediction").innerText = "Cardiomegaly: Calculating...";
  document.getElementById("mass_div_prediction").innerText = "Mass: Calculating...";
  document.getElementById("pneumotorax_div_prediction").innerText = "Pneumotorax: Calculating... ";
  document.getElementById("edema_div_prediction").innerText = "Edema: Calculating ";
}

function register_load_image() {
  const image_input = document.querySelector("#image-input");
  image_input.addEventListener("change", function(event) {
    let image = document.getElementById('example_image');
    image.src = URL.createObjectURL(event.target.files[0]);
    clear_canvases();
    document.getElementById("image-input").disabled = true;
    run_prediction();
  });
}

async function run_prediction() {
  // Import model
  const image = document.getElementById("example_image");
  const model = await tf.loadLayersModel('saved_model.tfjs/model.json');
  // Normalize the same way that was done for training:
  //  - scale, substract the training average value, and divide by the standard deviation. 
  const training_avg = 126.0662;
  const training_stdev = 63.458977;
  const threshold = 0.5;
  // Use resizeBilinear given this was the resize method used for training
  let tensorImg = tf.browser.fromPixels(image).resizeBilinear([320, 320]).toFloat().sub(tf.scalar(training_avg)).div(tf.scalar(training_stdev));
  const prediction = model.predict(tensorImg.expandDims());
  const prediction_data = prediction.dataSync();
  document.getElementById("cardiomegaly_div_prediction").innerText = "Cardiomegaly: " + Math.round(prediction_data[0]*100.0) + "%";
  if (prediction_data[0] >= threshold) {
    document.getElementById("cardiomegaly_div").style.borderColor="red";
  }
  document.getElementById("mass_div_prediction").innerText = "Mass: " + Math.round(prediction_data[5]*100.0) + "%";
  if (prediction_data[5] >= threshold) {
    document.getElementById("mass_div").style.borderColor="red";
  }
  document.getElementById("pneumotorax_div_prediction").innerText = "Pneumotorax: " + Math.round(prediction_data[8]*100.0) + "%";
  if (prediction_data[8] >= threshold) {
    document.getElementById("pneumotorax_div").style.borderColor="red";
  }
  document.getElementById("edema_div_prediction").innerText = "Edema: " + Math.round(prediction_data[12]*100)+"%";
  if (prediction_data[12] >= threshold) {
    document.getElementById("edema_div").style.borderColor="red";
  }
  // for DenseNet121, we want to look for the BN layer and not the last Conv one
  let cam = gradClassActivationMap(model, 0, tensorImg.expandDims(), layerClassName = 'BatchNormalization')
  // gradClassActivationMap returns a tensor of dimension [1, height, width, 3]
  // we need to use squeeze to remove the first dimension
  cam = tf.squeeze(cam);
  let canvas = document.getElementById("canvas_result_image_cardiomegaly");
  tf.browser.toPixels(cam, canvas);
  cam = gradClassActivationMap(model, 5, tensorImg.expandDims(), layerClassName = 'BatchNormalization')
  // gradClassActivationMap returns a tensor of dimension [1, height, width, 3]
  // we need to use squeeze to remove the first dimension
  cam = tf.squeeze(cam);
  canvas = document.getElementById("canvas_result_image_mass");
  tf.browser.toPixels(cam, canvas);
  cam = gradClassActivationMap(model, 8, tensorImg.expandDims(), layerClassName = 'BatchNormalization')
  // gradClassActivationMap returns a tensor of dimension [1, height, width, 3]
  // we need to use squeeze to remove the first dimension
  cam = tf.squeeze(cam);
  canvas = document.getElementById("canvas_result_image_pneumotorax");
  tf.browser.toPixels(cam, canvas);
  cam = gradClassActivationMap(model, 12, tensorImg.expandDims(), layerClassName = 'BatchNormalization')
  // gradClassActivationMap returns a tensor of dimension [1, height, width, 3]
  // we need to use squeeze to remove the first dimension
  cam = tf.squeeze(cam);
  canvas = document.getElementById("canvas_result_image_edema");
  tf.browser.toPixels(cam, canvas);
  document.getElementById("image-input").disabled = false;
}
register_load_image();
run_prediction();
