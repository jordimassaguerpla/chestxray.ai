# SUSE Hack Week 2022

This is the result of this SUSE Hack Week project:
https://hackweek.opensuse.org/projects/chest-x-ray-medical-diagnosis-with-deep-learning-and-javascript

The goal was to learn TensorflowJS and to put on practice what I have learn in week 1 of https://www.coursera.org/learn/ai-for-medical-diagnosis.

## Resources used

The dataset used is ChestX-ray8: Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers.ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017@InProceedings{wang2017chestxray,author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald},title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},booktitle = {2017 IEEE Conference on Computer Vision and Pattern Recognition(CVPR)},pages     = {3462--3471},year      = {2017}}

NIH Clinical Center , America's Research Hospital: https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774

Other resources:
   - https://www.coursera.org/learn/ai-for-medical-diagnosis
   - https://nihcc.app.box.com/v/ChestXray-NIHCC
   - https://observablehq.com/[@tvirot](/users/tvirot)/machine-learning-in-javascript-with-tensorflow-js-part-ii
   - https://www.tensorflow.org/js/tutorials/conversion/import_keras

## Model

This is using the saved model that was trained as an exercise in the first week of https://www.coursera.org/learn/ai-for-medical-diagnosis. Then performs the same transformations that were done during training (normalization) to n X-Ray Image, 00008270_015.png, and computes a prediction. All this is done with javascript, meaning that we are using the hardware (CPU/GPU) from the user.

Note I can't publish the saved model as I do not have explicit permission to do so. This is why I can't publish the webpage.

To save the model I had to add this to the coursera assignment:
```model.save("models/nih/saved_model.h5")```

To transform the model from keras to tensorflowjs:

```
docker run -it --rm -v `pwd`:/python evenchange4/docker-tfjs-converter tensorflowjs_converter --input_format=keras /python/save_model.h5 /python/saved_model.tfjs
```

## Resources needed

What is interesting is that using a saved model with tensorflowjs, **we do not need a "big server"**, but just an apache to serve the files, that could also be distributed with a Content Delivery Network (CDN). The training was already done on a big cluster and the prediction is done on the user's browser.

## Results

This is a screenshot of the resulting page:


![image](https://user-images.githubusercontent.com/1148215/176908228-ffae44cc-92b8-4d62-af8c-c22b6a3bce6d.png)

