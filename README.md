# ChestXRay AI

> :warning: :warning: DISCLAIMER: This is a project to demonstrate the usage of [Javascript Tensorflow](https://www.tensorflow.org/js/). Do NOT use this for REAL for Medical Diagnosis. :warning: :warning:

This is a very simple page that lets you upload an X-Ray image of you Chest, and predicts the probability of having any of these 4 diseases:

* Cardiomegaly
* Mass
* Pneumotorax
* Edema

![image](https://user-images.githubusercontent.com/1148215/178037624-4cf14e28-ba8b-4907-8dda-191c17a2ab54.png)

## Privacy

X-Ray images are never uploaded to a server. Instead, all the calculations are done in the browser, meaning your X-Ray image never leaves your computer, and thus it is not shared with anyone else.

## Credits

This project was initially developed at the [21st SUSE Hackweek project](https://hackweek.opensuse.org/projects/chest-x-ray-medical-diagnosis-with-deep-learning-and-javascript), and based on [DeepLearning.AI](https://deeplearning.ai) [AI for Medical Diagnosis course](https://www.coursera.org/learn/ai-for-medical-diagnosis).

Thanks to [SUSE](https://www.suse.com) for letting me hack on this project for a week.
Thanks to [DeepLearning.AI](https://deeplearning.ai) for such brilliant course, and the permission to use and publish the model for non-comercial purposes.

## Dataset

The dataset used is the result of this research paper: [Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers.ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases, IEEE CVPR, pp. 3462-3471,2017@InProceedings](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774).

The dataset is available at the [NIH Clinical Center, America's Research Hospital](https://nihcc.app.box.com/v/ChestXray-NIHCC)

This dataset has been annotated by consensus among four different radiologists for 5 of our 14 pathologies:

   * Consolidation
   * Edema
   * Effusion
   * Cardiomegaly
   * Atelectasis

## Model

This is using the saved model that was trained as an exercise in the first week of [AI for Medical Diagnosis](https://www.coursera.org/learn/ai-for-medical-diagnosis). Then, it performs the same transformations that were done during training to an X-Ray Image: scaling and normalization, and computes a prediction. All this is done with javascript, meaning that we are using the hardware (CPU/GPU) from the user.

To save the model I had to add this to the coursera assignment:
```model.save("models/nih/saved_model.h5")```

To transform the model from keras to tensorflowjs:

```
docker run -it --rm -v `pwd`:/python evenchange4/docker-tfjs-converter tensorflowjs_converter --input_format=keras /python/save_model.h5 /python/saved_model.tfjs
```
This model was implemented and trained by [DeepLearning.AI](https://deeplearning.ai).

## Model Metrics

See https://github.com/jordimassaguerpla/model.chestxray.ai/blob/main/metrics.md for the metrics.

## Deploying

Since we are using a saved model with tensorflowjs, **we do not need a "big server"**, but just the web server of your choice (i.e. apache2) to serve the static files. This could also easily be distributed with a Content Delivery Network (CDN).

Just copy the model from [https://github.com/jordimassaguerpla/model.chestxray.ai/tree/main/saved_model.tfjs](https://github.com/jordimassaguerpla/model.chestxray.ai/tree/main/saved_model.tfjs) into the webserver, and then copy the files from this project.

## Legal

Except for the model, you can mostly do what you want (MIT license).
For the model, see https://github.com/jordimassaguerpla/model.chestxray.ai/blob/main/README.md#legal






