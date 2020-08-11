# Build, Deploy, and Monitor ML Models with Amazon SageMaker

**Note**: This branch contains updates to port the code to [v2 of the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/v2.html) which was [released on August 4th, 2020](https://pypi.org/project/sagemaker/#history).

This is the repository for my online course [Build, Deploy, and Monitor ML Models with Amazon SageMaker](https://mlinproduction.teachable.com/p/build-deploy-and-monitor-ml-models-with-amazon-sagemaker?utm_source=mlinproduction&utm_medium=github&utm_campaign=course_repo).

If you'd like to learn more about the course, [check out this webinar](https://twimlai.com/sagemaker?utm_source=ml_in_production_webinar&utm_medium=affiliate_course&utm_campaign=sagemaker_course_202008) I recorded with Sam Charrington of the TWiML podcast.

## Notebooks

* [Ch. 2 Setting Up SageMaker Studio](https://github.com/lpatruno/sagemaker-course/blob/master/notebooks/ch02_setup.ipynb) - In this chapter we set up the SageMaker Studio development environment and interact with Studio's visual interface. This notebook installs additional depenencies.
* [Ch. 3 Interactive Model Training](https://github.com/lpatruno/sagemaker-course/blob/master/notebooks/ch03_interactive_model_training.ipynb) - In this chapter we learn how to interactively train models in SageMaker using built-in algorithms and custom training code. In particular, we train customer churn prediction models using XGBoost and scikit-learn.
* [Ch. 4 Experiment Management](https://github.com/lpatruno/sagemaker-course/blob/master/notebooks/ch04_experiment_management.ipynb) - In this chapter we learn how to use SageMaker Experiments to organize the results of deep learning experiments. We train models using the Tensorflow and PyTorch frameworks, perform hyperparameter optimization, and find the best performing model.
* [Ch. 5 Model Deployment](https://github.com/lpatruno/sagemaker-course/blob/master/notebooks/ch05_model_deployment.ipynb) - In this chapter learn how to deploy trained models in SageMaker by deploying the models we trained in previous chapters. We demonstrate how to perform batch inference using SageMaker Batch Transform and how to perform online inference with hosted API endpoints. Finally, we configure autoscaling for hosted endpoints.
* [Ch 6. Model Monitoring](https://github.com/lpatruno/sagemaker-course/blob/master/notebooks/ch05_model_deployment.ipynb) - In this chapter we learn how to use Model Monitor to monitor deployed endpoints. This lets us detect data drift by capturing and storing incoming feature data and comparing the distributions of live data to training data.


## Additional Information

Many of the code samples in this repository have been adapted from the examples in the [Amazon SageMaker Examples](https://github.com/awslabs/amazon-sagemaker-examples/) repository. That repository is a fantastic source to learn more about SageMaker. 
