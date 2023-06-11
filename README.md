# OVERALL

The src repository contains all functionality for creating synthetic data and training models. 
To train models and generate synthetic data, access to the "dikuAngiograms" repository on ERDA is required.


# GENERATING SYNTHETIC DATA

First in the command prompt navigate to the src.parameter_estimation folder.

## Create backgrounds. 
	* Open make_background.py. 
	* Optionally, change save_path such that backgrounds are saved locally.
	* Then run "python make_background.py" in command prompt

## Create bias fields. 
	* open make_bias_fields.py. 
	* Optionally, change the relavant paths such bias fields are saved where the user wishes)
	* Then run "python make_bias_fields.py" in command prompt

## Create synthetic data. 
	* open scripts.gen_syn_train_val_test.py. 
	* Change num_samples to number of desired synthetic samples. 
	* If paths of bias fields and backgrounds were changed, 
	  change the paths of bias field dataset "bfDataSet" and
	  background dataset "BG" such that the paths correspond to where data is saved.
	* Then run "python scripts.gen_syn_train_val_test.py"


# TRAINING MODELS

* Models are trained in the "Training.ipynb" notebook. 
* The notebook assumes the existance of folders for saving model weights, training loss and test metrics. 
To train models change create relavant folders and change relavant data paths. 
