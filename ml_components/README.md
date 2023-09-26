# What it is?

## AWS CLI

```shell
aws  --endpoint-url http://192.168.1.194:9000 s3 ls s3://config/train
```


## Design

### Workflow

```mermaid
flowchart TD;
    k8s-->|mode:str, parameters:str|main;
    local_console-->|parameter_path:str|main;
    main-->|train_parameters:str|train_pipeline; 
    main-->|prediction_parameters:str|inference_pipeline;
    main-->|accuracy_meas_parameters:str|accuracy_meas_pipeline;
```

### Main

This is an endpoint of whole program

- Tasks:
    - Load parameters and type of tasks as a form of string.
    - Convert parameters into an instance of parameters dataclass of each tasks.
    - Start up the tasks with giving the instance of parameters dataclass.


### InferencePipeline

This part implements a pipeline process of inference.

- Args:
    - An instance of parameters dataclass of inference pipeline.
- Tasks:
    - Initialization:
        - Initialize the instance with a given parameters.
        - Initialize the data transfer instance of cloud storage.
        - Load inference model from storage.
    - Run:
        - Run prediction with the loaded model.


### AccuracyMeasurementPipeline

This part implements a pipeline process of accuracy measurement of models trained with `TrainPipeline`.

There could be a series of models that are trained with `TrainPipeline`, but which is the best model is not yet fixed. 
This process evaluates each model using a test dataset and a series of models.

```mermaid
flowchart TD;
    subgraph __init__;
        start_init-->|parameters_str|load_parameters;
        load_parameters-->|parameters|load_basemodel;
        load_basemodel-->|parameters and basemodel|load_model_path_list;
        load_model_path_list-->|parameters|load_dataset;
    end;
    subgraph run;
        start-->models_path_list;
        models_path_list-->proc_model{end of models};
        proc_model-->load_model;
        load_model-->|no|dataset;
        proc_model-->|yes|calc_confision_matrix;
        dataset-->proc_dataset{end of dataset};
        proc_dataset-->|no|preict-->proc_dataset;
        proc_dataset-->|yes|models_path_list;
        calc_confision_matrix-->|confusion_matrix|finish;
    end;
```


