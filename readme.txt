this is an end to end project of machine learning where we will be go throgh phases of project
topic: 

phases:
1. setup github repositry
    1a. eniornment setup (python 3.13)
    1b. setup.py
    1c. requirements.txt

2. create folder structure
    2a. create folders
        
        src/components
            data_ingestion.py (codes related to read data)
            data_transformation.py (transform data)
            model_trainer.py (trains model)
        
        src/pipeline
            training.py (training pipeline)
            prediction.py(prediction pipeline)
        
        src/logger.py (for logging)

        src/exception.py (for exception handling)

        src/utils (for utility management)


3. coding part 1 - 
    src/exception.py
        setting custom exception using sys

    src/logging.py
        setting logging for all file


4. performing to understand data
    4a. performing EDA to understand data
    4b. creating the model training in notebook


5. creating a moduler code version
    5a. dataingestion.py creating a dataingestion pipeline