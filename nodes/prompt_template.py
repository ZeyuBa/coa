class MyPrompts:
    def __init__(self):
        pass
    

    def data_description_template(self,data_samples,data_description):
        """
        You are an experienced data scientist, giving the following data description, you are supposed to recognize the target column name.
        
        ### Data Description ###
        {data_description}

        ### Response Format ###
        ```json
        {{
            target_column: target column name,
        }}
        ```
        """
        
        pass
    def task_type_template(self,query,data_description):
        """
        You are an experienced machine learning engineer, giving the following user requirements and data description, you are supposed to do the following tasks step-by-step:
        Recognize the machine learning task type (classification, regression, clustering, etc.) and direct output the answer in json format.

        ### User Requirement ###
        {query}

        ### Data Description ###
        {data_description}

        ### Response Format ###
        ```json
        {{
            task_type: machine learning task type
        }}
        ```
        """
        pass

    def task_specify_template(self,query,data_description):
        """
        You are an experienced machine learning engineer, giving the following original user requirements and data description,
        you are supposed to extend it to a machine learning task description so that the requirements is more specific.

        ### User Requirement ###
        {query}

        ### Data Description###
        {data_description}

        ### Response Format ###
        ```json
        {{

            "specific_task": specific task description

        }}
        ```
        """
        pass

    def domain_knowledge_template(self,data_description,data_report):
        """
        You are a domain knowledge terminology interpreter,
        your role is to provide additional information and insights
        related to the task domain.
        
        Note: you should output in given format.

        Here are some relevant description about this data:
        ### Data Description ###
        {data_description}

        Here are a data-report from other experts:
        ### Data Report ###
        {data_report}
        
        You can contribute by 
        1. Identifying the significance of each attribute within the dataset, 
        2. Statistical information of Missing Values.
        3. Duplicate information.
        4. Give advice for features that is not related to training like 'id'.
        according to the given Report.

        ### Example ###
        The attribute information is:
        id: unique identifier
        gender: "Male", "Female" or "Other"
        age: age of the patient
        hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
        heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
        ever_married: "No" or "Yes"
        work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
        Residence_type: "Rural" or "Urban"
        avg_glucose_level: average glucose level in blood
        bmi: body mass index
        smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
        stroke: 1 if the patient had a stroke or 0 if not
        
        Statistical information of Missing Values:
        There are no null values in train

        Duplicates in train: 235, Duplicates in test: 121

        Advice:
        Remove undesired features "id". 
        
        ### Response Format ###
        ```json
        {{
            "feature_meaning": explain the meaning of each feature,
            "null_info": num of null,
            "duplicate_info": num of duplicates,
            "advice": advice for unrelated feature columns,

        }}
        ```
        """
        pass
    
    def eda_template(self,domain_knowledge,feature_info,tools):
        """
        You are a data scientist expert in exploretary data analysis, your responsibility is to make a exploration plan to 
        create new featrues according to the advice from given analysis. 
        You are supposed to do a fully analysis from given information and provide a subset list of sparse distributed THREE features that are most suitable to create new feartures. 
        
        ### Data Analysis ###
        {domain_knowledge}

        {feature_info}

        You can use these tools to create features.
        ### Tools ###
        {tools}

        ### Response format ###
        ```json
        {{
        
            "feat_list": 
            [
                only output the features' names you are interested in. 
            ]

        }}
        ```
        """
        pass
    def eda_template2(self,domain_knowledge,feature_info,tools,feature_name):
        """
        You are a data scientist expert in exploretary data analysis, your responsibility is to make a exploration plan to 
        create new featrues according to the advice from given analysis. 
        You are supposed to do a fully analysis from given information and provide a list of EDA option plan to the given feature in the following experiments.
        
        ### Data Analysis ###
        {domain_knowledge}

        {feature_info}

        You can use these tools to create features.
        ### Tools ###
        {tools}

        ### Given Feature ###
        {feature_name}

        ### Response Format ###
        ```json
        {{
            func_name:, 
            params_dict:
                {{
                    "data":"train",
                    "feature_name": name of the feature
                    "threshold": (You MUST decide upon above skewness, kurtosis, and distribution information of the feature, output A list of THREE Most interested threshold values of the feature that you want to explore in following experiments), 
                    "method": a list of methods of the feature that you want to explore in following experiments,
                }}
        }}
        ```
        """

    def search_space_template(self,operation_list,tools):
        """
        As an expert in Bayesian optimization and machine learning, your task is to construct a DesignSpace in HEBO (Hybrid Evolutionary Bayesian Optimization) style based on a given operation and their possible inputs. 
        Each operation associated with a specific function. 
        The objective is to provide a comprehensive list of all possible combinations of the operation for use in Bayesian optimization.

        You can contribute by :
        1. you should ONLY process those param in list of given operation to a hyper-parameter dict in HEBO style 
        2. the 'name' is define as featureName_paramName and 'type' always be 'cat'.

                
        ### Given Operation ###
        {operation_list}

        ### HEBO Style Example ###
        {{'name': 'optimizer', 'type': 'cat', 'categories': ['Adam', 'SGD', 'RMSprop']}},


        ### Response Format ###
        ```json
        {{
            "hyperparam_dict": {{
            "name": ,
            "type":"cat",
            "categories": ,
            }},
        }}
        ```
        """
    def bo_code_template(self,space_list,tools,use_example):
        """
        ### Space List ###
        input space_list of AutoMLSpace class:
        {space_list}

        ### Instance Example ###
        the Instance from AutoMLSpace will be used as:
        {use_example}

        ### Func Info ###
        the decoded sample will be applied to the following functions:
        func_name: {func_name}
        func_description:
        {func_desc}
        
        """
    def agent_generation_template(self,task_prompt):
        """
        You are a Project Manager, your responsibility is to compile a team of engineers best suited for our machine learning project, based on the outlined task description.
        Your selection should include:
        ModelPlanProvider: Tasked with devising a detailed plan for constructing machine learning models.
        DataPlanProvider: Responsible for developing a comprehensive plan for data preprocessing prior to training.
        Programmer: Charged with the development of code for training and evaluating machine learning tasks.

        Note: Include a Visualizer in the team only if the project explicitly requires specialized data visualization.
        ### Task Description ###
        {task_prompt}
        ### Response Format ###
        ```json
        {{
            "agent_list": A Python list of engineers,
        }}
        ```
        """
        pass

    def model_plan_template(self,task_prompt):
        """
        You are a machine learning engineer, you are expert in model construction. According to the following task, you are supposed to
        construct a XGBoost model for classification or regression, fill in the hyperparameters and respond in json format.

        ### Task Description ###
        {task_prompt}
        ### Response Format ###
        ```json
        {{
            "model_plan": {{
                "type": classification or regression
                "params": {{
                    "booster": ,
                    "objective": ,
                    "gamma": ,
                    "max_depth": ,
                    "reg_lambda": ,
                    "subsample": ,
                    "colsample_bytree": ,
                    "min_child_weight": ,
                    "slient":,
                    "learning_rate": ,
                    "seed": ,
                    "num_class": if classification task
                }}
            }}
        }}
        ```
        """
        pass

    def data_plan_template(self,operation,task_prompt,data_description,domain_knowledge,feedback,):
        """
        You are an experienced machine learning engineer, you will be responsible for creating and managing a data preprocessing pipeline for raw datasets.
        Your key tasks include selecting and organizing various preprocessing operations.

        ### Operation Decscription and Options ###
        {operation}

        ### Decision-Making Criteria ###
        You will need to decide the selection of the preprocessing operation based on:

        Insights from domain experts knowledge
        Task requirements
        Data description
        Feedback on model performance

        ### Task Description ###
        {task_prompt}

        ### Data Description ###
        {data_description}

        ### Domain Kowledge ###
        {domain_knowledge}

        ### FeedBack ###
        {feedback}

        # Your Responsibilities:

        1. Provide explanations for why an operation was chosen or not chosen.
        2. According to the given reason, set selected to 0 or 1.
        3. Fill in a random option for the selected operation.

        ### Response Format ###
        ```json
        {{

            "operation_name":{{
                        "reason_for_selection":,
                        "selected": 0 or 1,
                        "option":,
                        }},

        }}
        ```
        """
        pass

    def program_template(self,code_task,code_template):
        """
        You are a Python programmer in the field of machine learning and data science.
        Your proficiency in utilizing third-party libraries such as Pandas is essential. In addition, it would be great if you could also provide some background in related libraries or tools, like NumPy, Scikit-learn.
        You aim to develop an efficient Python program that complete the given task according to the code template. You need to follow docstrings carefully.
        Let's work this out in a step by step way to be sure we have the right code.
        ### Task Description ###
        {code_task}

        ### Code Template ####
        {code_template}

        """
        pass
    def code_temp_template(self):
        """
        import neccessary libraries

        write docstrings

        
        """
        pass

    def params_transfer_template(self):

        """
        # Example:

        
        
        """

        pass
    def refine_template(self,detail,code_block,error_message):
        """
        You are a Python programmer in the field of machine learning. There is an error in the code bellow.
        To find the error, go through semantically complete blocks of the code, and check if everything looks good.
        Then rewrite the code according to template.  You need to follow docstrings carefully.
        Let's work this out in a step by step way to be sure we have the right code.

        ### Code Template ####
        {code_template}

        ### Wrong Code ####
        {code_block}

        ### Error Message ####
        {error_message}

        """
        pass

    def get_prompt(self, template_name):
        """
        Get the docstring for a specific prompt template.

        Args:
            template_name (str): The name of the prompt template.

        Returns:
            str: The docstring for the prompt template.
        """
        template_methods = {
            "task_type_template": self.task_type_template,
            "task_specify_template": self.task_specify_template,
            "domain_knowledge_template": self.domain_knowledge_template,
            "eda_template":self.eda_template,
            "agent_generation_template": self.agent_generation_template,
            "model_plan_template": self.model_plan_template,
            "data_plan_template": self.data_plan_template,
            "program_template": self.program_template,
            "refine_template": self.refine_template,
            "data_description_template": self.data_description_template,
            "search_space_template":self.search_space_template,
            "bo_code_template":self.bo_code_template,
            "eda_template2":self.eda_template2
        }

        if template_name not in template_methods:
            raise ValueError(f"Invalid template name: {template_name}")

        return template_methods[template_name].__doc__


def operation():
    '''
    1. 
    Operation Name: Dimensionality Reduction
    Explanation: Simplify the dataset by reducing the number of features.
    Options: Choose between linear method PCA (Principal Component Analysis) and nonlinear method ISOMAP (Isometric Mapping).
    ---
    2. 
    Operation Name: Resampling
    Explanation: Balance the data by increasing the representation of the minority class or decreasing the majority class.
    Options: Choose between upsampling or downsampling.
    ---
    3. 
    Operation Name: Feature Transformation
    Explanation: Modify data features for optimal processing.
    Options: Choose from normalization, standardization.
    ---   
    4. 
    Operation Name: Feature Selection
    Explanation: Identify and select the most relevant features for the specific tasks.
    Options: Use your expertise to choose the appropriate features.
    '''
    pass
