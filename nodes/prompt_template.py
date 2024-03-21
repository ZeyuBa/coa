class MyPrompts:
    def __init__(self):
        pass
    

    def data_description_template(self,data_samples,data_description):
        """
        You are an experienced data scientist, giving the following data samples and description, you are supposed to recognize the target column name.

        ### Data Samples ###
        {data_samples}

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

    def domain_knowledge_template(self,data_description,data_report,task_prompt):
        """
        You are a domain knowledge terminology interpreter,
        your role is to provide additional information and insights
        related to the task domain.
        
        Note: you should output in given format.

        Here are some relevant background knowledge about this problem:
        ### Data Description ###
        {data_description}

        Here are some pre-test data-report from other experts:
        ### Data Report ###
        {data_report}
        
        You can contribute by sharing your expertise, explaining relevant concepts,
        and offering suggestions to improve the task understanding, data preprocessing and model selection.
        Please provide your input based on the given problem description:

        ### Task Description ###
        {task_prompt}

        ### Response Format ###
        ```json
        {{
            "domain_knowledge": domain knowledge description,
        }}
        ```
        """
        pass

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

    def program_template(self,code_template):
        """
        You are a Python programmer in the field of XGBoost model construction and data preprocessing.
        Your proficiency in utilizing third-party libraries such as Pandas is essential. In addition, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy.
        You aim to develop an efficient Python program that complete the given task according to the code template. You need to follow docstrings carefully.
        Let's work this out in a step by step way to be sure we have the right code.

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
            "agent_generation_template": self.agent_generation_template,
            "model_plan_template": self.model_plan_template,
            "data_plan_template": self.data_plan_template,
            "program_template": self.program_template,
            "refine_template": self.refine_template,
            "data_description_template": self.data_description_template
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