from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory

# Initialize the OpenAI model with GPT-3.5
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Initialize memory for state sharing
memory = ConversationBufferMemory()

# Define the prompt for requirements gathering
requirements_gather_prompt = ChatPromptTemplate.from_template(
    "You are a project manager. Please provide a detailed user story for a feature to be implemented in a Flask web application."
)

# Define the prompt for code generation
app_code_gen_prompt = ChatPromptTemplate.from_template(
    "You are a skilled Python developer. Based on the following requirements, generate a complete Python Flask application:\n\nRequirements:\n{requirements}\n\nGenerated Code:"
)

# Define the prompt for test code generation
test_code_gen_prompt = ChatPromptTemplate.from_template(
    "You are a Python tester. Here is a Flask application code:\n{generated_code}\n\nWrite a test case for this application using pytest. Provide the test result and any errors encountered:"
)

# Create the chains for the agents
requirements_gather_chain = LLMChain(
    llm=llm,
    prompt=requirements_gather_prompt,
    memory=memory,
    output_key="user_story",
    output_parser=StrOutputParser()
)

app_code_gen_chain = LLMChain(
    llm=llm,
    prompt=app_code_gen_prompt,
    memory=memory,
    output_key="generated_code",
    output_parser=StrOutputParser()
)

test_code_gen_chain = LLMChain(
    llm=llm,
    prompt=test_code_gen_prompt,
    memory=memory,
    output_parser=StrOutputParser()
)

# Functions to run the agents
def gather_requirements():
    return requirements_gather_chain.run()

def generate_app_code(requirements):
    return app_code_gen_chain.run({"requirements": requirements})

def generate_test_code(generated_code):
    return test_code_gen_chain.run({"generated_code": generated_code})

def save_code_to_file(filename, code):
    with open(filename, 'w') as file:
        file.write(code)

# Define the AppDevelopmentGraph class without inheriting from TypedDict
class AppDevelopmentGraph:
    def __init__(self, requirements_list):
        self.state_schema = {
            "GatherRequirements": {},
            "GenerateAppCode": {},
            "GenerateTestCode": {}
        }
        self.config_schema = {}
        self.input_schema = {}
        self.output_schema = {}

        if not isinstance(requirements_list, list):
            raise ValueError("requirements_list must be a list.")

        self.requirements_list = requirements_list  # Queue of user stories
        self.story_count = 0
        self.testing_rounds = 0  # Track the number of testing rounds
        self.state_data = {}

    def execute(self):
        # Start the StateGraph execution
        if self.requirements_list:
            self.gather_requirements()
        else:
            print("No requirements to process.")
            self.end()

    def gather_requirements(self):
        if self.requirements_list:
            self.story_count += 1
            print(f"\nProcessing Story {self.story_count}...\n")

            # Get the next requirement (user story)
            user_story = self.requirements_list.pop(0)
            print("User Story:\n", user_story)

            # Store the user story in the state memory
            self.state_data["user_story"] = user_story

            # Generate app code based on the user story
            self.generate_app_code()
        else:
            self.end()

    def generate_app_code(self):
        user_story = self.state_data.get("user_story")
        if user_story:
            generated_code = generate_app_code(user_story)
            save_code_to_file(f"app_code_{self.story_count}.py", generated_code)
            print("Application code generated and saved.")

            # Generate test code for the application
            self.generate_test_code(generated_code)
        else:
            print("No user story found to generate code.")

    def generate_test_code(self, generated_code):
        test_code = generate_test_code(generated_code)
        save_code_to_file(f"test_code_{self.story_count}.py", test_code)
        print("Test code generated and saved.")

        self.testing_rounds += 1
        # Proceed to the next state or end if all requirements are processed
        if self.requirements_list:
            self.gather_requirements()
        else:
            self.end()

    def end(self):
        print("Processing complete.")

# Example usage
requirements_list = [
    "Create a Flask web application with one endpoint `/hello` that returns 'Hello, World!'.",
    "Add an endpoint `/goodbye` that returns 'Goodbye, World!'.",
    "Implement error handling for undefined routes."
]

app_development_graph = AppDevelopmentGraph(requirements_list)
app_development_graph.execute()
