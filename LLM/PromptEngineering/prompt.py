

class Prompt:
    def __init__(self):
        pass

    def zero_shot_prompt(self):
        print("Please Provide Your Zero Shot Instruction")
        input_prompt = input("Enter your prompt: ")
        return input_prompt

    def validate_sample_count(self, number_of_samples: int):
        assert number_of_samples > 0, "Number of samples must be greater than 0"

    def few_shot_prompt(self, number_of_examples: int):
        print("Please Provide Your Few Shot Instruction")
        input_prompt = input("Enter your prompt: ")
        examples = []
        
        self.validate_sample_count(number_of_examples)

        for i in range(number_of_examples):
            print(f"Please Provide Example {i + 1}")
            example = input("Enter your example: ")
            examples.append(example)
        return input_prompt, examples

    def run_prompt_engineering(self, prompt_type: str, number_of_examples: int = 0):
        if prompt_type == "zero_shot":
            return self.zero_shot_prompt()
        elif prompt_type == "few_shot":
            return self.few_shot_prompt(number_of_examples)
        else:
            raise ValueError("Invalid prompt type. Please choose 'zero_shot' or 'few_shot'.")

