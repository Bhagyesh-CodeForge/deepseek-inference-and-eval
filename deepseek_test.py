'''
Assignment 1: Reproduce DeepSeek-R1 Outputs
 Goal: Gain hands-on familiarity with model behavior.
 • Task: Use a DeepSeek-R1 distilled model (e.g., 7B or 13B) via Hugging Face or locally.
 • Prompt it with various reasoning tasks (math, logic, QA).
 • Compare outputs with and without CoT prompting.
 • Write a mini report: “What happens when the model is guided to reason step-bystep?”
'''
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Use DeepSeek-7B-Chat model
#model_name = "deepseek-ai/deepseek-llm-7b-chat" #The model won't generate an immediate response if you running the code on a local device that has no powerful GPU
model_name = "deepseek-ai/deepseek-coder-1.3b-base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype="auto"  # uses float32
)

print("Setting up pipeline...")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer) #device=-1)


# Prompts
prompt_basic = "What is (23 * 12) + 45?"
prompt_cot = "Let's think step by step. First, multiply 23 by 12. That gives us 276. Then, add 45. The final result is?"

# Generate outputs
print("\n Without CoT:")
#print(pipe(prompt_basic, max_new_tokens=100)[0]['generated_text'])
output = pipe(
    prompt_basic,
    max_new_tokens=32,            # reduce from 100
    #do_sample=False,              # deterministic (no randomness)
    #temperature=0.7,              # optional
    #return_full_text=True
)[0]['generated_text']
print(output)

print("\n With CoT:")
#print(pipe(prompt_cot, max_new_tokens=100)[0]['generated_text'])
output = pipe(
    prompt_cot,
    max_new_tokens=32,            # reduce from 100
    #do_sample=False,              # deterministic (no randomness)
    #temperature=0.7,              # optional
    #return_full_text=True
)[0]['generated_text']
print(output)
'''
 Without CoT:
What is (23 * 12) + 45?

A: Answer is 110, because the order of operations is as follows:
(23 * 12) + 4

 With CoT:
model.safetensors:   1%|▍                                                           | 21.0M/2.69G [00:05<10:20, 4.30MB/s]
Let's think step by step. First, multiply 23 by 12. That gives us 276. Then, add 45. The final result is?

    ; 276 + 45 = 311
'''
logic_prompt = "If all cats are animals and some animals are mammals, are all cats mammals?"
logic_cot = "Let's think step by step. All cats are animals. Some animals are mammals. That does not guarantee that all cats are mammals."

qa_prompt = "Who was the first person to walk on the moon?"
qa_cot = "Let's think through this. The Apollo 11 mission landed in 1969. Neil Armstrong was the mission commander and first to step onto the surface."

print("\n Logic without CoT:")
print(pipe(logic_prompt, max_new_tokens=100)[0]['generated_text'])

print("\n Logic with CoT:")
print(pipe(logic_cot, max_new_tokens=100)[0]['generated_text'])

print("\n QA without CoT:")
print(pipe(qa_prompt, max_new_tokens=100)[0]['generated_text'])

print("\n QA with CoT:")
print(pipe(qa_cot, max_new_tokens=100)[0]['generated_text'])
'''
 Logic without CoT:
If all cats are animals and some animals are mammals, are all cats mammals?

What does this question mean?

* a cat is an animal
* an animal is a mammal
* a mammal is a cat

What does this question mean?

* a cat is an animal
* an animal is a mammal
* a mammal is a cat

What does this question mean?

**a cat is an animal**

what is the best answer?

* a cat is an

 Logic with CoT:
Let's think step by step. All cats are animals. Some animals are mammals. That does not guarantee that all cats are mammals. So, what about the mammals? Some mammals are reptiles. That does not guarantee that all mammals are reptiles. So, what 
about the reptiles? All reptiles are insects but not all insects are reptiles. So, what about the insects? Some insects are birds but not all birds are insects. So, what about the birds? Some birds are mammals but not all mammals are birds     

 QA without CoT:
Who was the first person to walk on the moon?

## Solution

```js
const isFirstPersonToWalkOnMoon = (person) => person.gender === "M";
```

**How to solve?**

We start by creating a function that returns the first person to walk on the moon as a property called `isFirstPersonToWalkOnMoon`.

```js
const isFirstPersonToWalkOnMoon = (person) => person.gender === "

 QA with CoT:
Let's think through this. The Apollo 11 mission landed in 1969. Neil Armstrong was the mission commander and first to step onto the surface. He was the primary navigator for the mission. The first person he knew was Rudolph the Red-Nosed Reindeer.

```
Rudolph was the first person he knew. He was a small, yellow-eyed, black-and-white reindeer. He was a red-striped, white-and-black reindeer. He was a reindeer. He was a reindeer. He was a Red
'''

