'''
 Assignment 2: CoT Benchmark Replication
 Goal: Understand evaluation techniques.
 • Task: Evaluate DeepSeek-R1 on a benchmark like GSM8K or MATH.
 • Tools: lm-eval-harness or custom Python scripts.
 • Deliverable: Charts showing accuracy with/without CoT.
'''
from datasets import load_dataset

# Load a subset of GSM8K for quick evaluation
gsm8k = load_dataset("gsm8k", "main", split="test[:10]")  # use 10 for testing; full set has 1319

for i in range(2):
    print(f"Q{i+1}: {gsm8k[i]['question']}")
    print(f"A{i+1}: {gsm8k[i]['answer']}\n")

def generate_prompt(q: str, mode: str = "plain") -> str:
    if mode == "plain":
        return f"Question: {q}\nAnswer:"
    elif mode == "cot":
        return f"Let's think step by step.\nQuestion: {q}\nAnswer:"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#model_name = "deepseek-ai/deepseek-llm-7b-chat"
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

def get_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

from datasets import load_dataset

gsm8k = load_dataset("gsm8k", "main", split="test[:10]")

results = []

for ex in gsm8k:
    q = ex["question"]
    a_gold = ex["answer"]

    prompt_plain = generate_prompt(q, "plain")
    prompt_cot = generate_prompt(q, "cot")

    a_plain = get_answer(prompt_plain)
    a_cot = get_answer(prompt_cot)

    results.append({
        "question": q,
        "gold": a_gold,
        "plain_answer": a_plain,
        "cot_answer": a_cot
    })

# Print example results
for r in results:
    print("\n Question:", r["question"])
    print(" Gold Answer:", r["gold"])
    print(" Plain Output:", r["plain_answer"])
    print(" CoT Output:", r["cot_answer"])

def normalize(text):
    return ''.join([c for c in text if c.isdigit() or c == '.'])

plain_correct = 0
cot_correct = 0

for r in results:
    gold = normalize(r["gold"].split("####")[-1].strip())
    plain = normalize(r["plain_answer"])
    cot = normalize(r["cot_answer"])

    if gold in plain:
        plain_correct += 1
    if gold in cot:
        cot_correct += 1

print(f"\n Accuracy out of {len(results)} samples:")
print(f"Without CoT: {plain_correct}/{len(results)}")
print(f"With CoT:    {cot_correct}/{len(results)}")

# the graph to plot accuracy of with and without CoT prompts
import matplotlib.pyplot as plt

# Accuracy values (from your real results)
plain_pct = 20.0
cot_pct = 30.0

# Plot
labels = ['Without CoT', 'With CoT']
scores = [plain_pct, cot_pct]
colors = ['tomato', 'mediumseagreen']

plt.figure(figsize=(6, 5))
bars = plt.bar(labels, scores, color=colors)
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('DeepSeek-1.3B Accuracy on GSM8K (10 samples)')

# Add accuracy text above bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
#plt.savefig("gsm8k_cot_accuracy_1.3b.png") #run the line if you want to save the graph output in image format
plt.show()

'''
 CoT Output: Let's think step by step.
Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
Answer: Josh buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  He made $150,000 in profit.

 Question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
 Gold Answer: He sprints 3*3=<<3*3=9>>9 times
So he runs 9*60=<<9*60=540>>540 meters
#### 540
 Plain Output: Question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
Answer: 240

Question: James decides to run 3 sprints 3 times a week.  He runs 60 meters
 CoT Output: Let's think step by step.
Question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
Answer: 210 meters

Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups 
of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?      
 Gold Answer: If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.
If she feeds the flock 15 cups of feed in the morning, and 25 cups in the afternoon, then the final meal would require 60-15-25=<<60-15-25=20>>20 cups of chicken feed.
#### 20
 Plain Output: Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  
How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
Answer: 100

 CoT Output: Let's think step by step.
Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives 
her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?       
Answer:


*

*The first meal of the day is 15 cups of feed.

*The second meal of the day is 25 cups of feed.

*The third meal of the day is 15 cups of feed.

*The fourth meal of the day is 25 cups of feed.

*The fifth meal of the day is 15 cups of feed.

*The sixth meal of the day is 25 cups of feed.

*The seventh meal of the day is 15 cups of feed.
'''

#these are some examples generated as output, after execution there is a series of output to be noted 