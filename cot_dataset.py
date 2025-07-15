'''
Assignment 3: Build a CoT Dataset
 Goal: Learn dataset formatting and preprocessing.
 • Task: Collect and clean 500–1000 QA samples from open sources (math word problems, trivia, logic puzzles).
 • Format them in JSON or CSV with explicit CoT reasoning steps.
 • Deliverable: A high-quality, ready-to-train dataset.
'''
import json

# Initial dataset structure
dataset = []

dataset = [
    # MATH
    {
        "question": "A farmer has 20 cows and buys 15 more. He sells 5. How many cows does he have now?",
        "cot": "Let's think step by step. He had 20, bought 15 → 20 + 15 = 35. Then sold 5 → 35 - 5 = 30.",
        "answer": "30",
        "domain": "math",
    },
    {
        "question": "Tom has 5 boxes of pencils. Each box has 12 pencils. How many pencils does Tom have?",
        "cot": "Each box has 12 pencils. 5 boxes × 12 = 60 pencils.",
        "answer": "60",
        "domain": "math",
    },
    {
        "question": "Jenny buys 3 shirts for $15 each and 2 pairs of pants for $25 each. How much does she spend in total?",
        "cot": "3 shirts × $15 = $45. 2 pants × $25 = $50. Total: $45 + $50 = $95.",
        "answer": "95",
        "domain": "math",
    },
    {
        "question": "A rectangle is 8 meters long and 3 meters wide. What is its area?",
        "cot": "Area of rectangle = length × width. So, 8 × 3 = 24 m².",
        "answer": "24",
        "domain": "math",
    },
    {
        "question": "There are 12 cookies. If each child gets 3 cookies, how many children can share them?",
        "cot": "12 ÷ 3 = 4 children.",
        "answer": "4",
        "domain": "math",
    },
    {
        "question": "Lily reads 30 pages per day. How many pages does she read in a week?",
        "cot": "30 pages/day × 7 days = 210 pages.",
        "answer": "210",
        "domain": "math",
    },
    {
        "question": "If 1 book costs $8, how many books can you buy with $40?",
        "cot": "40 ÷ 8 = 5 books.",
        "answer": "5",
        "domain": "math",
    },
    {
        "question": "A pizza has 8 slices. If 3 friends share 2 pizzas equally, how many slices does each get?",
        "cot": "2 pizzas = 16 slices. 16 ÷ 3 = 5.33 slices per friend.",
        "answer": "5.33",
        "domain": "math",
    },
    {
        "question": "There are 5 bags with 6 marbles each. How many marbles are there in total?",
        "cot": "5 × 6 = 30 marbles.",
        "answer": "30",
        "domain": "math",
    },
    {
        "question": "A triangle has sides of 5 cm, 6 cm, and 7 cm. What is its perimeter?",
        "cot": "5 + 6 + 7 = 18 cm.",
        "answer": "18",
        "domain": "math",
    },
    {
        "question": "Each table has 4 legs. How many legs do 9 tables have?",
        "cot": "9 × 4 = 36 legs.",
        "answer": "36",
        "domain": "math",
    },
    {
        "question": "A box weighs 12 kg. Another weighs 3 kg more. What’s the weight of the second box?",
        "cot": "12 + 3 = 15 kg.",
        "answer": "15",
        "domain": "math",
    },
    {
        "question": "A phone costs $600. It’s on sale with a 25% discount. What’s the sale price?",
        "cot": "25% of $600 = $150. Sale price = $600 - $150 = $450.",
        "answer": "450",
        "domain": "math",
    },
    {
        "question": "An apple costs $0.80. How much do 7 apples cost?",
        "cot": "0.80 × 7 = $5.60.",
        "answer": "5.60",
        "domain": "math",
    },
    {
        "question": "You run 3 km per day. How far do you run in 10 days?",
        "cot": "3 × 10 = 30 km.",
        "answer": "30",
        "domain": "math",
    },
    {
        "question": "Jack has $50. He buys a toy for $18. How much money is left?",
        "cot": "50 - 18 = 32.",
        "answer": "32",
        "domain": "math",
    },
    {
        "question": "If there are 24 hours in a day, how many hours in 5 days?",
        "cot": "24 × 5 = 120 hours.",
        "answer": "120",
        "domain": "math",
    },
    {
        "question": "A baker makes 40 loaves of bread in 5 hours. How many loaves per hour?",
        "cot": "40 ÷ 5 = 8 loaves/hour.",
        "answer": "8",
        "domain": "math",
    },
    {
        "question": "You save $10 per week. How much after 6 weeks?",
        "cot": "$10 × 6 = $60.",
        "answer": "60",
        "domain": "math",
    },
    {
        "question": "If 2 pencils cost $3, how much do 10 pencils cost?",
        "cot": "$3 ÷ 2 = $1.50/pencil. 10 × 1.5 = $15.",
        "answer": "15",
        "domain": "math",
    },
    {
        "question": "What is the average of 6, 12, 18?",
        "cot": "(6 + 12 + 18) = 36; 36 ÷ 3 = 12.",
        "answer": "12",
        "domain": "math",
    },
    {
        "question": "Sarah has 3 notebooks, each with 120 pages. She uses 30 pages from each. How many pages remain unused?",
        "cot": "Each notebook: 120 - 30 = 90 pages unused. 3 notebooks: 3 × 90 = 270 pages remain.",
        "answer": "270",
        "domain": "math",
    },
    # LOGIC
    {
        "question": "All mammals are animals. All dogs are mammals. Are all dogs animals?",
        "cot": "All dogs are mammals and all mammals are animals → by transitivity, all dogs are animals.",
        "answer": "Yes",
        "domain": "logic",
    },
    {
        "question": "All cats are animals. Some animals are dogs. Are all cats dogs?",
        "cot": "Not all animals are dogs. So, being a cat doesn't imply being a dog.",
        "answer": "No",
        "domain": "logic",
    },
    {
        "question": "If no pencils are pens and some pens are markers, can pencils be markers?",
        "cot": "Pencils are not pens, and some pens are markers. No connection exists between pencils and markers.",
        "answer": "Not necessarily",
        "domain": "logic",
    },
    {
        "question": "If A > B and B > C, then is A > C?",
        "cot": "Yes, transitive property: A > B > C implies A > C.",
        "answer": "Yes",
        "domain": "logic",
    },
    {
        "question": "If the moon is made of cheese, and cheese is dairy, is the moon dairy?",
        "cot": "If both statements are true, then yes, the moon would be dairy.",
        "answer": "Yes",
        "domain": "logic",
    },
    {
        "question": "If all red flowers are roses and this flower is not red, is it a rose?",
        "cot": "The flower not being red doesn’t prove it’s not a rose; could be a rose of different color.",
        "answer": "Not necessarily",
        "domain": "logic",
    },
    {
        "question": "If today is Monday, what day will it be in 4 days?",
        "cot": "Monday + 4 = Friday.",
        "answer": "Friday",
        "domain": "logic",
    },
    {
        "question": "All squares are rectangles. Are all rectangles squares?",
        "cot": "No. All squares are rectangles, but not all rectangles are squares.",
        "answer": "No",
        "domain": "logic",
    },
    {
        "question": "If it rains, the grass grows. It rained. Did the grass grow?",
        "cot": "Yes, according to the conditional statement.",
        "answer": "Yes",
        "domain": "logic",
    },
    {
        "question": "If Bob is taller than Alice and Alice is taller than Tom, who is tallest?",
        "cot": "Bob > Alice > Tom. So Bob is the tallest.",
        "answer": "Bob",
        "domain": "logic",
    },
    {
        "question": "If you have to be at least 18 to vote, and Emma is 20, can she vote?",
        "cot": "20 ≥ 18, so yes, she can vote.",
        "answer": "Yes",
        "domain": "logic",
    },
    {
        "question": "A bus leaves every 15 minutes. If one left at 2:45, when is the next one?",
        "cot": "2:45 + 15 mins = 3:00.",
        "answer": "3:00",
        "domain": "logic",
    },
    {
        "question": "If Peter is not younger than 20 and not older than 25, what is his age range?",
        "cot": "Peter is between 20 and 25 inclusive.",
        "answer": "20–25",
        "domain": "logic",
    },
    {
        "question": "If pencils are not edible and pens are pencils, can pens be eaten?",
        "cot": "Pens are pencils. Pencils are not edible → pens are not edible.",
        "answer": "No",
        "domain": "logic",
    },
    {
        "question": "If no birds can swim and penguins are birds, can penguins swim?",
        "cot": "If we accept no birds can swim, then penguins can’t swim. But this contradicts real-world knowledge.",
        "answer": "No (under assumption)",
        "domain": "logic",
    },
    {
        "question": "If 3 > 2 and 2 > 1, is 3 > 1?",
        "cot": "Yes. Transitive relation applies: 3 > 2 > 1 ⇒ 3 > 1.",
        "answer": "Yes",
        "domain": "logic",
    },
    {
        "question": "If Sam is older than John, and John is older than Lily, who is the youngest?",
        "cot": "Sam > John > Lily → Lily is the youngest.",
        "answer": "Lily",
        "domain": "logic",
    },
    {
        "question": "If all apples are fruits and all fruits are tasty, are apples tasty?",
        "cot": "Yes. Apples are fruits, and all fruits are tasty → apples are tasty.",
        "answer": "Yes",
        "domain": "logic",
    },
    {
        "question": "If trains run every hour and one left at noon, when is the next train?",
        "cot": "Next train: 1:00 PM.",
        "answer": "1:00 PM",
        "domain": "logic",
    },
    {
        "question": "If Jane arrives before Tom and Tom arrives before Mark, who arrives last?",
        "cot": "Jane < Tom < Mark. So Mark arrives last.",
        "answer": "Mark",
        "domain": "logic",
    },
    {
        "question": "If cars need fuel and mine is empty, what should I do?",
        "cot": "Fuel is required. Car is empty. Refill the tank.",
        "answer": "Refuel",
        "domain": "logic",
    },
    {
        "question": "Some pens are blue. All blue things are useful. Can we conclude all pens are useful?",
        "cot": "Only some pens are blue. Only blue things are useful. We cannot conclude all pens are useful.",
        "answer": "No",
        "domain": "logic",
    },
    # COMMONSENSE
    {
        "question": "You left a chocolate bar in the sun. It turned soft and sticky. What likely happened?",
        "cot": "Chocolate melts in heat. Being in the sun caused it to absorb heat and melt.",
        "answer": "It melted.",
        "domain": "commonsense",
    },
    {
        "question": "You drop a glass on the floor. What happens?",
        "cot": "A glass is fragile. Dropping it on a hard surface typically causes it to break.",
        "answer": "It breaks",
        "domain": "commonsense",
    },
    {
        "question": "You feel wet after being outside without an umbrella. What likely happened?",
        "cot": "If you're wet and outside without cover, it likely rained.",
        "answer": "It rained",
        "domain": "commonsense",
    },
    {
        "question": "A car won’t start and the fuel tank is empty. What's the issue?",
        "cot": "Cars need fuel to start. An empty tank prevents ignition.",
        "answer": "No fuel",
        "domain": "commonsense",
    },
    {
        "question": "You hear thunder and see lightning. What kind of weather is it?",
        "cot": "Thunder and lightning occur during a storm.",
        "answer": "Stormy",
        "domain": "commonsense",
    },
    {
        "question": "If your clothes are wet and hung in the sun, what happens?",
        "cot": "The sun evaporates moisture. Clothes will dry.",
        "answer": "They dry",
        "domain": "commonsense",
    },
    {
        "question": "You smell smoke in the kitchen. What might be happening?",
        "cot": "Smoke indicates something may be burning.",
        "answer": "Food might be burning",
        "domain": "commonsense",
    },
    {
        "question": "You wake up, it’s light outside, and your clock says 9:00. What time is it?",
        "cot": "Morning light + 9:00 = 9 AM.",
        "answer": "9 AM",
        "domain": "commonsense",
    },
    {
        "question": "You wear a coat and still feel cold. What does that suggest?",
        "cot": "The temperature is likely very low.",
        "answer": "It's very cold",
        "domain": "commonsense",
    },
    {
        "question": "You plug in a lamp but it doesn’t turn on. What's a possible reason?",
        "cot": "Possible issues: bulb burnt out, switch off, or power off.",
        "answer": "The bulb might be dead",
        "domain": "commonsense",
    },
    {
        "question": "You leave food out overnight and it smells bad. What happened?",
        "cot": "Left out food can spoil due to bacteria.",
        "answer": "It spoiled",
        "domain": "commonsense",
    },
    {
        "question": "You hear a baby crying. What might it need?",
        "cot": "Crying babies may be hungry, tired, or need changing.",
        "answer": "Food or attention",
        "domain": "commonsense",
    },
    {
        "question": "You drink coffee at night. What might happen?",
        "cot": "Caffeine is a stimulant. You may not sleep easily.",
        "answer": "Difficulty sleeping",
        "domain": "commonsense",
    },
    {
        "question": "You're stuck in traffic and late for a meeting. What should you do?",
        "cot": "Let others know. Try alternative routes.",
        "answer": "Inform the meeting organizer",
        "domain": "commonsense",
    },
    {
        "question": "If you feel thirsty, what should you do?",
        "cot": "Thirst signals a need for hydration.",
        "answer": "Drink water",
        "domain": "commonsense",
    },
    {
        "question": "You're hungry but only candy is available. What’s the better option?",
        "cot": "Candy is not nutritious. Better to wait for a meal.",
        "answer": "Wait for proper food",
        "domain": "commonsense",
    },
    {
        "question": "You're outside without sunscreen and it’s sunny. What might happen?",
        "cot": "You may get sunburned due to UV exposure.",
        "answer": "Sunburn",
        "domain": "commonsense",
    },
    {
        "question": "You smell gas in the house. What should you do?",
        "cot": "Gas leaks are dangerous. Leave the house and call emergency services.",
        "answer": "Leave and call for help",
        "domain": "commonsense",
    },
    {
        "question": "You’re tired and it’s 11 PM. What’s the next reasonable action?",
        "cot": "Tiredness and late time suggest it’s time to sleep.",
        "answer": "Go to bed",
        "domain": "commonsense",
    },
    {
        "question": "Your phone isn't charging. What should you check first?",
        "cot": "Check if the cable is plugged in and working.",
        "answer": "Check cable and power",
        "domain": "commonsense",
    },
    {
        "question": "You hear a loud alarm while cooking. What might be the cause?",
        "cot": "Smoke detector may have been triggered by cooking fumes.",
        "answer": "Smoke set off the alarm",
        "domain": "commonsense",
    },
    {
        "question": "If you're running late for a meeting and the road is blocked, what should you do?",
        "cot": "You need to reach the meeting. The road is blocked, so you should find an alternate route or call ahead.",
        "answer": "Take a different route or inform the meeting host.",
        "domain": "commonsense",
    },
    # TRIVIA
    {
        "question": "What planet is known as the Red Planet?",
        "cot": "The Red Planet is named for its color. Mars has iron oxide (rust) on its surface, making it appear red.",
        "answer": "Mars",
        "domain": "trivia",
    },
    {
        "question": "What is the capital of France?",
        "cot": "France's capital city is Paris.",
        "answer": "Paris",
        "domain": "trivia",
    },
    {
        "question": "Who painted the Mona Lisa?",
        "cot": "The Mona Lisa was painted by Leonardo da Vinci.",
        "answer": "Leonardo da Vinci",
        "domain": "trivia",
    },
    {
        "question": "What is the largest planet in our solar system?",
        "cot": "Jupiter is the biggest planet in the solar system.",
        "answer": "Jupiter",
        "domain": "trivia",
    },
    {
        "question": "Which language has the most native speakers?",
        "cot": "Mandarin Chinese has the highest number of native speakers.",
        "answer": "Mandarin Chinese",
        "domain": "trivia",
    },
    {
        "question": "Which element has the chemical symbol O?",
        "cot": "O is the symbol for oxygen.",
        "answer": "Oxygen",
        "domain": "trivia",
    },
    {
        "question": "How many continents are there?",
        "cot": "There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, Australia.",
        "answer": "7",
        "domain": "trivia",
    },
    {
        "question": "What gas do plants use for photosynthesis?",
        "cot": "Plants absorb carbon dioxide to produce oxygen.",
        "answer": "Carbon dioxide",
        "domain": "trivia",
    },
    {
        "question": "Who discovered gravity after observing a falling apple?",
        "cot": "Isaac Newton formulated gravity after observing an apple fall.",
        "answer": "Isaac Newton",
        "domain": "trivia",
    },
    {
        "question": "Which country is known as the Land of the Rising Sun?",
        "cot": "Japan is famously known as the Land of the Rising Sun.",
        "answer": "Japan",
        "domain": "trivia",
    },
    {
        "question": "What is the longest river in the world?",
        "cot": "The Nile is often considered the longest river.",
        "answer": "Nile",
        "domain": "trivia",
    },
    {
        "question": "What is the smallest prime number?",
        "cot": "The first and smallest prime number is 2.",
        "answer": "2",
        "domain": "trivia",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "cot": "The famous play was written by William Shakespeare.",
        "answer": "William Shakespeare",
        "domain": "trivia",
    },
    {
        "question": "Which ocean is the largest?",
        "cot": "The Pacific Ocean is the largest.",
        "answer": "Pacific",
        "domain": "trivia",
    },
    {
        "question": "What planet is closest to the sun?",
        "cot": "Mercury is the innermost planet.",
        "answer": "Mercury",
        "domain": "trivia",
    },
    {
        "question": "What is H2O commonly known as?",
        "cot": "H2O is the chemical formula for water.",
        "answer": "Water",
        "domain": "trivia",
    },
    {
        "question": "Which animal is the fastest on land?",
        "cot": "The cheetah is the fastest land animal.",
        "answer": "Cheetah",
        "domain": "trivia",
    },
    {
        "question": "How many sides does a hexagon have?",
        "cot": "A hexagon has 6 sides.",
        "answer": "6",
        "domain": "trivia",
    },
    {
        "question": "What is the name of the fairy in Peter Pan?",
        "cot": "The fairy companion of Peter Pan is Tinker Bell.",
        "answer": "Tinker Bell",
        "domain": "trivia",
    },
    {
        "question": "In which year did World War II end?",
        "cot": "World War II ended in 1945.",
        "answer": "1945",
        "domain": "trivia",
    },
    {
        "question": "What instrument has 88 keys?",
        "cot": "A standard piano has 88 keys.",
        "answer": "Piano",
        "domain": "trivia",
    },
    {
        "question": "In which year did humans first land on the Moon?",
        "cot": "Apollo 11 landed in July 1969. Neil Armstrong stepped onto the lunar surface that year.",
        "answer": "1969",
        "domain": "trivia",
    },
]

with open("cot_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("Saved cot_dataset.json with", len(dataset), "examples.")
