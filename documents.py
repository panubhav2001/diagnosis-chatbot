from langchain_core.documents import Document

docs = [
    Document(
        page_content=(
            "Gastrointestinal (GI) bleeding is a symptom of many digestive system disorders, including reflux, "
            "ulcers and cancer. It can occur in any part of the digestive system (GI tract), which runs from the "
            "mouth to the anus. Bleeding can be mild and ongoing or come on suddenly and be life-threatening. "
            "Symptoms include abdominal cramping, dark-colored or bloody stool, pale appearance, shortness of breath, "
            "tiredness, vomit with blood or a substance that looks like coffee grounds, and weakness. Diagnosis involves "
            "blood tests, fecal occult blood tests, CT scans, GI X-rays, endoscopy, and potentially more intensive tests "
            "like angiography or capsule endoscopy if initial tests do not pinpoint the source. Treatments range from "
            "medication to manage symptoms and aid clotting to procedures such as endoscopy for direct intervention or "
            "emergency surgery for severe cases."
        ),
        metadata={
            "condition": "Gastrointestinal (GI) Bleeding",
            "symptoms": (
                "abdominal cramping, dark-colored poop or regular-colored poop with blood in it, "
                "pale appearance, shortness of breath, tiredness, vomit with blood or a substance that looks like coffee grounds, "
                "weakness and fatigue"
            ),
            "diagnosis": (
                "blood tests, fecal occult blood test, CT scan, GI X-rays, upper endoscopy, balloon enteroscopy, "
                "colonoscopy, sigmoidoscopy, angiography, capsule endoscopy, radionuclide scan"
            ),
            "triage": (
                "urgent care needed for acute symptoms like severe bleeding, dizziness, difficulty urinating, rapid pulse, or shock; "
                "non-emergency follow-ups for chronic symptoms"
            )
        },
    ),
    Document(
        page_content=(
            "Gastrointestinal Diseases\n\n"
            "GERD, diarrhea and colorectal cancer are examples of gastrointestinal diseases. When examined, "
            "some diseases show nothing wrong with the GI tract, but there are still symptoms. Other diseases "
            "have symptoms, and there are also visible irregularities in the GI tract. Most gastrointestinal "
            "diseases can be prevented and/or treated.\n\n"

            "What are gastrointestinal diseases?\n\n"
            "Gastrointestinal diseases affect your gastrointestinal (GI) tract, from mouth to anus. There are two types: "
            "functional and structural. Some examples include colitis, food poisoning, lactose intolerance and diarrhea.\n\n"

            "What are functional gastrointestinal diseases?\n\n"
            "Functional diseases are those in which the GI tract looks normal when examined, but doesn't move properly. "
            "They are the most common problems affecting the GI tract (including your colon and rectum). "
            "Constipation, irritable bowel syndrome (IBS), nausea, gas, bloating and diarrhea are common examples. "
            "Many factors can upset your GI tract and its motility (ability to keep moving), including: ... (text truncated)\n\n"

            "What are some of the common gastrointestinal diseases that healthcare providers treat?\n\n"
            "Healthcare providers who specialize in gastrointestinal diseases are called gastroenterologists. "
            "Surgeons who specialize in gastrointestinal diseases are called colorectal surgeons (proctologists). "
            "Some of the most common conditions they treat include: ... (text truncated)\n\n"

            "Prevention\n\n"
            "Can gastrointestinal diseases be prevented?\n\n"
            "Many intestinal disease can be prevented or minimized by maintaining a healthy lifestyle, practicing good "
            "bowel habits and getting screened for cancer. A colonoscopy is recommended for average-risk patients at age 45. "
            "If you have a family history of colorectal cancer or polyps, a colonoscopy may be recommended at a younger age. "
            "Typically, a colonoscopy is recommended 10 years younger than the affected family member. "
            "(For example, if your brother was diagnosed with colorectal cancer or polyps at age 45, you should begin screening "
            "at age 35.) If you have symptoms of colorectal cancer you should consult your healthcare provider right away. "
            "Common symptoms include: ... (text truncated)\n\n"

            "Other types of gastrointestinal diseases\n\n"
            "There are many other gastrointestinal diseases. Some are discussed, but others are not covered here. Other functional "
            "and structural diseases include peptic ulcer disease, gastritis, gastroenteritis, celiac disease, Crohn's disease, "
            "gallstones, fecal incontinence, lactose intolerance, Hirschsprung disease, abdominal adhesions, Barrett's esophagus, "
            "appendicitis, indigestion (dyspepsia), intestinal pseudo-obstruction, pancreatitis, short bowel syndrome, "
            "Whipple’s disease, Zollinger-Ellison syndrome, malabsorption syndromes and hepatitis."
        ),
        metadata={
            "condition": "Gastrointestinal Diseases",
            "symptoms": (
                "abdominal cramping, diarrhea, rectal bleeding, abdominal cramps, urgency to empty bowels, "
                "change in bowel habits, blood in stool, narrow stool, abdominal pain, weight loss, fatigue, "
                "gas, bloating, constipation"
            ),
            "diagnosis": (
                "colonoscopy, biopsy, blood tests, fecal occult blood test, imaging tests (CT scans, X-rays), "
                "endoscopy, sigmoidoscopy"
            ),
            "triage": (
                "seek immediate medical care for symptoms like rectal bleeding, abdominal pain, or weight loss; "
                "regular screenings recommended for individuals at higher risk or with concerning symptoms"
            )
        },
    ),
    Document(
        page_content=(
            "Stomach Flu\n\n"
            "Stomach flu, viral gastroenteritis, is a viral infection in your digestive system. It causes gastrointestinal "
            "symptoms like vomiting and diarrhea. It’s usually brief, but can be very contagious.\n\n"

            "Overview\n\n"
            "What is stomach flu?\n\n"
            "Stomach flu is a viral infection that affects your stomach and intestines. The medical term is viral gastroenteritis. "
            "“Gastro” means stomach and “enter” means small intestine. “Itis” means inflammation, which is usually due to an "
            "infection. And “viral” means that a virus has caused the infection. ... (text truncated)\n\n"

            "Why is viral gastroenteritis called “stomach flu?”\n\n"
            "Stomach flu isn’t related to “the flu” (influenza), which is a viral infection in your respiratory system. Different "
            "viruses cause the two conditions, and they affect different body systems. It’s not clear how the nickname “stomach flu” "
            "developed or why it’s been associated with influenza, but we can speculate. ... (text truncated)\n\n"

            "How common is stomach flu?\n\n"
            "Viral gastroenteritis is extremely common worldwide, but it’s hard to estimate exactly how many people get it each year. "
            "Many different viruses cause it, and most people don’t get clinically tested for it. Experts estimate that norovirus, the "
            "most common cause, infects 685 million people every year. ... (text truncated)\n\n"

            "Symptoms and Causes\n\n"
            "The most common stomach flu symptoms are nausea, vomiting and diarrhea.\n"
            "With stomach flu, gastrointestinal symptoms often come on suddenly.\n"
            "What are stomach flu symptoms?\n\n"
            "The most common stomach flu symptoms are:\n\n"
            "Diarrhea.\n"
            "Nausea and vomiting.\n"
            "Loss of appetite.\n"
            "Abdominal pain and cramping.\n"
            "These symptoms come from inflammation in your stomach and intestines. ... (text truncated)\n\n"

            "What does the beginning of stomach flu feel like?\n\n"
            "For many people, stomach flu symptoms seem to come on suddenly and out of nowhere. You might throw up or have diarrhea many "
            "times on that first day. Symptoms occur one to two days after you were exposed to the virus. Fortunately, they’re usually "
            "over just as quickly, resolving in one to two days. ... (text truncated)\n\n"

            "What are the stages of stomach flu?\n\n"
            "The stages of stomach flu infection are:\n\n"
            "Exposure. You’re most likely to get the stomach flu from someone in your community, especially in a closed environment like "
            "a school, nursing home or cruise ship. ... (text truncated)\n\n"

            "How long does stomach flu last?\n\n"
            "Stomach flu usually only lasts a few days, but it may last up to a week or two in severe cases. People with weaker immune "
            "systems may have a harder time defeating the virus, and it may take longer. ... (text truncated)\n\n"

            "Is stomach flu contagious?\n\n"
            "Yes, it’s very contagious. You should limit your contact with others when you have it. If you live with others, make sure to "
            "wash your hands often and disinfect shared surfaces, especially in the bathroom. ... (text truncated)\n\n"

            "What causes stomach flu?\n\n"
            "Many different viruses can infect your gastrointestinal system, causing gastroenteritis.\n\n"
            "The most common ones are:\n\n"
            "Norovirus. This is the leading cause of stomach flu in adults, estimated to account for 50% of cases worldwide. ... (text truncated)\n\n"

            "How does stomach flu spread?\n\n"
            "Stomach flu usually spreads by the “fecal-to-oral route”. The virus lives in the poop — and vomit — of infected people. "
            "Microscopic traces of infected poop or vomit may linger on people’s hands or surfaces. ... (text truncated)\n\n"

            "Who gets stomach flu?\n\n"
            "Anyone can get stomach flu, but certain people are more vulnerable. If you have a weaker-than-average immune system, you "
            "might be more likely to get an infection or get a more severe infection. ... (text truncated)\n\n"

            "Diagnosis and Tests\n\n"
            "How do healthcare providers diagnose stomach flu?\n\n"
            "Healthcare providers often diagnose gastroenteritis based on your symptoms. They won’t know if it’s viral or which virus "
            "it is without doing a lab test to find out. But most of the time, this isn’t necessary. ... (text truncated)\n\n"

            "Management and Treatment\n\n"
            "How do you get rid of stomach flu?\n\n"
            "Your immune system gets rid of stomach flu through its own natural processes. It just takes a few days to do its work. Your "
            "symptoms, while unpleasant, are a sign that your immune system is working. There’s no medicine for stomach flu. ... (text truncated)\n\n"

            "What helps stomach flu go away faster?\n\n"
            "Some research shows that taking probiotics may help stomach flu go away faster. Probiotics are helpful bacteria that live in "
            "your gastrointestinal system. ... (text truncated)\n\n"

            "What should you eat when you have stomach flu?\n\n"
            "What you eat won’t improve stomach flu, but it can make it worse. Foods high in fat, sugar, caffeine or dairy milk might "
            "make you more likely to throw up or have diarrhea. ... (text truncated)\n\n"

            "Can healthcare providers treat severe symptoms or complications?\n\n"
            "Healthcare providers can treat dehydration with intravenous fluids. This is a way of delivering hydration directly to your "
            "bloodstream, bypassing your digestive system. ... (text truncated)\n\n"

            "When should I see a healthcare provider about stomach flu?\n\n"
            "Contact a healthcare provider if: ... (text truncated)\n\n"

            "Prevention\n\n"
            "How can stomach flu be prevented?\n\n"
            "You can reduce your risk of getting stomach flu or spreading it to others by practicing good hygiene. ... (text truncated)\n\n"

            "Outlook / Prognosis\n\n"
            "When will I feel better?\n\n"
            "For most people, symptoms get better in a few days. Contact a healthcare provider if they aren’t improving. ... (text truncated)\n\n"

            "When can I return to work or school?\n\n"
            "It’s best to say isolated until two days after your symptoms have stopped. This is the time when you’re most contagious. ... (text truncated)\n\n"

            "Additional Common Questions\n\n"
            "What’s the difference between stomach flu and the flu (influenza)?\n\n"
            "Stomach flu isn’t related to “the flu” (influenza), which is a viral infection in your respiratory system. The flu vaccine "
            "won’t protect you from stomach flu. Different viruses cause these two conditions, and they affect different body systems. ... (text truncated)\n\n"

            "What are “flu-like symptoms?”\n\n"
            "People probably mean different things when they use this phrase. They might be referring to symptoms of influenza. ... (text truncated)\n\n"

            "What's the difference between stomach flu and food poisoning?\n\n"
            "Stomach flu and food poisoning are both infections that cause gastroenteritis, with similar symptoms. The different names "
            "describe the different origins of the infection. ... (text truncated)"
        ),
        metadata={
            "condition": "Stomach Flu",
            "symptoms": "nausea, vomiting, diarrhea, loss of appetite, abdominal pain, cramping, fever, chills, fatigue, body aches, headaches, swollen lymph nodes",
            "diagnosis": "clinical diagnosis based on symptoms, lab testing may be done in some cases",
            "triage": (
                "contact healthcare provider if symptoms persist for more than four days, if fever is above 102°F (39°C) for four days, "
                "if unable to urinate or defecate for two days, if signs of dehydration are present, if blood is present in stool, if severe "
                "abdominal pain is experienced"
            )
        },
    ),
    Document(
        page_content=(
            "Rotavirus\n\n"
            "Rotavirus is a gastrointestinal (stomach and intestines) infection that mostly affects children. "
            "While the disease used to happen more often, rotavirus vaccines have kept many kids healthy. For those "
            "who do get infected, rotavirus can cause severe diarrhea and vomiting. This can lead to dehydration, "
            "so make sure to keep your child hydrated.\n\n"

            "Overview\n\n"
            "What is rotavirus?\n\n"
            "Rotavirus is a contagious gastrointestinal (GI) infection that causes inflammation of the stomach and "
            "intestines (gastroenteritis). This can lead to severe diarrhea and vomiting, especially in young children. "
            "Kids tend to get rotavirus during the winter and spring. It spreads when they come in contact with the poop "
            "(stool) of someone who has it and then touch their own mouth. ... (text truncated)\n\n"

            "What are the symptoms of rotavirus?\n\n"
            "The most common rotavirus symptoms include:\n\n"
            "Severe, watery diarrhea.\n"
            "Vomiting.\n"
            "Fever.\n"
            "Loss of appetite.\n"
            "The vomiting and diarrhea may also cause dehydration in babies and young children. Contact your child’s "
            "healthcare provider right away if your child has symptoms of dehydration, including: ... (text truncated)\n\n"

            "Why is dehydration serious for babies?\n\n"
            "Dehydration means your baby doesn’t have enough water and salts to function properly. Babies younger than "
            "1 year old can dehydrate easily. Many times, dehydrated children need IV fluids to rehydrate. If dehydration "
            "gets severe, a child could start convulsing (experiencing sudden, erratic body movements) or go into shock. "
            "It could be life-threatening. ... (text truncated)\n\n"

            "What causes rotavirus?\n\n"
            "Rotavirus is a virus that spreads through hand-to-mouth contact. It shows up in an infected child’s poop "
            "(stool) a few days before symptoms start. And it can remain in your child’s bowel movements for up to 10 days "
            "after symptoms stop. ... (text truncated)\n\n"

            "Is rotavirus contagious?\n\n"
            "Rotavirus is very contagious. Before the rotavirus vaccines, most children got rotavirus by the age of 5. "
            "Since the vaccines, the number of children who get rotavirus infection has dropped. ... (text truncated)\n\n"

            "Who’s at risk for rotavirus?\n\n"
            "Typically, children in daycare or other programs with large numbers of children are at a higher risk. "
            "Children between the ages of 3 months and 3 years who haven’t received the vaccine tend to get the most "
            "severe disease. ... (text truncated)\n\n"

            "Can adults get rotavirus?\n\n"
            "Rotavirus in adults does occur, but they tend to get less sick than young children do. Adults at risk for "
            "rotavirus include those who:\n\n"
            "Are age 65 and older.\n"
            "Care for children who have rotavirus.\n"
            "Have compromised immune systems.\n"
            "Diagnosis and Tests\n\n"
            "How is rotavirus diagnosed?\n\n"
            "If your child has signs of rotavirus, contact their healthcare provider. Providers can often diagnose "
            "rotavirus based on symptoms and a physical examination. In some cases, they may take a stool (poop) sample "
            "to test it for rotavirus. But this step usually isn’t necessary. ... (text truncated)\n\n"

            "Management and Treatment\n\n"
            "Is there a medicine for rotavirus?\n\n"
            "A virus causes rotavirus, not bacteria. So antibiotics won’t help your child feel better. The virus should "
            "clear on its own after about a week. The main rotavirus treatment is to keep your child hydrated. ... (text truncated)\n\n"

            "Can rotavirus be treated at home?\n\n"
            "Yes. Contact your child’s healthcare provider if you notice rotavirus symptoms. They may recommend you:\n\n"
            "Give your child smaller, more frequent feedings instead of larger meals.\n"
            "Make sure your child gets enough fluids.\n"
            "Use an electrolyte replacement such as Pedialyte®. Electrolytes help keep body systems in balance, but "
            "you can lose them through vomiting and diarrhea. Replacements can fix that. Just make sure to follow the "
            "instructions on the label carefully. ... (text truncated)\n\n"

            "Prevention\n\n"
            "How can I prevent rotavirus?\n\n"
            "The best way to prevent rotavirus and protect the health of your family is to make sure they get one of the "
            "rotavirus vaccines. About 70% of children who receive the vaccine don’t get rotavirus. For those who still get "
            "infected, the symptoms are much milder. ... (text truncated)\n\n"

            "Outlook / Prognosis\n\n"
            "What’s the outlook for children with rotavirus?\n\n"
            "Most children recover from rotavirus without long-term health effects. Symptoms last about a week. If your child "
            "becomes dehydrated, it could lead to serious complications and even death. Call your healthcare provider right "
            "away if your child shows symptoms of rotavirus. Your provider can help you prevent dehydration. ... (text truncated)\n\n"

            "Can my child get rotavirus again?\n\n"
            "Children can get re-infected. However, a second infection tends to be milder. ... (text truncated)\n\n"

            "How long does rotavirus last?\n\n"
            "Symptoms of rotavirus usually last from three to eight days. Most children are contagious for around 12 days in "
            "total. That’s because infection starts a few days before symptoms do. ... (text truncated)\n\n"

            "When can my child return to school or daycare?\n\n"
            "Your school or daycare will probably let you know how long to keep your child home. Usually, the requirement is "
            "that children have no symptoms for at least 24 hours before returning to these settings. ... (text truncated)\n\n"

            "Living With\n\n"
            "When should my child see their healthcare provider?\n\n"
            "Call your child’s provider if you notice an increase in vomiting or diarrhea. Also contact their provider if you "
            "see signs of dehydration, which may happen because of vomiting and diarrhea: ... (text truncated)\n\n"

            "What questions should I ask my child’s healthcare provider?\n\n"
            "If your child has rotavirus, ask their provider about:\n\n"
            "Medicine\n"
            "Which over-the-counter medications do you recommend to reduce fever? Are there any I shouldn’t use?\n"
            "How long should I give my child the medication? ... (text truncated)\n\n"

            "Additional Common Questions\n\n"
            "Rotavirus vs. norovirus — what’s the difference?\n\n"
            "Rotavirus and norovirus are both gastrointestinal infections that cause inflammation of your stomach and intestines, "
            "but they’re different conditions caused by different viruses. Rotavirus mostly affects children, while norovirus can "
            "affect people of any age. ... (text truncated)"
        ),
        metadata={
            "condition": "Rotavirus",
            "symptoms": "severe diarrhea, vomiting, fever, loss of appetite",
            "diagnosis": "clinical diagnosis based on symptoms and physical examination, stool sample testing in some cases",
            "triage": (
                "contact healthcare provider if symptoms persist, if signs of dehydration are present, "
                "if child becomes dehydrated, if symptoms worsen, if child is unable to keep down fluids or food, "
                "if there's blood in stool, if fever is persistent or high-grade, if child is unable to urinate or has reduced urine output"
            )
        },
    ),
]