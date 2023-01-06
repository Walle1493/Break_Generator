import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BertTokenizer
import pdb


# 1: Complex question -> Simple questions
def Solution1():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-break_data")
    model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-break_data")
    model.to(torch_device)

    def get_decomposition(question):
        input_text = "paraphrase: %s </s>" % question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=32)

        return tokenizer.decode(output[0])

    # question = "The composer of Sands Theme plays what type of guitar?"
    # question = "How were the people that the Somali Muslim Ajuran Empire made coins to proclaim independence from, expelled from the country where Mohinga is eaten?"
    question = "Who is the spouse of the Green performer?"

    result = get_decomposition(question)

    # logging
    print("This is Decomposition:")
    print("Complex question:")
    print(question)
    print("Simple questions:")
    print(result)


# 2: Single questions -> Complex question
def Solution2():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-break_data-question-retrieval")
    model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-break_data-question-retrieval")
    model.to(torch_device)

    def get_nautural_question(decomposition):
        input_text = 'translate QDMRs to Natural Language %s </s>' % decomposition
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=64)

        return tokenizer.decode(output[0])

    # decomposition = "return the city that was the birthplace of Bernard Berrian ;return the  city that was  the home of Pablo Picasso ;return the  city of both #1 and #2"
    decomposition = "Who was the performer of Green; Who was the spouse of #1"
    result = get_nautural_question(decomposition)
    
    # logging
    print("This is Composition:")
    print("Simple questions:")
    decomp = decomposition.split(";")
    for i, d in enumerate(decomp):
        print("Question %d: " % (i + 1), end="")
        print(d)
    print("Complex question:")
    print(result)


# Batch Decomposition
def Solution3():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-break_data")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    # model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-break_data")
    model = BartForConditionalGeneration.from_pretrained("/home/mxdong/Model/Decomposition/MuSiQue/Bart-Large")
    model.to(torch_device)

    def get_decomposition(question):
        # input_text = "paraphrase: %s </s>" % question
        input_text = question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=100)

        return tokenizer.decode(output[0])

    questions = [
        "Who is the spouse of the Green performer?",
        "Who founded the company that distributed the film UHF?",
        "What administrative territorial entity is the owner of Ciudad Deportiva located?",
        "Who is the president of the newly declared independent country, that established the Timor Leste Commission of Truth and Friendship, with the country containing the airport that includes Lion Air?",
        "How long was the place where the Yongle Emperor greeted who the Ming court thought representatives were sent by the capitol of the city in which Guangling District is located?",
        "What are the Genesis's advantages over the game platform with a 3 letter name abbreviation, featuring a game named after the league that chooses where the Super Bowl is held?",
    ]
    # answers = [
    #     "Green >> performer; #1 spouse",
    #     "UHF >> distributed by; #1 >> founded by",
    #     "Ciudad Deportiva >> owned by; #1 >> located in the administrative territorial entity",
    #     "What airport is Lion Air part of?; #1 >> country; #2 \u2013Timor Leste Commission of Truth and Friendship >> country; who is the president of newly declared independent country #3",
    #     "Who did the Ming court think the representatives were sent by?; Where did the Yongle Emperor greet the #1 ?; Guangling District >> located in the administrative territorial entity; How long had #2 been the capital city of #3 ?",
    #     "who chooses where the super bowl is held; #1 >> platform; What is the abbreviation of #2 ?; What were the Genesis's advantages over the #3 ?"
    # ]

    for question in questions:
        decomp = get_decomposition(question)
        print("[Complex]", end=" ")
        print(question)
        print("[Simples]", end=" ")
        print(decomp)


# Batch Decomposition 2
def Solution4():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("msterbentz/t5-base-break-high")
    model = AutoModelForSeq2SeqLM.from_pretrained("msterbentz/t5-base-break-high")
    model.to(torch_device)

    def get_decomposition(question):
        input_text = "paraphrase: %s </s>" % question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=64)

        return tokenizer.decode(output[0])

    questions = [
        "Who is the spouse of the Green performer?",
        "Who founded the company that distributed the film UHF?",
        "What administrative territorial entity is the owner of Ciudad Deportiva located?",
        "Who is the president of the newly declared independent country, that established the Timor Leste Commission of Truth and Friendship, with the country containing the airport that includes Lion Air?",
        "How long was the place where the Yongle Emperor greeted who the Ming court thought representatives were sent by the capitol of the city in which Guangling District is located?",
        "What are the Genesis's advantages over the game platform with a 3 letter name abbreviation, featuring a game named after the league that chooses where the Super Bowl is held?",
    ]
    # answers = [
    #     "Green >> performer; #1 spouse",
    #     "UHF >> distributed by; #1 >> founded by",
    #     "Ciudad Deportiva >> owned by; #1 >> located in the administrative territorial entity",
    #     "What airport is Lion Air part of?; #1 >> country; #2 \u2013Timor Leste Commission of Truth and Friendship >> country; who is the president of newly declared independent country #3",
    #     "Who did the Ming court think the representatives were sent by?; Where did the Yongle Emperor greet the #1 ?; Guangling District >> located in the administrative territorial entity; How long had #2 been the capital city of #3 ?",
    #     "who chooses where the super bowl is held; #1 >> platform; What is the abbreviation of #2 ?; What were the Genesis's advantages over the #3 ?"
    # ]

    for question in questions:
        decomp = get_decomposition(question)
        print("[Complex]", end=" ")
        print(question)
        print("[Simples]", end=" ")
        print(decomp)


# LZF Demo
def Solution5():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("/home/mxdong/Model/Decomposition/MuSiQue/T5-Large")
    model.to(torch_device)

    def get_decomposition(question):
        input_text = "paraphrase: %s </s>" % question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=32)

        return tokenizer.decode(output[0])

    question = "Who is the spouse of the Green performer?"

    result = get_decomposition(question)

    # logging
    print("Complex question:", end=" ")
    print(question)
    print("Simple questions: ", end=" ")
    print(result)


# Batch Decomposition 2-4 hop
def Solution6():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("/home/mxdong/Model/Decomposition/MuSiQue/Bart-Large")
    model.to(torch_device)

    def get_decomposition(question):
        # input_text = "paraphrase: %s </s>" % question
        input_text = question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=100)

        return tokenizer.decode(output[0])

    questions = [
        "Who founded the company that distributed the film UHF?",
        "What is the birthplace of the man who does the voice of Stan on the series that includes the episode The Hobbit?",
        "When did the party holding the majority in the House of Representatives take control of the determiner of rules of the House and Senate?",
        "What were the Genesis's advantages over the platform with a three letter abbreviation, that had a video game named after the league in charge of the Super Bowl halftime show?",
        "The military group of which the Air Defense Artillery is a branch was unprepared for the invasion of the territory the Nazis occupied. The country of this group was the only communist country to have an embassy where?",
        "Despite being located in East Belgium, the Carnival of the birth place of Guido Maus harks purely to an area. What was the language having the same name as this area of the era with Fastrada's spouse's name later known as?",
    ]
    # answers = [
    #     "UHF >> distributed by; #1 >> founded by",
    #     "The Hobbit >> part of the series; who does the voice of stan on #1; #2 >> place of birth",
    #     "who determines the rules of the us house and us senate; who hold the majority in the house of representatives; when did #2 take control of the #1",
    #     "who is in charge of the super bowl halftime show; #1 >> platform; What is the abbreviation of #2 ?; What were the Genesis's advantages over the #3 ?",
    #     "What territory did the Nazi occupy?; The Air Defense Artillery is a branch of what?; What #2 was unprepared for the invasion of #1 ?; #3 was the only communist country to have an embassy where?",
    #     "Guido Maus >> place of birth; Despite being located in East Belgium, #1 's Carnival harks purely to what area?; What is Fastrada's spouse's name?; What was the #2 of #3 's era later known as?"
    # ]

    for question in questions:
        decomp = get_decomposition(question)
        print("[Complex]", end=" ")
        print(question)
        print("[Simples]", end=" ")
        print(decomp)


# Batch Decomposition 2-4 hop
def Solution7():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("/home/mxdong/Model/MultiStep/MuSiQue/Bart-Large")
    model.to(torch_device)

    def get_decomposition(question):
        # input_text = "paraphrase: %s </s>" % question
        input_text = question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=100)

        return tokenizer.decode(output[0])

    questions = [
        "Who is the spouse of the Green performer?",
        "Who is the spouse of the Green performer? </s> Green >> performer Steve Hillage",
        "Who founded the company that distributed the film UHF?",
        "Who founded the company that distributed the film UHF? </s> UHF >> distributed by Orion Pictures",
        "How were the people that the Somali Muslim Ajuran Empire made coins to proclaim independence from, expelled from the country where Mohinga is eaten?",
        "How were the people that the Somali Muslim Ajuran Empire made coins to proclaim independence from, expelled from the country where Mohinga is eaten? </s> New coins were a proclamation of independence by the Somali Muslim Ajuran Empire from whom? the Portuguese",
        "How were the people that the Somali Muslim Ajuran Empire made coins to proclaim independence from, expelled from the country where Mohinga is eaten? </s> New coins were a proclamation of independence by the Somali Muslim Ajuran Empire from whom? the Portuguese </s> Which was the country for Mohinga? Myanmar",
    ]
    # answers = [
    #     "Green >> performer",
    #     "Steve Hillage >> spouse </s> </s>",
    #     "UHF >> distributed by",
    #     "Orion Pictures >> founded by </s> </s>",
    #     "New coins were a proclamation of independence by the Somali Muslim Ajuran Empire from whom?",
    #     "Which was the country for Mohinga?",
    #     "How were the the Portuguese expelled from Myanmar ? </s> </s>"
    # ]

    for question in questions:
        decomp = get_decomposition(question)
        print("[Complex]", end=" ")
        print(question)
        print("[Simples]", end=" ")
        print(decomp)


# Batch Decomposition 2-4 hop
def Solution8():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("/home/mxdong/Model/Seq2seq/MuSiQue/Bart-Large-eos-64")
    model.to(torch_device)

    def get_decomposition(question):
        # input_text = "paraphrase: %s </s>" % question
        input_text = question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=100)

        return tokenizer.decode(output[0])

    questions = [
        "Who founded the company that distributed the film UHF?",
        "What is the birthplace of the man who does the voice of Stan on the series that includes the episode The Hobbit?",
        "When did the party holding the majority in the House of Representatives take control of the determiner of rules of the House and Senate?",
        "What were the Genesis's advantages over the platform with a three letter abbreviation, that had a video game named after the league in charge of the Super Bowl halftime show?",
        "The military group of which the Air Defense Artillery is a branch was unprepared for the invasion of the territory the Nazis occupied. The country of this group was the only communist country to have an embassy where?",
        "Despite being located in East Belgium, the Carnival of the birth place of Guido Maus harks purely to an area. What was the language having the same name as this area of the era with Fastrada's spouse's name later known as?",
    ]
    # answers = [
    #     "UHF >> distributed by; #1 >> founded by",
    #     "The Hobbit >> part of the series; who does the voice of stan on #1; #2 >> place of birth",
    #     "who determines the rules of the us house and us senate; who hold the majority in the house of representatives; when did #2 take control of the #1",
    #     "who is in charge of the super bowl halftime show; #1 >> platform; What is the abbreviation of #2 ?; What were the Genesis's advantages over the #3 ?",
    #     "What territory did the Nazi occupy?; The Air Defense Artillery is a branch of what?; What #2 was unprepared for the invasion of #1 ?; #3 was the only communist country to have an embassy where?",
    #     "Guido Maus >> place of birth; Despite being located in East Belgium, #1 's Carnival harks purely to what area?; What is Fastrada's spouse's name?; What was the #2 of #3 's era later known as?"
    # ]

    for question in questions:
        decomp = get_decomposition(question)
        print("[Complex]", end=" ")
        print(question)
        print("[Simples]", end=" ")
        print(decomp)


def Solution9():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("/home/mxdong/Model/Decomposition/MuSiQue/Mix")
    model.to(torch_device)

    def get_decomposition(question):
        # input_text = "paraphrase: %s </s>" % question
        input_text = question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=100)

        return tokenizer.decode(output[0])

    questions = [
        "Who founded the company that distributed the film UHF?",
        "What is the birthplace of the man who does the voice of Stan on the series that includes the episode The Hobbit?",
        "When did the party holding the majority in the House of Representatives take control of the determiner of rules of the House and Senate?",
        "Who is the spouse of the Green performer? ; Green >> performer Steve Hillage",
    ]
    # answers = [
    #     "UHF >> distributed by; #1 >> founded by",
    #     "The Hobbit >> part of the series; who does the voice of stan on #1; #2 >> place of birth",
    #     "who determines the rules of the us house and us senate; who hold the majority in the house of representatives; when did #2 take control of the #1",
    #     "who is in charge of the super bowl halftime show; #1 >> platform; What is the abbreviation of #2 ?; What were the Genesis's advantages over the #3 ?",
    #     "What territory did the Nazi occupy?; The Air Defense Artillery is a branch of what?; What #2 was unprepared for the invasion of #1 ?; #3 was the only communist country to have an embassy where?",
    #     "Guido Maus >> place of birth; Despite being located in East Belgium, #1 's Carnival harks purely to what area?; What is Fastrada's spouse's name?; What was the #2 of #3 's era later known as?"
    # ]

    for question in questions:
        decomp = get_decomposition(question)
        print("[Complex]", end=" ")
        print(question)
        print("[Simples]", end=" ")
        print(decomp)
    

def Solution10():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("/home/mxdong/Model/Decomposition/MuSiQue/Mix")
    model.to(torch_device)

    def get_decomposition(question):
        # input_text = "paraphrase: %s </s>" % question
        input_text = question
        features = tokenizer([input_text], return_tensors='pt').to(torch_device)

        output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=100)

        return tokenizer.decode(output[0])

    questions = [
        "Who founded the company that distributed the film UHF?",
        "hen was the last time Duane Courtney's team beat the winner of the 1894-95 FA Cup?",
        "In what region of Phu Luong's country is John Phan's birthplace located?",
    ]
    # answers = [
    #     "UHF >> distributed by; #1 >> founded by",
    #     "The Hobbit >> part of the series; who does the voice of stan on #1; #2 >> place of birth",
    #     "who determines the rules of the us house and us senate; who hold the majority in the house of representatives; when did #2 take control of the #1",
    #     "who is in charge of the super bowl halftime show; #1 >> platform; What is the abbreviation of #2 ?; What were the Genesis's advantages over the #3 ?",
    #     "What territory did the Nazi occupy?; The Air Defense Artillery is a branch of what?; What #2 was unprepared for the invasion of #1 ?; #3 was the only communist country to have an embassy where?",
    #     "Guido Maus >> place of birth; Despite being located in East Belgium, #1 's Carnival harks purely to what area?; What is Fastrada's spouse's name?; What was the #2 of #3 's era later known as?"
    # ]

    for question in questions:
        decomp = get_decomposition(question)
        print("[Complex]", end=" ")
        print(question)
        print("[Simples]", end=" ")
        print(decomp)


if __name__ == "__main__":
    Solution10()

