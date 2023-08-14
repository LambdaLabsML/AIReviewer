import os
import string
import random
import time
import sqlite3
import json
import openai
from pathlib import Path
from datetime import datetime
import dataclasses
import re
from tqdm import tqdm
import pandas as pd

from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

prompt_template = \
    """Based on common sense, pretend to be a professional academic paper reviewer to help authors know their paper's feedback. Feel free to express the possible opinionss.
    
    {context}
    
    Question: {question}
    The possible review answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

os.environ["OPENAI_API_KEY"] = "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"  # lambda
openai.api_key = os.getenv("OPENAI_API_KEY")

chunk_size = 1024
chunk_overlap = 128


def pdf_retriever(pdf_path, embedding_function=OpenAIEmbeddings()):
    if not isinstance(pdf_path, Path):
        pdf_path = Path(pdf_path)
    loader = UnstructuredPDFLoader(pdf_path.as_posix())

    db_path = Path(f'db/{pdf_path.name}')
    pdf_path.parent.mkdir(exist_ok=True, parents=True)
    if not db_path.exists():
        print(f'Creating db at {db_path}')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        db = Chroma.from_documents(texts, embedding_function, persist_directory=db_path.as_posix())
        db.persist()
    else:
        print(f'Loading db from {db_path}')
        db = Chroma(persist_directory=db_path.as_posix(), embedding_function=embedding_function)

    retriever = db.as_retriever()
    return retriever


def generate_review(model_name='gpt-4'):
    """
    Using GPT-4 to generate the review based on paper PDF
    :param model_name:
    :return:
    """
    model = OpenAI(model_name=model_name, temperature=0)

    res = Path('cache') / 'NeurIPS2022.json'
    assert res.exists()
    res = json.load(res.open())

    path_cache = Path('cache')
    pd_list = list(path_cache.glob('*/*.pdf'))
    for pdf_path in tqdm(pd_list, desc='Processing', total=len(pd_list)):
        retriever = pdf_retriever(pdf_path)
        qa = RetrievalQA.from_chain_type(
            model,
            retriever=retriever,
            return_source_documents=True,
            # chain_type="map_reduce",
            chain_type="stuff",
            # chain_type="refine",
            chain_type_kwargs={"prompt": PROMPT}
        )

        question_meta_review = (
            "Give the final meta review. The output should strictly follow the format \"Recommandation: [Reject/Accept]\nCondifence:[Certain/Less Certain]\n[Your review]\". "
            "The review should be a short summary of explaination, such as the strengths and weaknesses of the paper.")
        ai_meta_review = qa(question_meta_review)['result']

        res[pdf_path.name[:-4]]['ai_meta_question'] = question_meta_review
        res[pdf_path.name[:-4]]['ai_meta_review'] = ai_meta_review

        questions = [
            # "Summary and contributions: Briefly summarize the paper and its contributions ",
            # "Strengths: Describe the strengths of the work. Typical criteria include: soundness of the claims (theoretical grounding, empirical evaluation), significance and novelty of the contribution.",
            # "Weaknesses: Explain the limitations of this work along the same axes as above. This is like above, but now focussing on the limitations of this work. Your comments should be detailed, specific, and polite. Please avoid vague, subjective complaints. Think about the times when you received an unfair, unjustified, short, or dismissive review. Try not to be that reviewer! Always be constructive and help the authors understand your viewpoint, without being dismissive or using inappropriate language. Remember that you are not reviewing your level of interest in the submission, but its scientific contribution to the field!",
            # "Correctness: Are the claims and method correct? Is the empirical methodology correct? Explain if there is anything incorrect with the paper. Incorrect claims or methodology are the primary reason for rejection. Be as detailed, specific and polite as possible. Thoroughly motivate your criticism so that authors will understand your point of view and potentially respond to you.",
            # "Clarity: Is the paper well written? Rate the clarity of exposition of the paper. Give examples of what parts of the paper need revision to improve clarity.",
            # "Relation to prior work: Is it clearly discussed how this work differs from previous contributions? Explain whether the submission is written with the due scholarship, relating the proposed work with the prior work in the literature. The related work section should not just list prior work, but explain how the proposed work differs from prior work appeared in the literature.",
            # "Reproducibility: Are there enough details to reproduce the major results of this work? Mark whether the work is reasonably reproducible. If it is not, lack of reproducibility should be listed among the weaknesses of the submission.",
            # "Additional feedback, comments, suggestions for improvement and questions for the authors. Add here any additional comment you might have about the submission, including questions and suggestions for improvement.",
            "Give the final meta review and overall score: You should NOT assume that you were assigned a representative sample of submissions, nor should you adjust your scores to match the overall conference acceptance rates. The “Overall Score” for each submission should reflect your assessment of the submission’s contributions. "
            "Score 10: Top 5% of accepted papers. Truly groundbreaking work. "
            "Score 9: Top 15% of accepted papers. An excellent submission; a strong accept. "
            "Score 8: Top 50% of accepted papers. A very good submission; a clear accept. "
            "Score 7: A good submission; accept. I vote for accepting this submission, although I would not be upset if it were rejected.) "
            "Score 6: Marginally above the acceptance threshold. I tend to vote for accepting this submission, but rejecting it would not be that bad. "
            "Score 5: Marginally below the acceptance threshold. I tend to vote for rejecting this submission, but accepting it would not be that bad. "
            "Score 4: An okay submission, but not good enough; a reject. I vote for rejecting this submission, although I would not be upset if it were accepted. "
            "Score 3: A clear reject. I vote and argue for rejecting this submission. "
            "Score 2: I'm surprised this work was submitted; a strong reject. "
            "Score 1: Trivial or wrong or already known.",
            # "Confidence score: "
            # "Score 5: You are absolutely certain about your assessment. You are very familiar with the related work. "
            # "Score 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. "
            # "Score 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked. "
            # "Score 2: You are willing to defend your assessment, but it is quite likely that you did not understand central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked. "
            # "Score 1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.",
            # "Have the authors adequately addressed the broader impact of their work, including potential negative ethical and societal implications of their work? Yes, no or only partially. In order to provide a balanced perspective, authors are required to include a statement of the potential broader impact of their work, including its ethical aspects and future societal consequences. Authors should take care to discuss both positive and negative outcomes. Indicate whether you believe the broader impact section was adequate.",
            # "Does the submission raise potential ethical concerns? This includes methods, applications, or data that create or reinforce unfair bias or that have a primary purpose of harm or injury. If so, please explain briefly. Yes or No. Explain if the submission might raise any potential ethical concern. Note that your rating should be independent of this. If the AC also shares this concern, dedicated reviewers with expertise at the intersection of ethics and ML will further review the submission. Your duty here is to flag only papers that might need this additional revision step.",
            # "Have you previously reviewed or area chaired (a version of) this work for another archival venue? Yes or No. This information will be useful to ACs and SACs in putting your review details in context of having already seen an earlier version. ACs and SACs have access to a resubmission statement by the authors which declares whether the submission was previously rejected and what changes were made to the current version. The AC may decide to share this information with reviewers if needed.",
            # "Confidential comments for the area chair. If you have comments that you wish to be kept confidential from the authors, you can use the “Confidential Comments to Area Chair” text field. Such comments might include explicit comparisons of the submission to other submissions and criticisms that are more bluntly stated. If you accidentally find out the identities of the authors, please do not divulge the identities to anyone.",
        ]
    with open('cache/NeurIPS2022.json', 'w') as f:
        json.dump(res, f, indent=4)
        # rounds = []
        # for question in questions:
        #     result = qa(question)
        #     # source = result["source_documents"]
        #     print(f"-> **Question**: {question} \n")
        #     print(f"**Answer**: {result['result']} \n")
        #     # add question and answer to the file
        #     rounds.append({"question": question, "answer": result["result"]})


def _chatgpt(
        sys_prompt="",
        user_prompt="Tell the world about the ChatGPT API in the style of a pirate.",
        history=None,
        # model="gpt-3.5-turbo"
        model="gpt-4",
        sleep=1.0,
):
    messages = []

    if history:
        messages = history
    else:
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

    messages.append({"role": "user", "content": user_prompt})

    res = []
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
        res = response['choices'][0]['message']['content']
        res = {"role": "assistant", "content": res}
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass
    except openai.error.ServiceUnavailableError as e:
        # retry
        print(f"OpenAI API request failed: {e}")
        time.sleep(sleep)
        return _chatgpt(sys_prompt=sys_prompt, user_prompt=user_prompt, history=history, model=model, sleep=sleep * 2)

    return res


def ai_judge():
    res_path = Path('cache') / 'NeurIPS2022.json'
    assert res_path.exists()
    res = json.load(res_path.open())

    history = []
    sys_prompt = (
        "Please act as an impartial judge and evaluate the similarity of the responses provided by a human reviewer (a) and AI reviewer (b) to a submitted paper. "
        "You judge whether the AI reviewer is similar to the human one. "
        # "Give a similarity score from 0 to 10: final recommendations (acceptance or rejection) is the most important factor, weighting 0.5. But confidence, content and explanation are less important, weighting 0.2 and 0.2 and etc. "
        # "For example, the score should be over 5 if the two have the same recommendation, but not over 5 if they are different. "
        "Avoid any position, length and order biases to influence your evaluation. "
        "Strictly following this format: Similarity Score: [score] \n Explanation: [explaination] \n\n"
    )

    prompt_template = ("[User Question]\n{question}\n\n[The Start of Human Meta Review]\n{answer_a}\n[The End of Human Meta Review]\n\n[The Start of AI Meta Review]\n{answer_b}\n[The End of AI Meta Review]\n"
                       "Give a similarity score from 0 to 10: final recommendations (acceptance or rejection) is the most important factor, weighting 5 to 7. But confidence, content and explanation are less important, weighting 1 to 2. "
                       "For example, the score should be over 5 if the two have the same recommendation, but not over 5 if they are different. Do not be too strict."
                       )

    for k, v in res.items():
        user_prompt = prompt_template.format(
            question="What is the estimated similarity score of the AI reviewer's review to the human reviewer's review?",
            answer_a=v['meta_review'], answer_b=v['ai_meta_review'])

        # if not v['ai_meta_review'].startswith("The paper demonstrates strong theoretical"):
        #     continue

        message_ai = _chatgpt(sys_prompt=sys_prompt, user_prompt=user_prompt, history=history)
        ai_message = message_ai["content"]
        print('*' * 20)
        print(ai_message)
        print('*' * 20)
        res[k]['ai_judge'] = ai_message

    with open(res_path, 'w') as f:
        json.dump(res, f, indent=4)


def generate_human_review():
    res_path = Path('cache') / 'NeurIPS2022.json'
    assert res_path.exists()
    res = json.load(res_path.open())

    accepts = {k: v for k, v in res.items() if v['is_accepted'] == True}
    accepts_select_10 = random.sample(list(accepts.items()), 10)
    rejects = {k: v for k, v in res.items() if v['is_accepted'] == False}
    rejects_select_10 = random.sample(list(rejects.items()), 10)

    # generate dataframe with columns ['Human meta review', 'AI meta review', 'AI judge', "Xi's decision", "Chuan's decision"], leave the last two columns empty
    to_use = accepts_select_10 + rejects_select_10
    df = pd.DataFrame(columns=['Human meta review', 'AI meta review', 'AI judge', "Xi's decision", "Chuan's decision"])
    for paper in to_use:
        df_new = pd.DataFrame({'Human meta review': paper[1]['meta_review'],
                               'AI meta review': paper[1]['ai_meta_review'],
                               'AI judge': paper[1]['ai_judge'],
                               "Xi's decision": "",
                               "Chuan's decision": ""}, index=[paper[0]])
        df = pd.concat([df, df_new])
    print()

    # save to excel
    df.to_excel('cache/NeurIPS2022_judge.xlsx', index=False)
    print()


if __name__ == '__main__':
    # generate_review()
    ai_judge()
    generate_human_review()
