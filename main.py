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
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

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


def generate_meta_from_pdf(model_name='gpt-4'):
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


def generate_meta_from_reviews(model_name='gpt-3.5-turbo-16k', strictness=None, confidence=None, score=True):
    """
    Use GPT to generate the summary (meta review) from other human reviews
    :param model_name: OpenAI model name
    :param strictness: float number between 0 and 1, higher is stricter
    :param confidence: 'Certain' or 'Less Certain'
    :param score: whether to use include the score of reviewers, for ablation study
    :return: generated meta review
    """

    src_path = Path('cache') / 'raw.json'
    dst_path = Path('cache') / 'gen_{}.json'.format(model_name)

    assert src_path.exists()
    res = json.load(src_path.open())

    # filtering existed
    accept_folder = list((Path('cache') / 'accepted').glob('*.pdf'))
    reject_folder = list((Path('cache') / 'rejected').glob('*.pdf'))
    names_existed = [x.stem for x in accept_folder + reject_folder]
    res = {k: v for k, v in res.items() if k in names_existed}

    prompt_template = "Please act as a meta reviewer to give the final metareview based on reviews from other reviewers."

    if strictness != None:
        assert 0 <= strictness <= 1, "Strictness should be a float number between 0 and 1, higher is stricter."
        strictness_words = (
            f"The strictness for this conference is {strictness}, which is a float number between 0 and 1, higher is stricter."
            "You should tend to reject a paper if the strictness is higher, and tend to accept a paper if the strictness is lower.")
        prompt_template += strictness_words
        dst_path = dst_path.parent / (dst_path.stem + '_strictness_{}.json'.format(strictness))

    if confidence is not None and confidence.lower() == 'certain':
        confidence_words = f"""Your confidence for this conference is "{confidence}", you should be more confident to give a final decision."""
        prompt_template += confidence_words
        dst_path = dst_path.parent / (dst_path.stem + '_confidence_{}.json'.format(confidence))

    if not score:
        dst_path = dst_path.parent / (dst_path.stem + '_NoScore.json')

    prompt_template += """
      Feel free to express the possible opinions.
      
        [The Start of Human Reviews]
        "{text}"
        [The End of Human Reviews]
        
      """

    prompt_template += f"""
    The output format should be:
      "Recommendation: [Reject/Accept]
      Confidence:{"[Certain/Less Certain]" if confidence is None else "Certain"}
      Meta Review: [Your review]"
      
      (Note there is no "weak" or "borderline" recommendation.)
      """

    prompt = PromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(temperature=0, model_name=model_name)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )

    for paper_name, v in tqdm(res.items(), total=len(res.keys())):
        reviews = v['reviews']
        if not score:
            new_reviews = []
            for r in reviews:
                new_reviews.append(
                    r.split('Rating:')[0].strip() + '\nConfidence:' + r.split('Confidence:')[1].strip()
                )
            reviews = new_reviews
        text = '\n'.join(reviews)
        docs = [Document(page_content=text, metadata={})]
        summary = stuff_chain.run(docs)
        res[paper_name][f'ai_sum_meta'] = summary
        print(summary)

        with open(dst_path, 'w') as f:
            json.dump(res, f, indent=4)

    print('Saved to {}'.format(dst_path.name))


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


def ai_judge(col):
    """
    Using GPT to judge whether the generated meta review is similar to the real human meta review
    :param col: column name of the generated meta review
    :return: None
    """
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

    prompt_template = (
        "[User Question]\n{question}\n\n[The Start of Human Meta Review]\n{answer_a}\n[The End of Human Meta Review]\n\n[The Start of AI Meta Review]\n{answer_b}\n[The End of AI Meta Review]\n"
        "Give a similarity score from 0 to 10: final recommendations (acceptance or rejection) is the most important factor, weighting 5 to 7. But confidence, content and explanation are less important, weighting 1 to 2. "
        "For example, the score should be over 5 if the two have the same recommendation, but not over 5 if they are different. Do not be too strict."
    )

    for k, v in res.items():
        user_prompt = prompt_template.format(
            question="What is the estimated similarity score of the AI reviewer's review to the human reviewer's review?",
            answer_a=v['meta_review'], answer_b=v[col])

        # if not v['ai_meta_review'].startswith("The paper demonstrates strong theoretical"):
        #     continue

        message_ai = _chatgpt(sys_prompt=sys_prompt, user_prompt=user_prompt, history=history)
        ai_message = message_ai["content"]
        print('*' * 20)
        print(ai_message)
        print('*' * 20)
        res[k]['judge_{}'.format(col)] = ai_message

    with open(res_path, 'w') as f:
        json.dump(res, f, indent=4)


def analysis(name):
    """
    Summarize the generated AI reviews with the real meta review and user study opinions
    :return:
    """

    src_path = Path('cache') / name
    dst_path = Path('cache') / 'analysis_{}.xlsx'.format(src_path.stem)

    assert src_path.exists()
    res = json.load(src_path.open())

    raw = json.load(Path('cache/raw.json').open())

    columns = ['Paper', 'Human meta review', 'Human meta decision',
               'AI meta', 'AI meta decision', 'AI judge',
               "R1", "R2", "R3", "R4", "R5", "R6"]
    df = pd.DataFrame(columns=columns)

    human_meta_decisions = []
    ai_meta_decisions = []

    human_avg_scores = []

    max_nb = 0
    for paper in res.items():
        ai_meta_decision = paper[1]['ai_sum_meta'].strip().split('\n')[0].split(': ')[1].strip()
        ai_meta_decision = 'Accept' if 'accept' in ai_meta_decision.lower() else ai_meta_decision
        ai_meta_decision = 'Reject' if 'reject' in ai_meta_decision.lower() else ai_meta_decision
        assert ai_meta_decision in ['Accept', 'Reject']
        human_meta_decision = paper[1]['meta_review'].strip().split('\n')[0].split(': ')[1].strip()
        human_meta_decision = 'Accept' if 'accept' in human_meta_decision.lower() else human_meta_decision
        human_meta_decision = 'Reject' if 'reject' in human_meta_decision.lower() else human_meta_decision
        assert human_meta_decision in ['Accept', 'Reject']

        paper_info = \
            (f"Title: {paper[0]}\n"
             f"Paper ID: {paper[1]['paper_id']}\n"
             f"Paper URL: {raw[paper[0]]['pub_url']}\n"
             f"PDF URL: {paper[1]['pdf_link']}\n"
             f"Avg_rating: {raw[paper[0]]['rating_avg']}\n"
             f"Avg_confidence: {raw[paper[0]]['confidence_avg']}\n"
             f"Avg_soundness: {raw[paper[0]]['soundness_avg']}\n"
             f"Avg_presentation: {raw[paper[0]]['presentation_avg']}\n"
             f"Avg_contribution: {raw[paper[0]]['contribution_avg']}\n")

        if len(paper[1]['reviews']) > max_nb:
            max_nb = len(paper[1]['reviews'])
        df_new = pd.DataFrame(
            {
                'Paper': paper_info,
                'Human meta review': paper[1]['meta_review'],
                'Human meta decision': human_meta_decision,
                'AI meta': paper[1]['ai_sum_meta'],
                'AI meta decision': ai_meta_decision,
                # 'AI judge': paper[1]['judge_ai_sum_meta'],
                "R1": paper[1]['reviews'][0] if len(paper[1]['reviews']) > 0 else "",
                "R2": paper[1]['reviews'][1] if len(paper[1]['reviews']) > 1 else "",
                "R3": paper[1]['reviews'][2] if len(paper[1]['reviews']) > 2 else "",
                "R4": paper[1]['reviews'][3] if len(paper[1]['reviews']) > 3 else "",
                "R5": paper[1]['reviews'][4] if len(paper[1]['reviews']) > 4 else "",
                "R6": paper[1]['reviews'][5] if len(paper[1]['reviews']) > 5 else "",
            },
            index=[paper[0]])
        df = pd.concat([df, df_new])
        human_meta_decisions.append(human_meta_decision)
        human_avg_scores.append(raw[paper[0]]['rating_avg'])
        ai_meta_decisions.append(ai_meta_decision)

    df.to_excel(dst_path, index=False)

    # compute the accuracy
    acc = sum([1 if x == y else 0 for x, y in zip(human_meta_decisions, ai_meta_decisions)]) / len(human_meta_decisions)
    # acc_accept considers the case that both human and AI accept the paper
    accept_nb = sum([1 if x == 'Accept' else 0 for x in human_meta_decisions])
    reject_nb = sum([1 if x == 'Reject' else 0 for x in human_meta_decisions])
    acc_accept = sum(
        [1 if x == y == 'Accept' else 0 for x, y in zip(human_meta_decisions, ai_meta_decisions)]) / accept_nb
    acc_reject = sum(
        [1 if x == y == 'Reject' else 0 for x, y in zip(human_meta_decisions, ai_meta_decisions)]) / reject_nb
    print('Accuracy: {}'.format(acc))
    print('Accuracy of Accept: {}'.format(acc_accept))
    print('Accuracy of Reject: {}'.format(acc_reject))

    # make a histogram, ranging score from 0 to 10, check whether the AI judge is similar to the human judge
    human_score2decision = zip(human_avg_scores, human_meta_decisions)
    human_score2decision = sorted(human_score2decision, key=lambda x: x[0])
    human_score2decision = list(human_score2decision)
    ai_score2decision = zip(human_avg_scores, ai_meta_decisions)
    ai_score2decision = sorted(ai_score2decision, key=lambda x: x[0])
    ai_score2decision = list(ai_score2decision)
    import matplotlib.pyplot as plt

    decisions = ['Accept', 'Reject']
    colors = ['tab:blue', 'tab:orange']

    for decision, color in zip(decisions, colors):
        human_scores = np.array([x[0] for x in human_score2decision if x[1] == decision])
        ai_scores = np.array([x[0] for x in ai_score2decision if x[1] == decision])

        # Calculate mean and confidence interval for each group
        ai_mean = np.mean(ai_scores)
        ai_ci = stats.norm.interval(0.95, loc=ai_mean, scale=stats.sem(ai_scores))

        human_mean = np.mean(human_scores)
        human_ci = stats.norm.interval(0.95, loc=human_mean, scale=stats.sem(human_scores))

        # Create the histogram with bins of width 0.3
        bins = np.arange(0, 11.3, 0.3)  # Bins from 0.0 to 11.0 with 0.3 intervals

        ai_density = plt.hist(ai_scores, bins=bins, alpha=0.3, color=color, label='AI', density=False)
        human_density = plt.hist(human_scores, bins=bins, alpha=0.3, color='gray', label='Human', density=False,
                                 hatch='//')

        plt.axvline(ai_mean, color=color, linestyle='dashed', linewidth=1)
        plt.axvline(human_mean, color='gray', linestyle='dashed', linewidth=1)

        plt.text(ai_mean - 0.2, 0.8 * ai_density[0].max(), f'{ai_mean:.2f}    \n[{ai_ci[0]:.2f}, {ai_ci[1]:.2f}]    ',
                 color=color, ha='right')
        plt.text(human_mean + 0.2, 0.8 * ai_density[0].max(),
                 f'{human_mean:.1f}    \n[{human_ci[0]:.2f}, {human_ci[1]:.2f}]    ', color='gray')

        plt.legend(loc='upper right')
        plt.title(
            f'Histogram of {decision}ed Papers Distribution' + "" if "confidence_Certain" not in name else " (Confidence: Certain)")
        plt.xlabel('Reviewer Average Score')
        plt.ylabel('Paper Number')

        # Save and display the histogram
        plt.savefig(dst_path.parent / f'histogram_{decision}_{src_path.stem}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print()


if __name__ == '__main__':
    generate_meta_from_reviews(model_name='gpt-3.5-turbo-16k')
    # generate_meta_from_reviews(model_name='gpt-3.5-turbo-16k', confidence='Certain')
    ai_judge(col='ai_sum_meta')
    analysis(name='gen_gpt-3.5-turbo-16k.json')
