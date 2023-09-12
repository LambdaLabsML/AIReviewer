# The AI Explainer

## Introduction
From the last article "Exploring AI's Role in Summarizing Scientific Reviews". We show some findings about how GPT is able to generate useful review summaries and "better than chance" accept/reject recommandations but also find GPT has a strong hesitancy to advise rejection. Several guessed reasons were investigated that may affect the AI's directive results, such as directive guidance "strictness" and indirective "certainty". We found the AI's different performance compared to human meta reviewers. We use accuracy as the metric, KL divergence and histogram comparison to illustratet the discrepancy between AI and Human Reviewers. There is one aspect missing, which is what causes their different decisions. To find this out, it is important to understand what aspects do AI and Human agree/disagree. We thus decide to adopt a `AI explainer` to generate explanations in terms of both the similarities and differences araised from the AI and Human meta reviews.

However, why AI has different views as Human still remains a problem. For example, in what aspects do human and AI act similarly or differently. We wonder whether we can use another AI to explain which makes the AI's decision. We hope this study can help us to understand the AI's decision and make the AI's decision more transparent. Also, we will test whether AI can directly output structured information, for example, in the statisticall perspective, what aspects mostly do AI and Human agree/disagree. 

This kind of task would be very challenging in real life, which requires a lot of human efforts to read and summarize the reviews in order to give the results. GPT is an ideal option to be selected for this task consider its outstanding performance in summarization. However, it is not easy to directly generate structured information from GPT, there are several challenges: 1. Summarizing from 100 reviews is a long input, which may not be directly taken as input in a one run. 2. If the context can be kept, whether GPT can generate statistical structured output, such as the frequency of most discussed aspects, remains unkown.

The experiments will conduct on the same dataset as before, which are 100 NeurIPS 2020 papers and their meta reviews, where 50 papers are accepted and 50 papers are rejected. The AI explainer will be based on OpenAI GPT-3.5-Turbo-16k, leveraging its high capacity for taking long inputs. The following sections will introduce the design of the AI explainer and the findings.

## AI explainer

### **Distill the keypoints**

According to OpenAI's best practice on [Tactic: Instruct the model to work out its own solution before rushing to a conclusion ](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-instruct-the-model-to-work-out-its-own-solution-before-rushing-to-a-conclusion), our first attempt is to first generate each paper's summary on similarities and difference about the AI and Human reviews. Specifically, we use a simple prompt to ask an OpenAI GPT (`GPT`) to write up to 3 aspects they agree/disagree, as shown below:

> Please act as an impartial explainer and evaluate the similarity and difference of the responses provided by a human meta reviewer (a) and AI reviewer (b) to a submitted paper. Avoid any position, length and order biases to influence your evaluation.
>
> List up to 3 aspects they agree/disagree, focusing on their explanation but not the" difference of their final recommendation or confidence. The aspects ([Aspect]) can be 'Novelty', 'Soundness', 'Presentation', 'Contribution', 'Related Work', 'Reproducibility', 'Ethics', 'Broader Impact', 'Correctness', 'Clarity', 'Strengths', 'Weaknesses', 'Relation to Prior Work', 'Additional Feedback', 'Questions for the Authors', or any other aspects you can give.
>
> Follow the format:
> Similarities:(Aspect 1): [explanation] 
> (Aspect 2): [explanation] 
> (Aspect 3): [explanation] 
> Differences:
> (Aspect 1): [explanation] 
> (Aspect 2): [explanation] 
> (Aspect 3): [explanation] 



#### Results

A sample results for a paper review explanation is as follows:

> Similarities:
> (Novelty): Both reviewers acknowledge the novelty of the paper's approach to solving statistical inverse problems using stochastic gradient descent.
> (Soundness): Both reviewers agree on the technical strength of the paper, with clear examples and mathematical exposition.
> (Relevance): Both reviewers express concern about the paper's relevance to a broader machine learning audience and suggest that the authors emphasize this more in their revision.
> Differences:
> (Presentation): The human reviewer suggests that the authors mention machine learning applications of their method early in the paper to pique reader interest, while the AI reviewer does not make this specific suggestion.
> (Strengths and Weaknesses): The AI reviewer mentions concerns about the completeness and consistency of the experimental results, which the human reviewer does not mention.
> (Confidence): The human reviewer is certain about their recommendation to accept the paper, while the AI reviewer is less certain due to the concerns raised by the reviewers and the need for further clarification on the limitations and advantages of the proposed method.

We can see the format has constrained the AI to have structured outputs, which is the clear aspects highlighting the keypoints where the human and AI's review agree or differ. 



### Detailed aspect-level analysis

Follwing the above steps, we got paper-level reviews analysis, consisting structrued aspects. We then made a deeper analysis on the aspect dimension to dig for frequent appearing aspects, what human or AI really agree or disagree. The prompt used is:

> Summarize a collection of similarities of the aspect {aspect} of Human/AI reviewers on 100 papers: 
>
> {context}
>
> Explain what can you find worth mentioning from these information. Give discussion in a higher level, do not mention method/paper details. The output format should be point by point.



#### Results on aspects of similar viewpoints

An example of deeper aspect-level analysis on "Novelty" is:

> 1. Both human and AI reviewers consistently acknowledge the novelty of the papers in various fields, such as...
> 2. The reviewers appreciate the introduction of novel approaches, ...
> 3. In some cases, the reviewers specifically mention the importance of the ..
> 4. There are instances where the reviewers acknowledge the novelty of the proposed algorithms, models ...
>
> ...
>
> Overall, the collection of similarities highlights the consistent recognition of novelty in the majority of the reviewed papers, indicating the importance of introducing new approaches, methods, and frameworks in various fields of research.

By first conducting paper-level summarization, followed by an aspect-level summarization, we got detailed point-by-point explanation of the similarities between human and Ai reviews.



#### Results on aspects of different viewpoints

We found "Presentation" is the most disagreed part between human and AI, appearing 34 times. An example analysis is:

> 1. The human reviewers often provide specific suggestions for improving the presentation of the papers, such as mentioning machine learning applications early on, emphasizing computational efficiency, and improving writing and presentation to highlight contributions. The AI reviewers, on the other hand, do not make these specific suggestions.
> 2. The AI reviewers tend to comment on the readability and clarity of the papers more frequently than the human reviewers. They mention concerns about small figures, dense writing, difficulties in understanding certain figures, and the need for more explanations and definitions of key concepts.
> 3. The human reviewers sometimes criticize the papers for lacking depth, realism, real impact, or a well-motivated or well-described restriction. The AI reviewers, however, do not mention these aspects.
> 4. The AI reviewers often commend the papers for being well-written, well-organized, easy to follow, and providing clear explanations and visualizations. The human reviewers, on the other hand, do not frequently comment on these aspects.
> 5. The human reviewers occasionally point out specific issues with the presentation, such as poorly formatted references or mistakes in citation codes. The AI reviewers, however, do not mention these issues.
> 6. The human reviewers sometimes suggest major revisions or additional discussions, plots, or sections for improving the papers. The AI reviewers, on the other hand, do not often provide specific areas for improvement.

The result clearly compares how the human and AI holding their viewpoints in the "Presentation" aspect in reviewing papers. This is a summary from 100 papers which is a general high-level conclusive comparison.



### The statistical capacity of AI

An important part of a analysis should also consider the quantatitive part. We wonder whether AI can directly do this, such as give a brief summary of how many aspects does the human and AI both agree. This is important to find what aspects are more important. To this end, after collecting each paper's concise summary, we then ask the GPT to consider the frequency of most discussed aspects, which is a kind of structure information. The prompt is shown below:

> Summarize an analysis of differences of human/AI meta reviews:
> {The 100 analysis from the previous step}
> Find out the common aspects that human and AI reviewers agree on. If possible, give the estimated frequency of each aspect.

#### Results
**Similarities**

> - Novelty: This aspect is mentioned in 38 out of the 100 papers analyzed.
> - Soundness: This aspect is mentioned in 34 out of the 100 papers analyzed.
> - Weaknesses: This aspect is mentioned in 26 out of the 100 papers analyzed.
> - Presentation: This aspect is mentioned in 14 out of the 100 papers analyzed.
> - Strengths: This aspect is mentioned in 13 out of the 100 papers analyzed.
> - Additional Feedback: This aspect is mentioned in 9 out of the 100 papers analyzed.
> - Relevance: This aspect is mentioned in 2 out of the 100 papers analyzed.
> - Contribution: This aspect is mentioned in 2 out of the 100 papers analyzed.
> - Clarity: This aspect is mentioned in 2 out of the 100 papers analyzed.
> - Recommendation: This aspect is mentioned in 2 out of the 100 papers analyzed.
> - Practical Impact: This aspect is mentioned in 2 out of the 100 papers analyzed.
> - Reproducibility: This aspect is mentioned in 2 out of the 100 papers analyzed.
> - Related Work: This aspect is mentioned in 2 out of the 100 papers analyzed.

**Differences**
> 1. Confidence in the recommendation: This aspect appears in multiple papers, where the human reviewers express a higher level of confidence in their recommendation compared to the AI reviewers. (Frequency: High)
> 2. Presentation: The human reviewers and AI reviewers often have different opinions on the clarity and quality of the presentation of the paper. While the human reviewers may not comment on the presentation, the AI reviewers may mention concerns about the readability, clarity, or organization of the paper. (Frequency: Medium)
> 3. Reproducibility: The AI reviewers may raise concerns about the reproducibility of the results, such as the lack of implementation details, code availability, or experimental details. The human reviewers may not comment on this aspect. (Frequency: Medium)
> 4. Related Work: The AI reviewers may point out the lack of discussion or comparison with relevant work in the related work section, while the human reviewers may not mention this aspect. (Frequency: Medium)
> 5. Weaknesses: The AI reviewers often provide more detailed analysis of the weaknesses of the paper, including concerns raised by other reviewers, while the human reviewers may not explicitly mention these weaknesses. (Frequency: Medium)
> 6. Additional Feedback: The AI reviewers often provide more detailed feedback, summarizing the comments of other reviewers and suggesting specific areas for improvement, while the human reviewers may not provide such detailed feedback. (Frequency: Medium)
> 7. Strengths: The human reviewers may not explicitly mention the strengths of the paper, while the AI reviewers may highlight the technical solidity, originality, or potential impact of the work. (Frequency: Low)
> 8. Clarity: The AI reviewers may raise concerns about the clarity of the paper, such as difficulties in understanding certain aspects or lack of explanations and definitions, while the human reviewers may not comment on this aspect. (Frequency: Low)
> 9. Recommendation: The human reviewers and AI reviewers may have different final recommendations for the paper, with the human reviewers often recommending rejection and the AI reviewers recommending acceptance, albeit with less certainty. (Frequency: Low)



We found the AI cannot corre

## Conclusion
