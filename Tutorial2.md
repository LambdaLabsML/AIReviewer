# The AI Explainer

## Introduction
From the last article "Exploring AI's Role in Summarizing Scientific Reviews". We show some findings about how GPT is able to generate useful review summaries and "better than chance" accept/reject recommandations but also find GPT has a strong hesitancy to advise rejection. Several guessed reasons were investigated that may affect the AI's directive results, such as directive guidance "strictness" and indirective "certainty".

However, why AI has different views as Human still remains a problem. For example, in what aspects do human and AI act similarly or differently. We wonder whether we can use another AI to explain which makes the AI's decision. We hope this study can help us to understand the AI's decision and make the AI's decision more transparent. Also, we will test whether AI can directly output structured information, for example, in the statisticall perspective, what aspects mostly do AI and Human agree/disagree. This kind of task would be very challenging in real life, which requires a lot of human efforts to read and summarize the reviews in order to give the results.

The experiments will conduct on the same dataset as before, which are 100 NeurIPS 2020 papers and their meta reviews, where 50 papers are accepted and 50 papers are rejected. The AI explainer will be based on OpenAI GPT-3.5-Turbo-16k, leveraging its high capacity for taking long inputs. The following sections will introduce the design of the AI explainer and the findings.

## Review explainer

### Review the previous findings
In the last article, we found the AI's different performance compared to human meta reviewerx. We use accuracy as the metric, KL divergence and histogram comparison to illustratet the discrepancy between AI and Human Reviewers. There is one aspect missing, which is what causes their different decisions. To find this out, it is important to understand what aspects do AI and Human agree/disagree. We thus decide to adopt a `AI explainer` to generate explanations in terms of both the similarities and differences araised from the AI and Human meta reviews.

**Challenges for the AI Explainer**
GPT is an ideal option to be selected for this task consider its outstanding performance in summarization. However, it is not easy to directly generate structured information from GPT, there are several challenges: 1. Summarizing from 100 reviews is a long input, which may not be directly taken as input in a one run. 2. If the context can be kept, whether GPT can generate statistical structured output, such as the frequency of most discussed aspects, remains unkown.

**Compressing the context**
According to OpenAI's best practice on [Tactic: Instruct the model to work out its own solution before rushing to a conclusion ](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-instruct-the-model-to-work-out-its-own-solution-before-rushing-to-a-conclusion), our first attempt is to first generate each paper's summary including the aspects comparing the AI and the Human meta review for the accepted and rejected papers, respectively. We first use a simple prompt to ask an OpenAI GPT (`GPT`) to write Up to 3 aspects they agree/disagree, as shown below:


After collecting each paper's concise summary, we then do further summarization to generate the final summary. This time, we ask the GPT to consider the frequency of most discussed aspects, which is a kind of structure information. The prompt is shown below:
> Summarize an analysis of differences of human/AI meta reviews:
> {The 100 analysis from the previous step}
> Find out the common aspects that human and AI reviewers agree on. If possible, give the estimated frequency of each aspect.

### Results
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

### The findings

## Conclusion
