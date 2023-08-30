# Exploring AI's Role in Summarizing Scientific Reviews

## Introduction

As AI continues to find its place in diverse areas, we were curious about its possible role in the scientific peer-review process. This investigation focuses on exploring whether a Large Language Model, such as OpenAI GPT, could offer potentially assist in helping human readers to digest scientific reviews. Specifically, our study takes human reviews as the input, and ask AI to craft a "meta review" which gives an accept/reject recommendation of paper followed by the confidence of the remmendation and an explanation.

Our findings are:

- GPT is able to generate useful review summaries and "better than chance" accept/reject recommandations.
- GPT has a strong hesitancy to advise rejection.
- Several attributes, such as "certainty" and "human raiting", can have directive impact on AI's recommendation.

### Background: Paper Reviews

While having an AI directly critique scientific content remains a challenging and complex endeavor, our curiosity lies in its ability to help consolidate diverse human opinions into a cohesive recommendation. In essence, we're exploring if AI can simulate the process where seasoned domain experts, such as area chairs, distill general reviews into decisive meta reviews. For clarity, our examination is a simplied version of the real-world process: we've set aside aspects like the rebuttal process and the paper's content, and there is no contraint such as overall acceptance rate. Nevertheless, this endeavor aims to illuminate if AI can effectively summarize varied human perspectives and suggest informed decisions. Through this lens, we also hope to discern any inherent biases in current LLMs, especially when tasked with [nuanced decisions](https://blog.neurips.cc/2021/12/08/the-neurips-2021-consistency-experiment/) like paper acceptances.

To get a clearer picture, we looked into data from the NeurIPS 2022 conference, using review information from [OpenReview.com](http://openreview.com/): each paper has 3 to 6 **general reviews**, along with a **meta review** that gives the final decision based on these reviews. These reviews will be used as the inputs and "ground truth" output of our study.

**General Review**

Usually, each paper has detailed general reviews from 3 to 6 different reviewers. These reviews include ratings for recommendation, confidence, soundness, and more. It's important to note that we only considered the feedback given during the first round of reviews and didn't look at any follow-up discussions. An example of a reviewer's review includes a "Summary" and some scores like "rating," "confidence," and "presentation":

> Summary: This paper studies a method of …
> …
>
> Rating: 8: Strong Accept
>
> Confidence: 3
>
> Code Of Conduct: Yes

**Meta Review**

The meta review gives the final decision, which can be "accept" or "reject." There's also a "confidence" rating that shows how sure the meta reviewer is about the decision, categorized as "Certain" or "Less Certain." Additionally, the meta reviewer provides an explanation or comment. Here's an example of a meta review:

> Recommendation: Accept
>
> Confidence: Certain
>
> All reviewers recommend accepting the paper. But …
>
> The paper will have greater influence if the final version can convince readers of its relevance to ML!

The following sections explain our experiments by crafting a AI meta reviewer followed by analysis of what elements contribute to the trend of AI decision.

## The Initial Attempt: AI as Meta Reviewer

The initial stage of our experiment involved deploying the AI as a 'Meta Reviewer', utilizing the General Reviews as input and generating formatted Meta Reviews as output.

Our AI's output was tailored to match the conference guidelines: it expressed its level of confidence in the decision (ranging from high to lower confidence), delivered a recommendation (either Accept or Reject), and substantiated the recommendation with an explanation.

### Approach

In the implementation of the AI Meta Reviewer, to efficiently handle the substantial feedback provided by various reviewers, we employed the **`gpt-3.5-turbo-16k`** model from OpenAI, combined with Langchain’s `StuffDocumentsChain` to facilitate the process. The guiding prompt provided to the AI was structured as follows:

> Please act as a meta reviewer to give the final metareview based on reviews from other reviewers. Feel free to express the possible opinions.
> [The Start of Human Reviews]
> …
> [The End of Human Reviews]
> The output format should be:
> "Recommendation: [Reject/Accept] Meta Review: [Your review]…"
> …

**Assessment:** Upon receiving the AI-generated result, evaluating its performance becomes crucial. A straightforward metric for gauging alignment between AI and human reviewers is the `Decision Accuracy`, which quantifies the degree of agreement. Additionally, we conducted separate analyses for papers recommended for acceptance and those for rejection, in order to identify potential trends.

### Findings

TODO: add dataset size so we don't mislead people o think we have analyzed the entire dataset.

Our "AI Meta Reviewer" reached a 67% accuracy rate. Impressively, it achieved 100% accuracy for accepted papers but struggled with only 34% accuracy for rejected ones. Upon closer examination, a noticeable variance between the average reviewer score of the AI and that of the human meta reviewer emerged, particularly in instances of paper rejections.

Histograms help visualize this. For accepted papers, AI and human rating distributions were similar. But for rejections, the paper rejected by human tended have higher scores. This discrepancy underscored the AI's hesitancy to advise rejection, an aspect that became central to our following investigations.

![In accepting papers, AI and Human Meta Reviewers have similar averaged paper ratings](cache/histogram_Accept_gen_gpt-3.5-turbo-16k.png) ![In rejecting papers, AI tend to reject papers with lower averaged ratings than human do.](cache/histogram_Reject_gen_gpt-3.5-turbo-16k.png)

## **The 'Certain' Directive: Harnessing AI's Confidence**

A standout observation from the first experiment was AI tends to be "less certain" about its decisions. To address this, we instructed the AI to be more decisive: we prompted the AI to be more confident in making its final decisions, without necessarily directing its choice of accept or reject.

TODO: a quick sentence to explain how such a "certain" directive is implemented (as a prompt)

The addition of this directive led to significant changes. The overall accuracy jumped to 72%, with rejected paper accuracy surging from 34% to 52%. However, the accuracy for accepted papers did witness a slight decline to 92%. A histogram comparison indicated that post-directive, the AI's average score distribution for rejected papers was closer to human judgments (avg score 4.18 vs. 4.8) than before (3.66 vs. 4.8). Even though the accuracy for accepted papers went down slightly, the gap between the average reviewers ratings related to AI and human reviewers actually narrowed from 0.27 to 0.09.

TODO: this "gap between the average reviewers ratings" is not a good metric to compare two distributions. You can try any of these [metrics](https://safjan.com/metrics-to-compare-histograms/#:~:text=There%20are%20other%20metrics%20such,to%20compare%20histograms%20as%20well.)

![histogram_Accept_gen_gpt-3.5-turbo-16k_strictness_0.9_confidence_Certain.png](cache/histogram_Accept_gen_gpt-3.5-turbo-16k_strictness_0.9_confidence_Certain.png) ![histogram_Reject_gen_gpt-3.5-turbo-16k_strictness_0.9_confidence_Certain.png](cache/histogram_Reject_gen_gpt-3.5-turbo-16k_strictness_0.9_confidence_Certain.png)

## Ratings vs. Plain Context: What Weighs Heavier?

The above highlights how the direction of an AI's decision-making can be influenced by the nuances of a given prompt. A related question that arises is: what elements within the reviews significantly steer the AI Reviewer's decisions? While the ratings provided by each reviewer might offer a snapshot of the general impression of papers, how would the AI fare without them? To assess this, we excluded the "ratings" of each reviews, leaving only the plain text as context for the AI, and replicated the experiment without pushing the AI toward certainty.

The overall accuracy declined to 59%, with rejection accuracy plummeting to 24%, clearly underscoring the importance of ratings. Nevertheless, even though the AI leans on ratings for peak performance, its dependency can be adjusted by tailoring specific prompts as directives during the evaluation process.

TODO: it is unclear what does the last sentence mean (start from "Nevertheless")

## Examples

TODO: A couple of examples of human meta review v.s. AI meta review, when they agree/disagree.

TODO: possibly link to the spread sheet for all the results.

## **Conclusion**

The experiments suggest the promising capabilities of LLMs as potential meta-reviewers. Yet, they also hint at the intricacies involved in aligning an AI's judgment with human insights, particularly in nuanced tasks like paper reviewing. While the LLM displayed commendable accuracy rates, notably when provided with the 'certainty' directive, its initial caution in rejecting papers indicates that reviewing isn't solely about quantitative metrics. The human perspective, shaped by years of experience, intuition, and a deep grasp of the scientific domain, is invaluable. Nevertheless, as we refine and learn from these experiments, there could be an emerging role for AI-assisted reviewing. In such a scenario, the unique strengths of human intuition and AI's data-driven approach might collaboratively contribute to even more refined review processes down the line.
