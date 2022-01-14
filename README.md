<<<<<<< HEAD
# **Supervised and Unsupervised Applications of Natural Language Processing on Free Text towards Tackling Scams**
## MATH5872M Dissertation in Data Science and Analytics
## University of Leeds, September 2020 
### Author: Zeya Lwin Tun
### Supervisors: Dr Daniel Birks and Dr Leonid Bogachev

This repository contains reproducible code written in Python 3.7.7 as part of my Master's dissertation at University of Leeds. My dissertation is titled **Supervised and Unsupervised Applications of Natural Language Processing on Free Text towards Tackling Scams**. 

### Abstract

Scams are becoming increasingly prevalent and a cause of concern globally. In Singapore, scams made up 27.0\% of overall crimes in 2019, compared to 17.5\% in 2018. In the first half of 2020, a total of S\$82 million was cheated from victims, almost twice the amount in the same period of 2019. Besides immediate financial losses, victims of scams also suffer from longer-term emotional and psychological effects. Despite efforts by authorities, victims continue to fall prey, owing partly to more sophisticated means used by scammers. There is therefore a strong need to increase the understanding of scams and how they can be prevented. 

The research in this dissertation aims to achieve this by drawing lessons from others’ scam experiences shared on `Scam Alert’, a Singapore-based website aimed at promoting scam awareness. More specifically, this research harnesses the hidden potential of free text in these scam reports using machine learning and Natural Language Processing (NLP) methods towards the following research goals: finding scam reports with similar modus operandi, extracting common characteristics from similar scam reports and classifying scam reports.

In pursuit of these research goals, this dissertation presents novel applications of machine learning and NLP on free text in scam reports in two areas: supervised and unsupervised. In supervised application, deep learning techniques are used for multi-class classification of scams. Given class imbalance in the data, text augmentation techniques and the Synthetic Minority Over-sampling Technique are explored. In addition, the efficacy of using pre-trained Global Vectors (GloVe) word embeddings is examined. Results show that the Long Short-Term Memory model trained without GloVe word embeddings on a dataset balanced with text augmentation outperformed the rest. 

In unsupervised application, the concept of vector semantics is leveraged using doc2vec models to encode scam reports as document embeddings. To evaluate doc2vec models, a new framework known as normalised Similarity-Dissimilarity Quotient (SDQ) is introduced. Normalised SDQ assesses a doc2vec model's ability to infer document embeddings that can recognise similar and dissimilar reports from sets of pre-identified scam reports. Using normalised SDQ, the most optimal doc2vec model is found to be the model trained with 150 epochs, 50-dimensional embeddings and the Distributed Memory Model of Paragraph Vector algorithm. 

Findings from both supervised and unsupervised applications lay the foundation for the development of tools towards achieving the research goals. It is envisioned that these tools will sharpen the sense-making capabilities of law enforcement authorities in better understanding how scams operate and in identifying intervention points where scams can be disrupted. With such insights, public education and engagement efforts can be more tailored and effective. They also boost quality of criminal investigations against scammers, which in turn serves as a deterrent and helps toughen the stance against scams. Additionally, these tools can nurture a stronger sense of awareness and guardianship within the society. After all, a discerning public is the strongest defence against scams.

### Table of Contents for iPython Notebooks

The .ipynb notebooks are organised as follow:

![](ipython_notebooks_content.png)
=======
# Supporting Crime Script Analyses of Scams with Natural Language Processing 
### Authors: Zeya Lwin Tun, Daniel Birks

This repository contains reproducible code written in Python 3.7.7 for this research paper.

### Abstract

In recent years, internet connectivity and the ubiquitous use of digital devices have afforded a landscape of expanding opportunity for offenders. Scams, here defined as schemes designed to deceive individuals into giving away money or personal information generally through the use of the Internet or some other communication medium such as a phone call or text message, are commonplace globally. Their impacts on victims have been shown to encompass social, psychological, emotional and economic harms. Consequently, there is a strong rationale to enhance our understanding of scams in order to devise ways in which they can be disrupted. 

Crime scripting is an analytical approach which seeks to characterise the processes which underpin crime events in sufficient detail to aid in the identification of potential points of intervention where crimes might be prevented. Crime scripting typically involves the analysis of large quantities of data describing crime events from which such key processes are drawn. While this approach has been shown to be effective in systematically understanding a diverse range of offences, it is highly resource intensive and as a result not particularly commonplace. 

In this paper, we explore how Natural Language Processing (NLP) methods might be applied to extract insights from unstructured textual data with the aim of supporting crime script analyses. To illustrate this approach, we apply NLP to a public dataset of victims' stories of scams perpetrated in Singapore. Applying a range of methods, we demonstrate approaches capable of automatically isolating scams with similar modus operandi, extracting key elements from their descriptions, and work towards identifying the temporal ordering of these elements in ways that characterise how a particular scam operates. Our exploratory results provide a further example of how NLP methods may in the future enable crime preventers to better harness unstructured free text data to better understand crime problems and in turn devise strategies that seek to reduce them. 

### Keywords
Scams, Crime, Policing, Crime Script Analysis, Unstructured Data, Natural Language Processing, Term Frequency-Inverse Document Frequency, Doc2Vec
>>>>>>> 21815d2 (latest updates)

### Acknowledgements

National Crime Prevention Council, Singapore. 
