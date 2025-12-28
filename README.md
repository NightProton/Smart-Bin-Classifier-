**Smart Bin Classifier**


**1. Objective**

Build a system that can:

- Understand Amazon bin images

- Identify which products (ASINs) are present

- Match text queries like *"2 Dove Shampoo"* or *"1 B00ABCXYZ"*

- Retrieve the best bin image containing the requested item

- Provide a clean UI for real-time use

- Enable future deployment, monitoring, and retraining

The challenge was **large-scale ASIN variability**, messy images, and
multiple items per bin.

**2. Methodology**

The project followed a step-wise experimentation pipeline:

1.  **Data extraction & EDA**  
    Analyzed image availability, empty vs non-empty bins, unique ASIN
    distribution, and quantity patterns (from metadata).  
    This revealed a long-tail product distribution and bins typically
    containing 2--4 items"  
   

2.  **Model experimentation**  
    Tried three major model families:

    - Swin Transformer (supervised multi-label)

    - CLIP (contrastive T-I alignment)

    - SIGLIP (Google's next-gen CLIP replacement)

3.  **Progressive refinement**  
    After each model, evaluated accuracy and feasibility, then pivoted
    based on findings.

4.  **Final SIGLIP retrieval system**  
    Text and image embeddings power a cosine-similarity search engine
    used in the UI.

> **3. Architecture Overview**

1.  **Dataset Layer**  
    Consists of bin images + metadata JSONs (product name, ASIN,
    quantity).

2.  **Embedding Layer (SIGLIP)**  
    Using fine-tuned SIGLIP, image and text embeddings are aligned in a
    shared vector space.

3.  **Search Engine**

    - Precompute embeddings of all images

    - For a query (product name + quantity), compute text embedding

    - Compare using cosine similarity

    - Retrieve the highest-scoring bin

4.  **User Interface (Gradio)**  
    Users select a product row, enter quantity, and see the best
    matching bin image.  
    Duplicate products removed, warning if requested quantity \>
    available.

**4. Model Selection Rationale**

**Attempt 1 --- Swin Transformer**

Supervised multi-label classification

> Could not scale beyond \~20 ASIN classes

Real dataset had hundreds of ASINs

Failed at generalization  
→ **Rejected due to class limit & scalability issues**

**Attempt 2 --- CLIP**

> Strong augmentation & text variants used

Even after fine-tuning:

Very poor top-1 accuracy

Weak top-10 retrieval

Low ASIN recall

CLIP struggled with cluttered bin images and very specific ASIN tokens  
→ **Rejected due to low retrieval performance**

**Attempt 3 --- SIGLIP (Final Choice)**

Much better contrastive alignment

Stronger text--image representation

Handles product-like tokens cleanly

Stable convergence

Gave consistent high-quality matches in the UI  
→ **Selected as final model**

**5. MLOps Considerations**

**Deployment**

- Deployed on AWS EC2 INSTANCE

**Monitoring**

Monitor:

- Retrieval latency

- Incorrect matches (feedback loop)

- Frequency of unseen ASINs

- Embedding drift (new product packaging/images)

**Retraining Strategy**

Triggered when:

- New dataset arrives

- New ASINs appear

- Packaging changes cause embedding drift

- UI logs show increasing mismatches

Retraining steps:

1.  Append new images + metadata

2.  Fine-tune SIGLIP for a few epochs

3.  Rebuild embedding cache

4.  Redeploy updated model

**6.Deployment on AWS  
** ![](media/image1.png){width="6.268055555555556in"
height="2.8673611111111112in"}

![A screenshot of a computer AI-generated content may be
incorrect.](media/image2.png){width="6.268055555555556in"
height="3.652083333333333in"}

![A close up of a label AI-generated content may be
incorrect.](media/image3.png){width="6.268055555555556in"
height="3.6979166666666665in"}![A screenshot of a
computer](media/image4.png){width="6.268055555555556in"
height="3.8868055555555556in"}

**7. Final Summary**

The Smart Bin Classifier evolved from limited classification-based
attempts into a powerful **SIGLIP-backed retrieval system** that:

- Works for hundreds of ASINs

- Handles complex multi-item bins

- Matches both product name & quantity

- Delivers fast, intuitive results via UI

- Is ready for real-world scaling and MLOps integration

This final system is accurate, scalable, and significantly more
practical than earlier Swin or CLIP approaches
