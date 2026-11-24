"""All prompt templates used by SliceLens.

These are transcribed *verbatim* from the appendix of the paper
"SliceLens: Fine-Grained and Grounded Error Slice Discovery for Multi-Instance
Vision Tasks" (arXiv:2512.24592v1, Apple, 2025):

    * KNOWLEDGE_DRIVEN_SYSTEM_PROMPT   -> Figure 8
    * GROUNDED_CAPTION_SYSTEM_PROMPT   -> Figure 9
    * ATTRIBUTE_EXTRACTION_PROMPT      -> Figure 10, Step 1
    * KEYWORD_CLUSTERING_PROMPT        -> Figure 10, Step 2
    * QUERY_REFINEMENT_PROMPT          -> Figure 11
    * FESD_DETECTION_USER_PROMPT       -> Figure 12
    * FESD_SEGMENTATION_USER_PROMPT    -> Figure 13
    * WATERBIRDS / CELEBA / NICO       -> Section A.2
    * SEMANTIC_RELEVANCE_PROMPT        -> Section A.3
    * verification templates           -> Section 3.2.1

Placeholders use ``str.format`` style: ``{label}``, ``{attribute}``,
``{task_description}``, ``{query}``, ``{xmin}`` etc.
"""

# ---------------------------------------------------------------------------
# A.1.1  Knowledge-driven hypothesis generation  (Figure 8)
# ---------------------------------------------------------------------------
KNOWLEDGE_DRIVEN_SYSTEM_PROMPT = """\
You are an expert in computer vision failure analysis. A user will provide a \
task description (e.g., "a cat body detection"). Your goal is to identify and \
list all possible reasons why a computer vision model might fail on this task. \
In your analysis, consider a comprehensive set of factors, including but NOT \
LIMITED TO:
- Object Attributes
- Background Descriptions
- Image Qualities
- Environmental Conditions
- Limitations in handling edge cases or extreme conditions
- Camera capturing conditions
- Any additional domain-specific challenges

For the given task description, generate a structured and detailed list of \
failure hypotheses. Your output must include:
- Comprehensive different factors (e.g., Object Attributes, Image Qualities, etc.).
- Under each factor, list comprehensive failure patterns.
- For each failure pattern, include:
    - A title that summarizes the potential failure reason.
    - A description that explains in detail why this might be an issue.
    - A list of prompts to further explore the hypothesis. Each prompt must include:
        * A prompt field with a specific query or cluster description.
        * A type field, where the value is either "cluster" or "search". Use \
"cluster" if the prompt provides a broad description with many possible \
attributes for further investigation (e.g., "cat colors" or "cat types"). Use \
"search" if the prompt can be directly used to retrieve relevant images (e.g., \
"a white cat").
- IMPORTANT: Ensure that your final output does not contain any duplicate \
failure patterns or prompts. Every entry must be unique and directly relevant \
to image retrieval or image clustering.

Please think through at least three iterations internally to ensure \
completeness, accuracy, and uniqueness. Then, output your final results in a \
valid JSON object formatted exactly as follows:
{
    "title": "Possible failure reasons for [task description] model",
    "hypothesis": {
        "Object Attributes": [
            {
                "title": "[Short title for failure pattern 1]",
                "description": "[Detailed explanation]",
                "prompts": [
                    {
                        "prompt": "[related query or cluster description term]",
                        "type": "[cluster or search]"
                    }
                ]
            }
        ],
        "Background Descriptions": [ ],
        "...": "(other categories as comprehensively as possible)"
    }
}"""

# ---------------------------------------------------------------------------
# A.2  Task / dataset specific *user* prompts
# ---------------------------------------------------------------------------

# Figure 12 -- FeSD object detection
FESD_DETECTION_USER_PROMPT = """\
An object detection model designed to detect {label}. Specifically, consider \
and describe possible scenarios for the following types of errors:

False Positive: In what situations might other objects be mistakenly detected \
as {label}?

False Negative: In what situations might {label} fail to be detected by the \
model?

Misclassification: In what situations might {label} be incorrectly detected as \
another object class?"""

# Figure 13 -- FeSD instance segmentation
FESD_SEGMENTATION_USER_PROMPT = """\
An Instance Segmentation model designed to detect and segment {label}. Think \
systematically from the following perspectives, and describe possible scenarios \
for each type of error below:

False Positive: In what situations might other objects be mistakenly segmented \
as {label}?

False Negative: In what situations might {label} fail to be segmented?

Misclassification: In what situations might {label} be incorrectly segmented as \
another object class?

Boundary Error: When might the segmentation mask boundary deviate from the true \
object contour (e.g., occlusion, uneven lighting, blurry edges)?

Incomplete Mask: When might the model segment only part of the {label} instead \
of the entire object?

Over-segmentation: When might the model mistakenly split one {label} instance \
into multiple segments?

Under-segmentation: When might the model merge multiple nearby {label} \
instances into a single mask?"""

# A.2.1 -- Waterbirds
WATERBIRDS_USER_PROMPT = (
    "A model that classifies whether a bird is a water bird or a land bird. "
    "The dataset is constructed by cropping out birds from photos and "
    "transferring them onto different backgrounds."
)

# A.2.2 -- CelebA
CELEBA_USER_PROMPT = (
    "A model that detects whether a person in an image has blonde hair, "
    "considering various attributes of the person."
)

# A.2.3 -- NICO++
NICO_USER_PROMPT = (
    "A model is used for multi-label classification, classifying images into "
    "one of six categories: mammals, birds, plants, airways, waterways, and "
    "landways. The dataset consists of real-world images of these categories "
    "appearing in six different contexts (dim lighting, outdoor, grass, rock, "
    "autumn, and water)."
)

# ---------------------------------------------------------------------------
# A.1.2  Data-driven hypothesis generation
# ---------------------------------------------------------------------------

# Figure 9 -- Grounded attribute-aware caption extraction (VLM system prompt)
GROUNDED_CAPTION_SYSTEM_PROMPT = (
    "You are an AI model designed to analyze images and provide detailed "
    "captions for a specific region relevant to the {attribute}. The specified "
    "region is defined by bounding box coordinates in the format "
    "[xmin, ymin, xmax, ymax]."
)

# A grounded user message that pins the caption to the region. The paper grounds
# the caption to a bounding box; we make that explicit in the user turn.
GROUNDED_CAPTION_USER_PROMPT = (
    "The region is [{xmin}, {ymin}, {xmax}, {ymax}]. Provide a detailed caption "
    "describing this region with respect to the {attribute}."
)

# Figure 10, Step 1 -- attribute value / keyword extraction
ATTRIBUTE_EXTRACTION_PROMPT = """\
You will be given a description of the {attribute}. Your job is to determine \
the most relevant {attribute} based on the provided description. Don't return \
anything except the result.
Example Input:
Attribute: activity
Caption: The image depicts a person performing a handstand on a purple mat in a \
spacious, well-lit room with white walls and a high ceiling.
Expected Output:
handstand
If the description is irrelevant to {attribute}, please output 'none'.
Now, process the provided list:
Attribute: {attribute}
Caption: {caption}"""

# Figure 10, Step 2 -- keyword refinement and clustering
KEYWORD_CLUSTERING_PROMPT = """\
You are a helpful text clustering assistant. You will be given a list of \
keyword text. Your job is to refine and categorize the list to clusters. Your \
goal is to process the provided keyword list to:
- Eliminate Duplicate Semantics: Identify and remove keywords that represent \
similar meaning, even if expressed differently (e.g., 'car' and 'automobile').
- Eliminate irrelevant keywords: Identify and remove keywords that are not \
related to {attribute}.
- Group into Categories: Organize keywords into distinct, non-overlapping \
categories based on their meaning and context.
- Ensure Clarity: Avoid overly broad or overly narrow categories. Each keyword \
should fit logically within its assigned category.
Example Input:
['car', 'automobile', 'vehicle', 'bike', 'bicycle', 'motorcycle', 'transport']
Expected Output (JSON format):
{{"transport":["car", "bike", "motorcycle"]}}
Please think and improve your results at least four turns. And then eliminate \
duplicate semantics in one more turn. Now, process the provided keyword list \
and output the final results in JSON format at the end with unique, \
non-overlapping semantics.
Keyword list: {keywords}"""

# Figure 11 -- refinement prompt for grounded query phrases
QUERY_REFINEMENT_PROMPT = """\
You will be given a JSON list of attributes with values or cluster names with \
categories. These cluster names are summarized from descriptions related to the \
{attribute}. Your job is to compose a text to image retrieval prompt from each \
category and item in the list. You can start or end with general descriptions, \
for example "a photo of" etc. Please DO NOT add or modify the context and \
semantics. The output JSON format is:
{{"results": [prompt1, prompt2, ...]}}
Now process the list and generate prompts one by one for each items.
Input: {clusters}"""

# ---------------------------------------------------------------------------
# 3.2.1  Grounded candidate slice discovery -- verification templates
# ---------------------------------------------------------------------------
# These are fed to the VLM; the logits of the "yes"/"no" output tokens yield the
# slice confidence P(yes | q, r) of Equation (1).

POINT_VERIFICATION_PROMPT = (
    "If the region pointed to by point_2d: ({x}, {y}) matches the description "
    "{query}, please answer yes else no."
)

BBOX_VERIFICATION_PROMPT = (
    "If the region [{xmin}, {ymin}, {xmax}, {ymax}] matches the description "
    "{query}, please answer yes else no."
)

# Used when the region is provided as an actual image *crop* (with margin)
# rather than as text coordinates -- the right format for VLMs that are not
# grounding-tuned (e.g. Gemma), which cannot localise a text bbox.
CROP_VERIFICATION_PROMPT = (
    "The image shows an object outlined by the crop. If it matches the "
    "description {query}, please answer yes else no."
)

# ---------------------------------------------------------------------------
# A.3  Semantic relevance evaluator (used only for evaluation against GT slices)
# ---------------------------------------------------------------------------
SEMANTIC_RELEVANCE_PROMPT = """\
I have a description of an error pattern from a computer vision model, and a \
list of ground truth error patterns.
Please identify which of the following ground truth patterns is same with the \
algorithm output in semantic. For example, Non-upright face (nonvertical pose) \
is similar to faces with extreme head tilt, but not similar to bicycle lying on \
ground or bicycle upside down. The ground truth patterns are provided below as \
a numbered list.
Please respond with ONLY the number of the matching ground truth pattern. If \
none of the ground truth patterns are a good semantic match, respond with "-1". \
NOTE:The ground truth patterns and algorithm patterns should indicate the same \
objects, context, conditions and actions.

Algorithm output: {algorithm_output}

Ground truth patterns:
{ground_truth_list}"""


# Convenience mapping of the named dataset user prompts (Section A.2).
DATASET_USER_PROMPTS = {
    "waterbirds": WATERBIRDS_USER_PROMPT,
    "celeba": CELEBA_USER_PROMPT,
    "nico++": NICO_USER_PROMPT,
    "nico": NICO_USER_PROMPT,
}
