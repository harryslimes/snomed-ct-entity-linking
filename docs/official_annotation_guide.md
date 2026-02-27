# SNOMED CT Entity Linking Challenge - Annotation Guidelines

## Introduction

These annotation guidelines are adapted from research methodology and from working sessions with terminology specialists.

Machine interpretation of clinical text is a difficult problem in data science. Entity Linking (the specific task that SNOMED will be asking competition participants to complete) is an important part of the interpretation process. In an Entity Linking task, the machine must identify all occurrences of biomedical concepts in clinical free-text and link them to the correct concepts in a knowledge base.

The goal of this annotation project is to highlight all occurrences of (in-scope) SNOMED CT concepts that appear in a corpus of deidentified clinical notes. Since this task is focused solely on detecting the presence of concepts, we explicitly remove from scope the task of "metadata annotations" (more nuanced judgements about the role a detected concept plays). In particular, during this project:

- It does not matter whether a clinical finding was affirmed, negated or hypothesised.
- It does not matter whether a procedure or finding was past or present.
- It does not matter whether a finding is related to the patient, or to somebody else.

(Some exceptions to these are covered in the rules below.)

## The Guidelines

*In the examples that follow, underlined segments indicate the annotated span.*

### 1. Use the most detailed concept that is fully entailed by a span

Given a record "The patient suffers from chronic back pain" the following annotations could be considered:

- 22253000 | Pain (finding) |
- 82423001 | Chronic pain (finding) |
- 134407002 | Chronic back pain (finding) |
- 161891005 | Backache (finding) |
- 77568009 | Structure of back of trunk (body structure) |

**Chronic back pain (finding)** is the correct annotation here; each of the other candidates contain only some of the entailed detail.

### 2. Annotate the most detailed concept only when it is fully entailed

Given a record "The patient suffers from back ache and chronic pain" it is not clear if the "chronic pain" is specifically referring to the back ache. The best annotations are therefore:

- back ache → 161891005 | Backache (finding) |
- chronic pain → 82423001 | Chronic pain (finding) |

In contrast, the record "The patient suffers from back pain described as chronic" suggests a direct link between the back pain, and the description as "chronic", therefore the correct annotation would be:

- 134407002 | Chronic back pain (finding) |

### 3. Use a single annotation, where possible

Where a single, suitable concept exists for a span, use it. Put another way: do not split a span into two separate concepts if a single concept would suffice.

For example, in the following:

"CABG x 3 saphenous vein graft"

You should use the annotation:

- 58821000168109 | Coronary artery bypass with three saphenous vein grafts |

It would be incorrect to use multiple annotations, for example:

- 41801008 | Coronary artery structure |
- 362072009 | Saphenous vein structure |
- 24486003 | Transplant |

### 4. Prefer multiple concepts over a single concept only where this would extract greater detail from the record

One exception to Rule 3, is where multiple annotations would increase the accuracy and fidelity of the concepts that are extracted.

For "Orthopedics was consulted who elected to place wrist splint" the annotator has a choice:

Either to use a single annotation:

- "wrist splint" → 430354007 | Application of upper extremity splint (procedure) |

Or to use two annotations:

- "wrist" → 8205005 | Wrist region structure (body structure) |
- "splint" → 79321009 | Application of splint (procedure) |

In this situation, the latter is to be preferred as using two concepts gives the additional detail that the splint was placed onto the wrist and not just somewhere on the arm.

### 5. Use multiple concepts if a single concept does not exist

If a pre-coordinated concept for a span exists, you should use it (see Rule 3). However, if no pre-coordinated choices exist for a procedure, you can use a compound of concepts. For example, a procedure could be represented as: Procedure + Morphological Abnormality + Body Structure.

A specific example:

"Excision of lipoma of scalp."

Could be annotated:

- 65801008 | Excision |
- 134328007 | Lipoma |
- 41695006 | Scalp structure |

Because a sufficient pre-coordinated procedural concept does not exist.

### 6. Annotation spans may not be dis-continuous or overlapping

Separate annotations may not overlap each other in any way (this is a limitation of the annotation software as much as anything else). Therefore if we had "The patient suffered a cut to the middle toe" then the word "cut" would be annotated with:

- 283440002 | Cut of toe (disorder) |

as it is clear from the broader phrase that it is a cut to the toe, and the phrase "middle toe" would be annotated with:

- 78132007 | Third toe structure (body structure) |

### 7. Contextual proximity: the +/- 1 sentence rule

Sometimes, text elsewhere in a note may give context to an annotation. In some cases, such text may be far-removed from the span being annotated. It is time-consuming for an annotator to read back and forth through an entire record, seeking clues to help make the annotations more specific.

Therefore, when considering which context to include, look only at the previous, current and next sentence. Any contextual information not appearing in those three sentences should be ignored. For example:

"Patient suffers from stomach discomfort. He was admitted through the emergency department complaining of acute pain. Patient underwent ultrasound"

The +/- 1 sentence rule tells us that "stomach" can be used to get specific annotations for both "acute pain" and "ultrasound":

- 271681002 | Stomach ache (finding) | - "stomach discomfort"
- 116290004 | Acute abdominal pain (finding) | - "acute pain"
- 420052009 | Ultrasound scan of lower abdomen (procedure) | - "ultrasound"

### 8. Assess the annotation for homonym's that been incorrectly associated with a concept

When using annotation software, check the suggested annotations (if any) carefully. For example, given a record "After an epileptic fit the patient was exhausted and could not fit in any exercise" only the first "fit" would be annotated, and if the second instance was also annotated, it should be marked as "Incorrect":

- 313307000 | Epileptic seizure (finding) |

### 9. Where a finding is negative and a specific concept exists, use it

Certain findings or situational contexts in SNOMED CT appear in both their positive (affirmed) and negative (absent) forms. In these situations, the appropriate positive / negative form should be used. For example:

"The patient denies having fever or chills"

would be annotated with:

- 86699002 | APYREXIAL (SITUATION) |

### 10. Unverified or negative findings should still be annotated

Oftentimes, an unconfirmed (e.g. self-reported) clinical finding or a negative finding will be present in the text, but SNOMED CT will not contain a negative form of the concept. For the purposes of developing an Entity Linking dataset, such concepts should still be annotated. For example, in both of the following texts:

- "The patient suspects they may have a broken rib"
- "The patient does not have a rib fracture"

The following annotation should be marked:

- 33737001 | Fracture of rib (disorder) |

### 11. Annotations should not be incidental

Occasionally spans may be present in a record which – although they would lexically match to a concept – are actually incidental and represent an opportunity (by not annotating them) to teach an algorithm about irrelevant contexts.

For example, given a record "The patient is married to an epilepsy nurse", although:

- 84757009 | Epilepsy (disorder) |

Exists in SNOMED CT, the surrounding context makes it clear that the disorder itself is not being referenced. Therefore, in this example, no annotation should be made.

### 12. Where concepts have been omitted due to spelling, abbreviation or synonyms, add an annotation

Given a record "The patient suffers from AHDD" the ADHD ConceptID should be added:

- 406506008 | Attention deficit hyperactivity disorder (disorder) |

Given a record "After surgery the patient was oriented X3", the context would indicate the annotation below:

- 426224004 | Oriented to person, time and place (finding) |

Given a record "The patient suffers from polydipsia", the annotation below should be added:

- 17173007 | Excessive thirst (finding) |

### 13. Where terms exist in multiple hierarchies, assign in a specified order of preference

Context will usually determine the correct ConceptID to use for a specific term occurring in multiple hierarchies:

Given a record "The patient was diagnosed with a bone cyst" this is clearly a disorder which is a child of "findings" hence:

- 203465002 | Bone cyst (disorder) |

would be used

For "The bone cyst removed from the left leg weighed 50g", the context suggests that the "body structure":

- 66954000 | Bone cyst (morphologic abnormality) |

would be more appropriate.

If you find yourself in a situation in which it is simply not clear which term is to be preferred, than prioritise as follows:

- If a choice between Procedure and Finding, choose Procedure
- If a choice between Finding and Body Structure, choose Finding
- If a choice between Procedure and Body Structure, choose Procedure

### 14. Ignore Structured Headings

MIMIC-IV discharge notes follow a structured pattern, with certain headings present in most records. These headings (usually followed by a colon) should be left unannotated. They include:

- Allergies:
- Chief Complaint:
- Major Surgical or Invasive Procedure:
- History of Present Illness:
- Past Medical History:
- Social History:
- Family History:
- Physical Exam:
- Pertinent Results:
- Brief Hospital Course:
- Medications on Admission:
- Discharge Medications:
- Discharge Disposition:
- Discharge Diagnosis:
- Secondary diagnoses:
- Discharge Condition:
- Discharge Instructions:
- Followup Instructions:

### 15. Since physical objects are not being annotated, the "in situ" finding should be used unless further context is present

Given a record "The catheter was removed" since objects are not available:

- 266737003 | Indwelling urethral catheter (physical object) |

won't be available. Instead:

- 439053001 | Urinary catheter in situ (finding) |

should be used.

If the record had read "The catheter was inserted" instead - then it is clear that a procedure has occured and:

- 45211000 | Catheterization (procedure) |

would be used.

### 16. Ignore lab results

Many MIMIC records contain blood test results under the headings "LABS ON ADMISSION" and "LABS ON DISCHARGE". In both sections the data are heavily redacted. Furthermore, specialist clinical knowledge would be required in order to determine whether the results are "normal" or "abnormal". Therefore these sections should be omitted from the annotation process.

## Tips, Tricks & Points to Remember

### Time boxing

These are the most important "rules of thumb":

1. If you spend longer than 60 seconds worrying about whether to include or exclude an annotation: **exclude it**.
2. If you spend longer than 60 seconds trying to decide whether to use a more detailed (child) concept vs using its parent, **use the parent**.
3. If it is not clear whether a span that describes a body structure is describing the entire structure or a part of the structure, then opt for the parent: **Structure of ….**

### Repetition

Records will often contain repetitions of a particular concept ID - scan the document to validate the same repeated concept to save time.

After manually adding a concept - it will often appear further in the document. Scan the document to add all instances of that concept at once to save time returning to search for the concept ID.

### Intention of the Exercise

Remember the intention of the exercise - this is meant to represent a human's best effort to spot and list the SNOMED CT concepts that appear in a document. Do not concern yourself with how a machine might tell the difference between commonly used identical abbreviations, if the meaning is clear to the reader. Similarly, don't try to filter out concepts that you believe will not be relevant to a clinician's decision-making. Machine learning models of the future will no doubt need to be able to do such filtering; right now we are trying to perfect their ability simply to detect and link concepts.

### Using a Browser

It can be helpful to have a terminology browser open whilst annotation. Either the IHSDO Browser or the Shrimp Browser from CSIRO are suitable. If using a browser, make sure that you looking at the following:

- SNOMED CT International Edition May 2023
- Active concepts only
- Use only the following top-level categories: Findings, Body Structures, Situation with Explicit Context and Procedures.

### Consistency

The output of this exercise is to be used in Machine Learning models. As such - consistency between team members is more important than precise definitions. If a judgement call is made, ensure it is communicated to the team, and this policy updated for future reference.