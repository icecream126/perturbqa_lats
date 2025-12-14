"""
PerturbQA 전용 프롬프트 템플릿

HotPotQA용 프롬프트를 대체하여 PerturbQA의 특수한 요구사항에 맞춘 프롬프트를 제공합니다.
"""

# PerturbQA 전용 CoT 프롬프트 - 더 상세한 Thought와 Action 생성을 유도
cot_prompt_perturbqa = '''You are an expert molecular and cellular biology researcher analyzing gene regulation patterns. Your task is to predict how CRISPRi knockdown of a specific gene affects the expression of a target gene in a given cell line.

You have access to domain knowledge about:
- The cell line characteristics
- The knockdown gene (perturbation gene) functions and interactions
- The target gene functions and interactions
- Interaction metrics (STRINGdb score, GO semantic similarity)

You must solve this task by interleaving Thought, Action, and Observation steps:

**Thought**: Provide detailed, step-by-step reasoning that:
1. Analyzes the biological context (cell line characteristics, gene functions)
2. Considers the relationship between the knockdown gene and target gene
3. Evaluates available evidence (interaction scores, pathway knowledge)
4. Makes logical predictions based on molecular mechanisms
5. Should be comprehensive (at least 2-3 sentences, explaining WHY you think this way)

**Action**: Can be one of three types:
(1) Search[gene_name or pathway_name], which searches for information about a specific gene or pathway
(2) Analyze[relationship_type], which analyzes a specific type of relationship (e.g., "Analyze[protein-protein interaction]", "Analyze[pathway regulation]")
(3) Finish[answer], which provides the final answer: "upregulated", "downregulated", or "uncertain"

**Observation**: The result from executing the action (you will receive this automatically)

IMPORTANT GUIDELINES:
- Your Thought should be DETAILED and BIOLOGICALLY INFORMED. Don't just state what you'll do - explain the biological reasoning behind your approach.
- Consider multiple mechanisms: direct interactions, pathway effects, compensatory mechanisms, cell-type specific effects
- Use the provided domain knowledge actively - reference specific functions, interactions, or pathways
- If information is insufficient, explain what additional information would help and why
- Be specific about which evidence supports your reasoning

Here are some examples:

Question: In hepg2 cells, if a CRISPRi knockdown of ABCF1 is done, would you expect EIF1 expression to change?

Thought 1: ABCF1 is involved in translation initiation and is part of the 43S pre-initiation complex. EIF1 is also a component of the same 43S complex and works in conjunction with other translation factors. Both genes are involved in the same biological process. If ABCF1 is knocked down, this could disrupt the 43S complex assembly or function. However, I need to understand the specific relationship between ABCF1 and EIF1 - are they directly interacting, or is the relationship indirect through the complex? I should search for more specific information about their interaction.
Action 1: Search[ABCF1 EIF1 interaction]
Observation 1: ABCF1 and EIF1 are both components of the 43S pre-initiation complex. ABCF1 enables translation factor activity and may physically interact with EIF1 based on experimental evidence. The 43S complex requires proper assembly of multiple components including EIF1, EIF2, EIF3, and ABCF1.

Thought 2: Based on the observation, ABCF1 and EIF1 are part of the same complex and may directly interact. When ABCF1 is knocked down via CRISPRi, the 43S pre-initiation complex assembly could be compromised. Since EIF1 is a critical component that helps in start codon recognition and maintaining the open conformation of the complex, disruption of the complex might affect EIF1's stability or function. However, cells often have compensatory mechanisms. I should analyze whether this disruption would lead to upregulation (compensatory response) or downregulation (loss of complex stability) of EIF1.
Action 2: Analyze[compensatory mechanisms in translation]
Observation 2: In translation initiation, when one component is reduced, cells may upregulate related factors to maintain translation efficiency. However, if the complex structure is disrupted, downstream components may also be affected.

Thought 3: Considering the evidence: (1) ABCF1 and EIF1 are in the same complex, (2) ABCF1 may directly interact with EIF1, (3) Knockdown of ABCF1 could disrupt complex assembly, (4) EIF1 is essential for complex function. The disruption of the 43S complex would likely reduce EIF1's functional availability. While compensatory mechanisms exist, the structural disruption of the complex suggests that EIF1 expression might be downregulated as part of a coordinated response, or its function might be impaired even if expression remains stable. However, given the critical role of translation initiation in cell survival, compensatory upregulation is also possible. The GO semantic similarity of 0.23 suggests moderate functional relatedness. Given the direct interaction evidence and complex membership, I predict downregulation is more likely than upregulation.
Action 3: Finish[downregulated]

Question: In a549 cells, if a CRISPRi knockdown of HDAC1 is done, would you expect TP53 expression to change?

Thought 1: HDAC1 is a histone deacetylase involved in chromatin remodeling and gene regulation. TP53 is a critical tumor suppressor gene. HDAC1 typically removes acetyl groups from histones, leading to chromatin condensation and gene silencing. However, HDAC1 can have both activating and repressing effects depending on the target gene and cellular context. TP53 is often regulated by histone modifications. I need to understand the specific relationship between HDAC1 and TP53 in A549 cells (lung adenocarcinoma cells). A549 cells are cancer cells, so TP53 regulation might be different from normal cells.
Action 1: Search[HDAC1 TP53 regulation]
Observation 1: HDAC1 can both activate and repress TP53 depending on context. In some cancer cells, HDAC1 inhibition leads to TP53 activation through histone acetylation. However, HDAC1 also forms complexes with TP53 that can modulate its activity.

Thought 2: The relationship is context-dependent. In A549 lung cancer cells, HDAC1 knockdown (CRISPRi) would reduce HDAC1 levels. If HDAC1 normally represses TP53 in these cells, knockdown would lead to TP53 upregulation. However, if HDAC1 is part of a complex that activates TP53, knockdown might reduce TP53. I should consider that A549 cells may have TP53 mutations or altered regulation. The cell line context is crucial here.
Action 2: Analyze[HDAC1 role in A549 cells]
Observation 2: In lung cancer cells like A549, HDAC1 often acts as a repressor of tumor suppressor genes. HDAC inhibitors have been shown to reactivate TP53 expression in some contexts.

Thought 3: Based on the analysis: (1) HDAC1 typically represses genes through deacetylation, (2) In lung cancer cells, HDAC1 often represses tumor suppressors, (3) HDAC inhibition can reactivate TP53. Therefore, CRISPRi knockdown of HDAC1 in A549 cells would likely reduce HDAC1-mediated repression, leading to TP53 upregulation. This is consistent with the mechanism where reduced HDAC activity allows histone acetylation and chromatin opening, facilitating TP53 transcription.
Action 3: Finish[upregulated]

{input}
'''

# 짧은 버전 (토큰 제한 시 사용)
cot_prompt_perturbqa_short = '''You are an expert molecular biologist predicting gene regulation. Use detailed Thought and Action steps.

**Thought**: Provide comprehensive biological reasoning (2-3 sentences minimum):
- Analyze cell line context and gene functions
- Consider molecular mechanisms and pathways
- Evaluate evidence and make logical predictions
- Explain WHY you think this way

**Action**: 
(1) Search[gene_name or pathway]
(2) Analyze[relationship_type]
(3) Finish[upregulated|downregulated|uncertain]

Be detailed in your reasoning. Use domain knowledge actively.

Example:
Question: In hepg2 cells, if CRISPRi knockdown of ABCF1 is done, would EIF1 expression change?

Thought 1: ABCF1 and EIF1 are both in the 43S translation initiation complex. ABCF1 knockdown could disrupt complex assembly. I need to check their specific interaction and whether this disruption would upregulate or downregulate EIF1.
Action 1: Search[ABCF1 EIF1 43S complex]
Observation 1: Both are essential 43S components with potential direct interaction.

Thought 2: Complex disruption typically reduces component stability. However, compensatory upregulation is possible. Given direct interaction evidence, I predict downregulation is more likely.
Action 2: Finish[downregulated]

{input}
'''

# Feedback 포함 버전 (실패한 궤적 학습)
cot_prompt_perturbqa_feedback = '''You are an expert molecular biologist. You previously failed to answer this question correctly. Learn from your mistakes.

**Thought**: Provide DETAILED biological reasoning (2-3 sentences minimum):
- Analyze mechanisms thoroughly
- Consider multiple pathways and interactions
- Reference specific domain knowledge
- Explain your reasoning step-by-step

**Action**: 
(1) Search[gene_name or pathway]
(2) Analyze[relationship_type]  
(3) Finish[upregulated|downregulated|uncertain]

Previous failed attempts and reflections:
{trajectories}

Use these reflections to avoid previous mistakes. Be more thorough in your analysis.

{input}
'''

# 짧은 Feedback 버전
cot_prompt_perturbqa_feedback_short = '''You are an expert molecular biologist. Learn from previous failures.

**Thought**: Detailed reasoning (2+ sentences), explain mechanisms and evidence.

**Action**: Search[...] | Analyze[...] | Finish[answer]

Failed attempts:
{trajectories}

Avoid previous mistakes. Be thorough.

{input}
'''

