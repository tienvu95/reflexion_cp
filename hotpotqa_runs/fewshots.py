WEBTHINK_SIMPLE6 = """Question: Are group 2 innate lymphoid cells increased in chronic rhinosinusitis with nasal polyps or eosinophilia?
Thought 1: I should search for the biopsy study that measured ILC2 frequencies in CRS.
Action 1: Search[sinus mucosa ILC2 frequency nasal polyps]
Observation 1: The abstract describes 35 CRS patients and reports that ILC2 frequencies tracked with nasal polyps, tissue eosinophilia, and eosinophil-dominant disease.
Thought 2: I need a statement explicitly saying ILC2s were elevated in the polyp/eosinophilic groups.
Action 2: Lookup[associated with nasal polyps]
Observation 2: “ILC2 frequencies were associated with the presence of nasal polyps (P = 0.002) as well as high tissue eosinophilia (P = 0.004).”
Thought 3: Because the investigators linked higher ILC2 counts to CRSwNP and eosinophilia, the answer is yes.
Action 3: Finish[yes]

Question: Does vagus nerve signaling contribute to steatohepatitis and obesity in PEMT-deficient mice?
Thought 1: I should examine the hepatic vagotomy experiment in the PEMT knockout model.
Action 1: Search[PEMT knockout hepatic vagotomy steatohepatitis]
Observation 1: After hepatic vagotomy, PEMT-/- mice lost their protection from high-fat diet obesity and steatohepatitis, whereas sham controls stayed resistant.
Thought 2: I want confirmation that interrupting vagal signals reversed both phenotypes.
Action 2: Lookup[vagotomy normalized steatohepatitis]
Observation 2: “Hepatic vagotomy abolished the protection against HFD-induced obesity and prevented steatohepatitis in Pemt(-/-) mice.”
Thought 3: Since cutting the vagus changed both outcomes, vagal signals must contribute.
Action 3: Finish[yes]

Question: Do systematic reviews in emergency medicine journals routinely search clinical trial registries?
Thought 1: I will review how many of the 2013 intervention reviews searched registries.
Action 1: Search[emergency medicine systematic review registry search frequency]
Observation 1: Only 8 of 41 reviews (20%) documented registry searches, though half of those searches found unpublished trials.
Thought 2: With only one-fifth performing the search, the practice is not routine.
Action 2: Finish[no]

Question: Does psammaplin A induce Sirtuin 1-dependent autophagic cell death in doxorubicin-resistant MCF-7 cells?
Thought 1: I should look for whether PsA decreased SIRT1 activity and triggered autophagy markers.
Action 1: Search[psammaplin A SIRT1 autophagy MCF-7/adr]
Observation 1: The abstract says PsA inhibited SIRT1 more potently than sirtinol and increased autophagy proteins and DRAM expression.
Thought 2: I want to confirm that autophagic death occurred in the resistant line and xenografts.
Action 2: Lookup[autophagy-related proteins increased]
Observation 2: “PsA markedly increased the expression levels of autophagy-related proteins…”
Thought 3: Therefore PsA induced SIRT1-dependent autophagic cell death in the resistant cells.
Action 3: Finish[yes]

Question: Did restaging CT after neoadjuvant chemoradiation alter surgical management in locally advanced rectal cancer?
Thought 1: I should recall how many scans detected new metastases and whether plans changed.
Action 1: Search[restaging CT neoadjuvant rectal management change]
Observation 1: Eighty-three of 91 patients underwent restaging CT, but only four scans showed suspicious distant lesions and none altered surgery.
Thought 2: Since no surgical plans changed, the answer should be no.
Action 2: Finish[no]

Question: Is cell death in mesial temporal sclerosis clearly apoptotic?
Thought 1: I will check whether hippocampi from mesial temporal sclerosis patients showed apoptotic markers.
Action 1: Search[MTS hippocampus apoptosis TUNEL Bax fractin]
Observation 1: Neurons had increased Bax and caspase-cleaved actin, but no TUNEL-positive cells or apoptotic nuclear morphology were detected.
Thought 2: Because some apoptotic pathways were activated but DNA fragmentation was absent, the evidence is inconclusive.
Action 2: Finish[maybe]"""

REFLECTIONS = """
Previous Trial:
Question: Are secretory phospholipases A2 secreted from ciliated cells and able to stimulate goblet cells?
Thought 1: I noted that IL-13-transformed goblet cells increased mucin when exposed to sPLA2, but I forgot to mention that only the ciliated-enriched cultures released the enzymes apically.
Action 1: Finish[yes]

Reflection: Next time I should reference the evidence that ciliated cells expressed sPLA2 V and X, secreted them to the lumen, and that those enzymes triggered cysteinyl leukotriene and mucin release from the goblet cells—clearly supporting “yes.”

Previous Trial:
Question: Is the cell death in mesial temporal sclerosis apoptotic?
Thought 1: I answered maybe but failed to cite the conflicting markers.
Action 1: Finish[maybe]

Reflection: I should emphasize that Bax and caspase activity increased, yet no TUNEL-positive neurons or apoptotic nuclear morphology were detected, explaining why apoptosis cannot be confirmed.
"""

COTQA_SIMPLE6 = """
Question: Does hepatic vagotomy eliminate the protection of PEMT-deficient mice against high-fat diet steatohepatitis?
Thought: Hepatic vagotomy restored obesity and normalized phospholipids while preventing steatohepatitis in PEMT-/- mice, so vagal signals are required for the phenotype.
Action: Finish[yes]

Question: Are secretory phospholipases A2 released by ciliated airway cells and able to stimulate goblet cells?
Thought: Ciliated-enriched cultures secreted sPLA2 V and X to the lumen, which triggered cysteinyl leukotriene and mucin release from IL-13-transformed goblet cells, so yes.
Action: Finish[yes]

Question: Do systematic reviews in emergency medicine routinely search clinical trial registries?
Thought: Only 8 of 41 intervention reviews documented registry searches, meaning the practice is uncommon.
Action: Finish[no]

Question: Does admission hyperglycemia predict failed fibrinolysis in ST-elevation myocardial infarction?
Thought: Failed reperfusion was associated with higher admission glucose (126 vs 110 mg/dL) and hyperglycemia independently predicted failure (HR 4.79), so yes.
Action: Finish[yes]

Question: Is the cell death observed in mesial temporal sclerosis definitively apoptotic?
Thought: Bax expression and caspase activity increased, yet no TUNEL-positive nuclei or apoptotic morphology were detected, so the answer is maybe.
Action: Finish[maybe]

Question: Does hidradenitis suppurativa carry a substantial comorbidity burden?
Thought: HS patients had higher prevalences of smoking, metabolic syndrome, arthropathies, PCOS, and psychiatric disorders than controls, so yes.
Action: Finish[yes]"""

COT_SIMPLE_REFLECTION = """
Question: Does hepatic vagotomy contribute to the development of steatohepatitis and obesity in PEMT-deficient mice?
Thought: I answered maybe because I only mentioned that PEMT-/- mice resist obesity.
Action: Finish[maybe]

Reflection: I should cite that hepatic vagotomy abolished the protection against high-fat diet obesity and prevented steatohepatitis in PEMT-/- mice, clearly supporting a “yes” answer.

Question: Do systematic reviews in emergency medicine routinely search clinical trial registries?
Thought: I answered no but failed to quote the actual proportion.
Action: Finish[no]

Reflection: I should explicitly report that only 8 of 41 intervention reviews (20%) documented registry searches to justify the “no.”
"""

COT = """Relevant PubMed Context: The chronic rhinosinusitis biopsy study counted ILC2s in 35 patients and 9 controls. ILC2 frequencies were linked to nasal polyps (P = 0.002), tissue eosinophilia (P = 0.004), eosinophil-dominant CRS (P = 0.001), and worse symptom scores.
Question: Are ILC2s increased in chronic rhinosinusitis with nasal polyps or eosinophilia?
Thought: Because ILC2 levels were significantly higher in CRSwNP and eosinophilic disease, the answer is yes.
Action: Finish[yes]

Relevant PubMed Context: Only 8 of 41 emergency medicine intervention reviews in 2013 documented registry searches, though half of those searches uncovered unpublished trials.
Question: Do emergency medicine systematic reviews routinely search clinical trial registries?
Thought: With just 20% conducting a search, registry queries are not routine.
Action: Finish[no]

Relevant PubMed Context: Temporal lobes resected from mesial temporal sclerosis patients showed increased Bax and fractin expression, but no TUNEL-positive cells or apoptotic nuclear morphology.
Question: Is cell death in mesial temporal sclerosis clearly apoptotic?
Thought: Caspase activation was present but DNA fragmentation was absent, so the evidence neither confirms nor refutes apoptosis.
Action: Finish[maybe]
"""

COT_REFLECT = """
Relevant PubMed Context: Eighty-three of ninety-one rectal cancer patients underwent routine restaging CT after neoadjuvant chemoradiation. Only four scans (5%) showed new suspicious distant lesions and none led to a change in surgical management.
Question: Are restaging CTs after chemoradiation high yield for altering rectal cancer surgery plans?
Thought: I mistakenly claimed yes by assuming any detection is useful.
Action: Finish[yes]

Reflection: The study explicitly states that no management changes occurred despite the scans. I should answer no and mention the 5% detection rate with zero impact on surgery.

Context: Among IPF lung transplant recipients, obese patients (BMI > 30 kg/m2) receiving bilateral grafts had 1.71× higher 90-day mortality than normal-weight BLT recipients.
Question: Do obese IPF patients face greater 90-day mortality after bilateral lung transplant?
Thought: I answered maybe without citing the odds ratio.
Action: Finish[maybe]

Reflection: I must quote the OR 1.71 (95% CI 1.03–2.85) showing significantly higher early mortality, so future answers should clearly state yes with that evidence.
"""

COT_REFLECT2 = """Relevant PubMed Context: In the systematic review audit, only 8 of 41 emergency medicine reviews searched a trial registry, so most did not.
Question: Do emergency medicine systematic reviews routinely include registry searches?
Thought: I said yes because I misread “identified unpublished trials” as common.
Action: Finish[yes]

Reflection: The numbers show registry searches were rare. I need to cite the 20% figure and conclude no.

Relevant PubMed Context: The nitric oxide study found active RA patients had mean NO ~82 µmol/L versus ~36 µmol/L in inactive disease and ~25 µmol/L in controls, correlating with DAS-28 and Sharp scores.
Question: Are higher nitric oxide levels linked to RA activity and joint damage?
Thought: I answered no because I focused on the lack of infection correlation.
Action: Finish[no]

Reflection: Since nitric oxide tracked with disease activity and radiographs, I should answer yes and mention the correlation coefficients (e.g., r=0.85 with DAS-28).

Relevant PubMed Context: Treatment of Helicobacter pylori-positive asthenozoospermic men lowered seminal IgA and improved progressive sperm motility.
Question: Does eradication therapy improve sperm motility in infertile men with seminal H. pylori?
Thought: I said maybe because I doubted the sample size.
Action: Finish[maybe]

Reflection: The study reported significant post-treatment motility gains, so future answers should acknowledge the positive effect even if the cohort was modest."""
