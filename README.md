# A validated model for early prediction of group A streptococcal aetiology in necrotising soft tissue infections


[Sonja Katz, Jaco Suijker, Steinar Skrede, Annebeth Meij-de Vries, Anouk Pijpe, Anna Norrby-Teglund, Laura M Palma Medina, Jan K Damås, Ole Hyldegaard, Erik Solligård, Mattias Svensson, PerAID/PerMIT/INFECT study group, Knut Anders Mosevoll, Vitor AP Martins dos Santos, Edoardo Saccenti](https://www.medrxiv.org/content/10.1101/2024.06.05.24308478v1)



## Installation

```bash
conda env create -f environment.yml
source activate env_permit_gas
```




## Abstract

**Background**: Necrotising Soft Tissue Infections (NSTI) are life-threatening infections caused by a variety of microbes. Treatment strategies in NSTI are most commonly universal and have longstanding remained unchanged, and little improvement in patient outcomes are observed over time (McDermott et al. 2024).  Recent advances in the understanding of pathogenesis in NSTI subcategories may lay the foundation for targeted improvements in patient handling. We wanted to develop and externally validate machine learning (ML) models for early prediction of microbial aetiology to guide clinical decisions and explore a possible value in predicting clinical endpoints, encompassing surgery, patient management, and organ support in NSTI. 
 
**Methods**: For this study, we used data from the INFECT study—an international, multicenter, prospective observational cohort study investigating the clinical characteristics and pathogenesis of NSTI. A total of 409 patients above the age of 18 and with surgically confirmed NSTI cases were prospectively enrolled between February 2013 and June 2017 from five Scandinavian hospitals. More than 700 clinically relevant parameters were recorded from hospital admission to admission into an intensive care unit.  Machine learning models were externally validated using a Dutch retrospective multicenter cohort that comprised 216 patients admitted for acute treatment of NSTI to 11 centres between January 1, 2013, and December 31, 2017. ML models for the presence of Streptococcus pyogenes (group A streptococcus;GAS) and five clinical endpoints (risk of amputation, size of skin defect, maximum skin defect size, length of ICU stay, and need for renal replacement therapy) were developed implementing unsupervised variable selection, and comparing several ML algorithms. SHapley Additive exPlanations (SHAP) analysis was used to interpret the model. GAS predictive models were externally validated using data from a Dutch retrospective multicenter cohort.  

**Results**: Eight variables available pre-surgery (age, diabetes, different anatomical location of infection, prior surgical intervention, blood creatinine and haemoglobin concentrations) sufficed for prediction of GAS aetiology with high discriminatory power in both the development (ROC-AUC: 0.828; 95%CI 0.763, 0.883) and validation cohort (ROC-AUC: 0.758; 95%CI 0.696, 0.821). The prediction of clinical endpoints related to surgical management, and organs support aspects was unsuccessful. 

**Conclusions**: An externally validated prediction model for GAS aetiology in NSTI was successfully developed.  Early substantiation of GAS can guide tailored clinical decisions on treatment and infection control measures.

**Trial registration**: The INFECT study is registered at ClinicalTrials.gov (NCT01790698). 
