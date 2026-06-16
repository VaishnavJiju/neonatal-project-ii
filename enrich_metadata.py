import pandas as pd
import re

def clean_label(text):
    return re.sub(r'\[.*?\]', '', str(text)).strip()

# Load
df = pd.read_parquet('mal_ed_data/mal_ed_final.parquet')
meta = pd.read_csv('mal_ed_data/MAL-ED_0-60m_OntologyMetadata.txt', sep='\t')

# Enrichment Logic for specific common features
clinical_enrichment = {
    'mean people per room': 'Crowding index. High density is a proxy for poor hygiene and increased infectious disease transmission risk.',
    'WAMI index': 'Water, Assets, Maternal Education, and Income. A composite metric used to measure household-level resource availability.',
    'Overall socioeconomic score': 'A holistic measure of household financial and social standing within the country context.',
    'Birth weight (kg)': 'Critical baseline for neonatal survival and growth. Low birth weight is a primary driver of stunting.',
    'Mean pathogen count in stools': 'Pathogen burden. Measures the diversity of infections even in the absence of clinical diarrhea.',
    'breastfeed': 'Nutritional and immunological proxy. Breastfeeding provides essential passive immunity against common enteric pathogens.',
    'weaning': 'The complementary feeding gap. Growth faltering often begins when low-quality solid foods replace nutrient-dense milk.',
}

results = []
for col in df.columns:
    # Match by IRI or Label
    match = meta[meta['iri'].str.contains(col, case=False, na=False) | meta['label'].str.contains(col, case=False, na=False)]
    
    if not match.empty:
        row = match.iloc[0]
        desc = str(row['definition']) if pd.notna(row['definition']) else "Study variable."
        cat = str(row['category']) if pd.notna(row['category']) else "Other"
    else:
        desc = "Study variable."
        cat = "Other"
    
    # Apply enrichment
    for key, val in clinical_enrichment.items():
        if key in col.lower():
            desc = f"{desc} | CLINICAL CONTEXT: {val}"
            break
            
    results.append({
        'Feature': col,
        'Category': cat,
        'Definition': desc
    })

enriched_df = pd.DataFrame(results)
enriched_df.to_csv('mal_ed_data/enriched_codebook.csv', index=False)
print(f"Enriched {len(enriched_df)} feature descriptions.")
