# Pipeline 01-01: UK Biobank Showcase Data Ingestion

## 🔢 Pipeline Sequence: `01.01`
**Execution Order**: This is the first sub-pipeline of main pipeline 01 (Data Ingestion).
- **Previous**: None
- **Current**: 01.01 - UK Biobank Showcase Data Ingestion
- **Next**: 01.02 - Restructure UKB Showcase Data

## 📋 Purpose
This pipeline step covers the manual download of essential metadata, encoding dictionaries, publication files, and the Medical Abbreviation and Acronym Meta-Inventory from the UK Biobank showcase and related sources. These files provide the foundational data schema, coding dictionaries, and acronym mappings required for all downstream data processing and feature engineering tasks.

## ⚙️ Hyperparameters
Not applicable for this manual pipeline step.

## 🎲 Seeds and Reproducibility
Not applicable for this manual pipeline step.

## 🚀 Execution Process

This is a **manual data gathering process** and does not involve automated scripts.

### Build Sequence:
1.  Navigate to the UK Biobank showcase website.
2.  Download the required files as specified below.
3.  Place the downloaded files into the `input/` directory.

### Execution Order:
1.  **field.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=1).
2.  **encoding.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=2).
3.  **category.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=3).
4.  **ehierstring.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=12).
5.  **ehierint.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=11).
6.  **esimpstring.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=6).
7.  **esimpint.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=5).
8.  **esimpreal.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=7).
9.  **esimptime.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=20).
10. **publications.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=19).
11. **Metainventory_Version1.0.0.csv**: Download the Medical Abbreviation and Acronym Meta-Inventory from either:
    - [Columbia University Department of Biomedical Informatics](https://github.com/cumc/medical_abbreviations)
    - [Zenodo Repository](https://zenodo.org/records/4567594)
    
    This is a large-scale American English database of medical abbreviations and their possible senses, sponsored by Columbia University Department of Biomedical Informatics and funded by the National Library of Medicine. Licensed under Creative Commons Attribution 4.0 International.

### Script Execution:
No scripts are executed in this pipeline stage.

## 💻 System Requirements
-   **Internet Connection**: Required to download files from the UK Biobank showcase.
-   **Storage**: Minimal storage required for the downloaded text files.

## 🐳 Docker Execution
Not applicable for this manual pipeline step.

## 📁 Folder Structure
```
01-ingest_ukb_showcase_data/
├── README.md
└── input/
    ├── field.txt                    # UK Biobank field definitions (Manual download)
    ├── category.txt                 # Category classifications (Manual download)
    ├── publications.txt             # Research publications (Manual download)
    ├── encoding.txt                 # Encoding metadata (Manual download)
    ├── ehierstring.txt              # Hierarchical string encodings (Manual download)
    ├── ehierint.txt                 # Hierarchical integer encodings (Manual download)
    ├── esimpstring.txt              # Simple string encodings (Manual download)
    ├── esimpint.txt                 # Simple integer encodings (Manual download)
    ├── esimpreal.txt                # Simple real number encodings (Manual download)
    ├── esimptime.txt                # Simple time encodings (Manual download)
    └── Metainventory_Version1.0.0.csv # Medical Abbreviation and Acronym Meta-Inventory (Manual download)
```

## 🔧 Configuration
Not applicable for this manual pipeline step.

## 📊 Output Format
The "output" of this pipeline is the set of raw files downloaded from the UK Biobank, which serve as the input for the next pipeline stage.

### field.txt
Contains UK Biobank field definitions. Key columns include:
- `field_id`, `title`, `sexed`, `units`, `main_category`, `instance_id`, `notes`, `num_participants`, `tier`

### encoding.txt
Contains encoding metadata linking fields to coding systems. Key columns include:
- `encoding_id`, `title`, `coded_as`, `structure`, `num_members`, `descript`

### category.txt
Contains category classifications for fields. Key columns include:
- `category_id`, `title`, `notes`

### Dictionary Files (ehier*.txt, esimp*.txt)
Contain the actual coding dictionaries for hierarchical and simple encodings:
- **ehierstring.txt**: Hierarchical string-based codings with parent-child relationships
- **ehierint.txt**: Hierarchical integer-based codings
- **esimpstring.txt**: Simple string codings (largest file, ~18MB)
- **esimpint.txt**: Simple integer codings
- **esimpreal.txt**: Simple real number codings
- **esimptime.txt**: Simple time-based codings

All dictionary files share common columns: `encoding_id`, `coding`, `meaning`, `node_id`, `parent_id`, `selectable`

### publications.txt
Contains research publications using UK Biobank data. Key columns include:
- `title`, `author(s)`, `journal`, `year of publication`, `abstract`, `DOI`, `URL`, `Total citations`

### Metainventory_Version1.0.0.csv
The Medical Abbreviation and Acronym Meta-Inventory containing mappings between medical abbreviations and their full forms. Key columns include:
- `SF`: Short form (abbreviation)
- `LF`: Long form (full phrase)
- `NormLF`: Normalized long form
- `Source`: Source database (UMLS, ADAM, Berman, Vanderbilt, etc.)

## 🐛 Troubleshooting
| Issue | Solution |
|-------|----------|
| Download links broken | The UK Biobank may have updated its schema URLs. Check the main showcase website for the latest links. |
| Files not found by next pipeline | Ensure all downloaded files are named exactly as specified and are placed inside the `input/` directory. |
| Meta-Inventory file missing | Download from Columbia University GitHub repository: https://github.com/cumc/medical_abbreviations or Zenodo: https://zenodo.org/records/4567594 |
| Large file handling | Dictionary files (especially esimpstring.txt ~18MB) may take time to download. Use `ls -lh input/` to verify file sizes. |
| Encoding issues | Ensure files are saved in UTF-8 encoding to prevent parsing errors in downstream processing. |

## 📝 Notes
-   For reproducibility, it is recommended to use a frozen dataset version if available (e.g., July 2025). However, live links are provided for the most current data.
-   This manual step is a prerequisite for the `01.02-restructure_ukb_showcase_data` pipeline, which processes these files.

---
✨ Pipeline ready for execution