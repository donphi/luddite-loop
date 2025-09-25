# Pipeline 01-01: UK Biobank Showcase Data Ingestion

## 🔢 Pipeline Sequence: `01.01`
**Execution Order**: This is the first sub-pipeline of main pipeline 01 (Data Ingestion).
- **Previous**: None
- **Current**: 01.01 - UK Biobank Showcase Data Ingestion
- **Next**: 01.02 - Restructure UKB Showcase Data

## 📋 Purpose
This pipeline step covers the manual download of essential metadata and publication files from the UK Biobank showcase. These files provide the foundational data schema and context required for all downstream data processing and feature engineering tasks.

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
2.  **category.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=3).
3.  **publications.txt**: Download from [UK Biobank Schema](https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=19).

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
    ├── field.txt         # (Manual download)
    ├── category.txt      # (Manual download)
    └── publications.txt  # (Manual download)
```

## 🔧 Configuration
Not applicable for this manual pipeline step.

## 📊 Output Format
The "output" of this pipeline is the set of raw files downloaded from the UK Biobank, which serve as the input for the next pipeline stage.

### field.txt
Contains UK Biobank field definitions. Key columns include:
- `field_id`, `title`, `sexed`, `units`, `main_category`, `instance_id`, `notes`, `num_participants`, `tier`

### category.txt
Contains category classifications for fields. Key columns include:
- `category_id`, `title`, `notes`

### publications.txt
Contains research publications using UK Biobank data. Key columns include:
- `title`, `author(s)`, `journal`, `year of publication`, `abstract`, `DOI`, `URL`, `Total citations`

## 🐛 Troubleshooting
| Issue | Solution |
|-------|----------|
| Download links broken | The UK Biobank may have updated its schema URLs. Check the main showcase website for the latest links. |
| Files not found by next pipeline | Ensure the downloaded files are named exactly `field.txt`, `category.txt`, and `publications.txt` and are placed inside the `input/` directory. |

## 📝 Notes
-   For reproducibility, it is recommended to use a frozen dataset version if available (e.g., July 2025). However, live links are provided for the most current data.
-   This manual step is a prerequisite for the `01.02-restructure_ukb_showcase_data` pipeline, which processes these files.

---
✨ Pipeline ready for execution