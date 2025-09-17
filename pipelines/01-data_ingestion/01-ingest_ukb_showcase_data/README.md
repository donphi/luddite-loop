## Getting Started

### Data Sources

Before running any scripts, you'll need to download three essential files from the UK Biobank showcase. These files form the backbone of our feature extraction pipeline.

#### Option 1: Frozen Dataset (July 2025)
For reproducible research using the frozen dataset from July 2025, download:
- **field.txt**: https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=1
- **category.txt**: https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=3
- **publications.txt**: https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=19

#### Option 2: Live/Updated Dataset
For the most current UK Biobank data, download from:
- **field.txt**: https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=1
- **category.txt**: https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=3
- **publications.txt**: https://biobank.ndph.ox.ac.uk/ukb/schema.cgi?id=19

**Important:** Place all downloaded files in the `input/` folder before proceeding.

---

### Data Schema

#### field.txt
Contains UK Biobank field definitions. We extract:
- `field_id` - Unique identifier for each field
- `title` - Field name/description
- `sexed` - Sex-specific field indicator (0=unisex, 1=male, 2=female)
- `units` - Measurement units
- `main_category` - Links to category_id in category.txt ⚠️
- `instance_id` - Instance identifier
- `instance_min` - Minimum instance value
- `instance_max` - Maximum instance value
- `notes` - Additional field information
- `num_participants` - Number of participants with this field
- `tier` - Access tier level

#### category.txt
Contains category classifications for fields. We extract:
- `category_id` - Links to main_category in field.txt ⚠️
- `title` - Category name
- `notes` - Category description

#### publications.txt
Contains research publications using UK Biobank data. We extract:
- `title` - Publication title
- `author(s)` - Author list
- `journal` - Publication venue
- `year of publication` - Publication year
- `publication date` - Full publication date
- `abstract` - Publication abstract
- `DOI` - Digital Object Identifier
- `URL` - Publication URL
- `Total citations` - All-time citation count
- `Recent citations (last 2 years)` - Recent impact metric

---