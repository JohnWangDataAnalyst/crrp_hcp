# CRRP HCP CSV analysis pipeline

This folder contains a single Python script that runs the main CRRP HCP manuscript analyses from CSV outputs.

## Required input CSVs

Place these files in one folder:

```text
crrp_subject_parcel_biomarkers.csv
crrp_subject_network_biomarkers.csv
crrp_subject_global_biomarkers.csv
crrp_shuffle_parcel_biomarkers.csv
crrp_shuffle_network_biomarkers.csv
crrp_shuffle_global_biomarkers.csv
crrp_failed_subjects.csv
```

The script is robust to missing files. If a global or network CSV is missing, it will try to derive it from the parcel CSV.

## Run

```bash
python src/crrp_hcp_csv_analysis_pipeline.py \
  --input-dir data/processed \
  --output-dir outputs/crrp_hcp_analysis
```

## Outputs

The script writes:

```text
crrp_hcp_analysis_outputs/tables/
crrp_hcp_analysis_outputs/figures/
```

Important tables include:

- `data_inventory.csv`
- `global_biomarker_summary.csv`
- `network_biomarker_summary.csv`
- `network_rank1_frequency.csv`
- `CRRP_Allocation_parcel_map_reliability_summary.csv`
- `CRRP_Allocation_subject_to_template_correlations.csv`
- `intact_vs_shuffle_global_paired_tests.csv`
- `intact_vs_shuffle_network_profile_correlations.csv`
- `route_hierarchy_statistics.csv`, if route-energy columns are present
- `manuscript_ready_numbers.txt`

Important figures include:

- histograms of global subject-level biomarkers
- network-level biomarker bar plots
- Allocation subject-to-template reproducibility histogram
- intact-vs-shuffle network scatter plots
- intact-vs-shuffle global bar plots

## Main analyses included

1. Data inventory and failed-subject summary.
2. Global CRRP biomarker summary.
3. Network-level CRRP biomarker summary.
4. Network rank-1 frequency for Allocation, Switching, Contrast, Residual, and Flexibility.
5. Parcel-level Allocation reproducibility across subjects:
   - all pairwise subject-subject map correlations
   - subject-to-leave-one-out group-template correlations
6. Intact-vs-shuffle paired global tests with paired t-test, Wilcoxon signed-rank test, FDR q values, and paired Cohen's dz.
7. Intact-vs-shuffle network-profile correlations and Allocation-gradient range reduction.
8. Route hierarchy tests if THA/BG/CB observer-energy columns are available.


## Notes

The script auto-detects columns beginning with `CRRP_`. It expects `subject_id`, `region`, and `network` columns for parcel-level analyses.
