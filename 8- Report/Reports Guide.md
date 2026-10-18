# Milestone 1 reports

These two reports complement each other and should be read together.

| Report | What it covers |
|---|---|
| [Analytical Report.pdf](Analytical%20Report.pdf) - Part I | Answers the three data-engineering questions: circuit front-row conversion, home-country podium association, and constructor mechanical retirements. Explains the selected features, their validity before race start, and historical/data limitations. **7 pages, 6 figures.** |
| [Technical Report.pdf](Technical%20Report.pdf) - Part II | Documents EDA and cleaning, data audits, chronological splitting, preprocessing, model comparison and metrics, learning/evaluation curves, feature ablation, preprocessing-order experiments, global/local XAI, inference, and evaluation limitations. **17 pages, 18 figures.** |

The analytical report explains the historical findings and feature choices. The technical report explains how the data and predictive models were prepared, tested and interpreted.

Together, the PDFs contain **all 24 notebook plots**, with explanations. The figures are embedded in the PDFs, so separate figure folders are unnecessary. The executed notebook remains the source for the code and full computational outputs.

[Technical+Analytical Report.pdf](Technical%2BAnalytical%20Report.pdf) is the combined submission copy containing both reports.

For local notebook execution, install the dependencies from the project-root `requirements-tested.txt` and select that Python environment. The notebook automatically finds the complete project `Dataset/` folder from either the project root or `Deliverables/1- Notebook/`. For a custom dataset location, optionally set:

```powershell
$env:F1_DATA_DIR = 'C:\Users\ROG STRIX\Downloads\ACL Project\Dataset'
```
