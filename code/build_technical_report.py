"""Build the companion PDF from the existing executed-notebook evidence.

Preserves the original analytical report and does not train or select models.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import shutil
import os
import re

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'Deliverables'
OUT = D / '8- Report'
SOURCE_OUT = ROOT / 'output/review/report_sources'
SOURCE_OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / 'tmp/technical_report_build/figures'
FIG.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'Technical Report.pdf'
MD = SOURCE_OUT / 'Technical Report.md'
ORIGINAL = OUT / 'Analytical Report.pdf'
original_hash = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()

def read(rel):
    return pd.read_csv(D / rel)

metrics = read('4- Model Comparison/Metrics.csv')
ablation = read('5- Experiments/Ablation.csv')
prep = read('5- Experiments/Preprocessing.csv')
inference = read('7- Inference/Inference Examples.csv')
tuning = read('4- Model Comparison/Tuning.csv')
metadata = json.loads((D / '7- Inference/Models/Metadata.json').read_text())
checks = json.loads((ROOT / 'output/explanation_checks.json').read_text())
summary = json.loads((ROOT / 'output/data_summary.json').read_text())
splits = pd.read_csv(ROOT / 'output/tables/split_summary.csv')
window = pd.read_csv(ROOT / 'output/tables/training_window_sensitivity.csv')
permutation = read('6- XAI/Permutation Importance.csv')
shap = read('6- XAI/Shap Importance.csv')
assert metadata['selected_model'] == checks['model'] == 'Shallow neural network'
assert metrics.loc[(metrics.model == metadata['selected_model']) & (metrics.split == 'validation'), 'AP'].iloc[0] == tuning.validation_AP.max()

font_dir = Path('C:/Windows/Fonts')
pdfmetrics.registerFont(TTFont('ReportSans', str(font_dir/'arial.ttf')))
pdfmetrics.registerFont(TTFont('ReportSansBold', str(font_dir/'arialbd.ttf')))
pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSansBold', italic='ReportSans', boldItalic='ReportSansBold')
NAVY = colors.HexColor('#124f70')
INK = colors.HexColor('#172635')
MUTED = colors.HexColor('#526777')
PALE = colors.HexColor('#edf4f8')
RED = colors.HexColor('#d94040')
W, H = A4
WIDTH = W - 88
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport', fontName='ReportSans', fontSize=9.6, leading=14, textColor=INK, spaceAfter=8))
styles.add(ParagraphStyle(name='SectionReport', fontName='ReportSansBold', fontSize=19, leading=23, textColor=NAVY, spaceAfter=13))
styles.add(ParagraphStyle(name='SubReport', fontName='ReportSansBold', fontSize=11.5, leading=15, textColor=NAVY, spaceBefore=8, spaceAfter=7))
styles.add(ParagraphStyle(name='CaptionReport', fontName='ReportSans', fontSize=8.2, leading=11, textColor=MUTED, spaceBefore=5, spaceAfter=11))
styles.add(ParagraphStyle(name='CellReport', fontName='ReportSans', fontSize=8.1, leading=10.7, textColor=INK))
styles.add(ParagraphStyle(name='HeadReport', fontName='ReportSansBold', fontSize=8.1, leading=10.7, textColor=colors.white))
styles.add(ParagraphStyle(name='CoverReport', fontName='ReportSansBold', fontSize=33, leading=38, textColor=NAVY, spaceAfter=14))
styles.add(ParagraphStyle(name='EyebrowReport', fontName='ReportSansBold', fontSize=10, leading=14, textColor=RED, spaceAfter=12))

story = []
markdown = ['# Formula 1 Podium Prediction - Technical Report', '', '**Part II: companion to the Analytical Report (Part I)**', '', 'Prepared 18 October 2026 from the existing executed-notebook evidence. No new modeling or selection was performed for this report.', '']
manifest = []

def plain(text):
    return str(text).replace('\u2013', '-').replace('\u2014', '-').replace('\u2011', '-').replace('\u2212', '-').replace('\u2192', ' > ')

def para(text, style='BodyReport', md=True):
    text = plain(text)
    story.append(Paragraph(escape(text), styles[style]))
    if md:
        markdown.extend([text, ''])

def heading(text):
    story.append(Paragraph(escape(plain(text)), styles['SectionReport']))
    markdown.extend(['## ' + plain(text), ''])

def sub(text):
    para(text, 'SubReport', md=False)
    markdown.extend(['### '+plain(text), ''])

def page(title):
    story.append(PageBreak())
    heading(title)

def table(headers, rows, widths=None, small=False):
    rows = [[plain(v) for v in row] for row in rows]
    data = [[Paragraph(escape(plain(h)), styles['HeadReport']) for h in headers]]
    data += [[Paragraph(escape(v), styles['CellReport']) for v in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, PALE]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5 if small else 7),('BOTTOMPADDING',(0,0),(-1,-1),5 if small else 7),
        ('LINEBELOW',(0,0),(-1,0),0.6,NAVY),
    ]))
    story.extend([t, Spacer(1, 9)])
    markdown.append('| ' + ' | '.join(map(plain,headers)) + ' |')
    markdown.append('| ' + ' | '.join(['---']*len(headers)) + ' |')
    markdown.extend('| ' + ' | '.join(v.replace('|','/') for v in row) + ' |' for row in rows)
    markdown.append('')

def figure(source, label, caption, max_h=235):
    src = ROOT / source
    assert src.is_file(), src
    dest = FIG / (label + '.png')
    shutil.copyfile(src, dest)
    with PILImage.open(dest) as im:
        iw, ih = im.size
    factor = min(WIDTH/iw, max_h/ih)
    img = Image(str(dest), width=iw*factor, height=ih*factor)
    img.hAlign = 'CENTER'
    n = len(manifest)+1
    cap = f'Figure T{n}. ' + plain(caption)
    story.append(KeepTogether([img, Paragraph(escape(cap), styles['CaptionReport'])]))
    relative_image = Path(os.path.relpath(dest, MD.parent)).as_posix()
    markdown.extend([f'![{plain(caption)}](<{relative_image}>)', '', '**'+cap+'**', ''])
    manifest.append({'figure': f'T{n}', 'file': dest.relative_to(ROOT).as_posix(), 'source': str(src.relative_to(ROOT)), 'caption':plain(caption), 'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})

def fmt(v, digits=4):
    return f'{float(v):.{digits}f}'

short = {'Logistic regression':'LR', 'Random forest':'RF', 'Shallow neural network':'NN'}

# 1. Cover and division of the two reports.
story.append(Spacer(1, 45))
para('FORMULA 1 / MILESTONE 1', 'EyebrowReport', md=False)
para('Technical Report', 'CoverReport', md=False)
para('Part II - Data quality, predictive modeling, experiments and explainability', 'SubReport')
para('Companion to the Analytical Report (Part I). Together, the two reports document the historical questions, pre-race feature validity and complete computational evidence from the run-all notebook.')
story.append(Spacer(1, 15))
table(['Report', 'Contents'], [
    ['Part I: Analytical Report', 'The three relational data-engineering answers and six figures; the 19 selected features and cutoff validity; historical/archive limitations.'],
    ['Part II: Technical Report', 'The remaining 18 figures; cleaning/audits; chronological split; all-model metrics; learning curves; ablation/order experiments; global/local XAI; inference and evaluation limitations.'],
], [140, WIDTH-140])
para('Prediction unit: one driver entering one race. Target: positionOrder <= 3. Prediction cutoff: after qualifying and final published grid, before the Grand Prix starts.')
para('Read Part I for the substantive answers to Q1-Q3. Read Part II for how data quality, validation, model behavior and inference support the predictive task. The accompanying notebook remains the executable source of the calculations.')
para('Evidence status: this document summarizes the existing corrected run; its creation is not a new blind test. Model choice and thresholds use validation data. The reused 2022-2024 test-period limitation is stated explicitly in Section 16.')
para('Prepared 18 October 2026', 'CaptionReport')

# 2. Raw EDA.
page('1. Raw data and initial EDA')
para('All 14 relational CSVs are loaded as strings before automatic missing-value conversion. This preserves literal \\N sentinels for counting, alongside blank cells and full-row duplicates. Raw inputs are hashed; cleaning is performed on copies.')
para(f'The archive contains {summary["raw_results"]:,} result records across 1950-2024. Season volume varies with calendar length and field composition, so row counts cannot be interpreted as a fixed-size panel. Race outcomes support retrospective analyses; they are excluded from current-race predictors.')
figure('Deliverables/2- EDA/01- Raw Placeholders.png','T01 Raw Placeholders','Raw sentinel and blank counts across all 14 tables. Qualifying-session absence is substantial and must be distinguished from an observed zero.',205)
figure('Deliverables/2- EDA/02- Raw Season Volume.png','T02 Raw Season Volume','Raw result records by season. Historical volume changes motivate calendar-aware comparisons and chronological evaluation.',170)

# 3. Cleaning.
page('2. Grain correction and missing values')
para('The prediction grain is one (raceId, driverId) pair. For a duplicated pair, retain the smallest positive grid position, then smallest resultId; if all grids are absent/nonpositive, retain the smallest resultId. This deterministic entry-based rule avoids choosing by finishing outcome, while retaining an audit of ambiguous shared drives.')
para(f'Removing {summary["duplicates_removed"]} extra rows across {summary["duplicate_pairs"]} pairs leaves {summary["unique_results"]:,} historical driver-race rows. The small prevalence change is measured rather than assumed harmless. In modeling, grid zero becomes missing and grid_unassigned records that state.')
figure('Deliverables/2- EDA/03- Duplicate Cleaning Effect.png','T03 Duplicate Cleaning','Deduplication changes historical row count and podium prevalence. The modeling table is subsequently restricted to 1994-2024.',200)
figure('Deliverables/2- EDA/04- Missingness Cleaning Effect.png','T04 Missingness Cleaning','Placeholder conversion preserves the observed missingness pattern; it changes representation without inventing measured values.',190)

# 4. Types and audits.
page('3. Semantic typing and integrity')
para('IDs/counts use nullable integers; measurements use numeric types; race/session dates and birth dates become datetimes. Qualifying, lap and pit durations become numeric seconds. Race result strings beginning with + are gaps, not total race durations, and are not coerced into elapsed time. Q2/Q3 absence often reflects elimination.')
table(['Audit', 'Observed result'], [
    ['Primary keys', '14 table checks: no null or duplicate primary-key rows.'],
    ['Foreign keys', '23 relationships: no null or unmatched required references.'],
    ['Semantic parsing', '87 conversion audit rows: zero unexpected nonmissing values lost.'],
    ['Source integrity', 'All 14 raw CSV hashes match before and after notebook execution.'],
], [118, WIDTH-118], small=True)
figure('Deliverables/2- EDA/05- Qualifying Coverage.png','T05 Qualifying Coverage','Observed qualifying coverage varies sharply across seasons. Historical absence is retained and represented through training-derived missingness indicators.',170)
figure('Deliverables/2- EDA/06- Typed Qualifying Duration.png','T06 Qualifying Duration','Qualifying Q1 duration in seconds. The 99th-percentile display truncation improves visibility; it does not delete raw observations.',155)

# 5. Split and correlation.
page('4. Chronology, preprocessing and features')
para('Train on 1994-2018, validate on 2019-2021 and evaluate on 2022-2024. Every driver from a race stays in one partition. Histories may use earlier completed races within validation/test, matching sequential pre-race deployment; model weights and preprocessing remain frozen.')
table(['Partition', 'Years', 'Rows', 'Races', 'Podiums'], [[r['split'],r['years'],f'{r["rows"]:,}',str(r['races']),str(r['podiums'])] for _,r in splits.iterrows()], [85,104,100,100,WIDTH-389], small=True)
para('Fit medians and StandardScaler on training only. Append 11 training-derived missingness indicators to the 19 original inputs. Part I documents every input and its cutoff validity. Driver histories are shifted; constructor outcomes are aggregated by team/race before shifting; standings come from the previous round of the same season.')
figure('output/figures/10_chronological_split.png','T07 Chronological Split','Whole-race chronological partitions and class prevalence. Random driver-row splitting would mix race conditions and future seasons across partitions.',145)
figure('output/figures/13_feature_correlation.png','T08 Feature Correlation','Training-only feature correlations. Related qualifying, grid, form and standings signals motivate regularization and grouped ablation; they are not independent causal effects.',275)

# 6. Models and metrics.
page('5. Models, selection and complete metrics')
para('Primary selection metric: validation average precision (AP), appropriate for a minority podium class. Secondary decision metric: F1, used to select a probability threshold on validation only. ROC-AUC summarizes ranking; precision and recall expose decision trade-offs. AP uses recall-weighted precision; trapezoidal PR-AUC interpolates the PR curve. Both are reported. Brier measures probability error.')
table(['Attempt', 'Mechanism / selected settings', 'Main limitation'], [
    ['LR baseline', 'Regularized additive log-odds. C=0.1; no class weighting. Six candidates.', 'Cannot automatically learn nonlinear interactions.'],
    ['RF', 'Bootstrap trees with random feature subsets. Unrestricted maximum depth, minimum leaf 12. Four candidates.', 'Can overfit historical relationships and drift.'],
    ['Shallow NN', 'One 32-unit ReLU layer, sigmoid output; L2=0.01, Adam 0.003, batch 256. Three candidates.', 'Limited historical sample, temporal drift and validation sensitivity.'],
], [65, 270, WIDTH-335], small=True)
para('All three attempts use identical rows, the same 19 original features and the same fitted preprocessing. Validation AP selects the NN (0.748718); its validation-F1 threshold is 0.347110. Weights remain training-only; choices are frozen before corrected holdout scoring.')
table(['Model','Split','ROC-AUC','AP','PR-AUC','F1','Precision','Recall','Brier','Thresh.'],
      [[short[r.model],r['split']]+[fmt(r[k]) for k in ['ROC_AUC','AP','PR_AUC','F1','precision','recall','Brier','threshold']] for _,r in metrics.iterrows()],
      [43,61]+[(WIDTH-104)/8]*8, small=True)
para('LR = logistic regression; RF = random forest; NN = shallow neural network. Thresholds maximize each model\'s validation F1, with the larger threshold breaking ties. Test scores never determine the chosen model or threshold in the current code.', 'CaptionReport')

# 7. Learning curves.
page('6. Neural learning and baseline comparison')
para('Keras Sequential uses one hidden ReLU layer and one sigmoid output, binary cross-entropy loss and Adam. A callback computes exact validation AP after each epoch before EarlyStopping reads it; the best-AP weights are restored. Validation is the explicit 2019-2021 partition, not a random row split.')
figure('output/figures/neural_learning_curve.png','T09 Neural Learning Curves','Training/validation loss and exact validation AP from Keras History. Checkpoint selection uses validation AP, not the final test period.',285)
table(['Attempt', 'Validation AP', 'Change from LR', 'Validation F1'],
      [[short[r.model],fmt(r.AP,6),f'{r.AP-metrics.loc[(metrics.model=="Logistic regression")&(metrics.split=="validation"),"AP"].iloc[0]:+.6f}',fmt(r.F1,6)] for _,r in metrics[metrics.split=='validation'].iterrows()], [95,140,140,WIDTH-375])
para('The selected NN restores epoch 140 from 160 trained epochs. Its validation AP improves on logistic regression by 0.034136; the RF also improves over that baseline. This supports the extra flexibility in this validation era, while repeated trials can still adapt to that era. Loss/learning curves do not establish future performance.')

# 8. Holdout.
page('7. Frozen-model holdout performance')
para('The 2022-2024 partition spans the regulation change. The selected NN has ROC-AUC 0.938333, AP 0.711050, trapezoidal PR-AUC 0.709687 and F1 0.668235 at threshold 0.347110. Precision is 0.642534 and recall 0.696078. A high ROC-AUC does not imply that nearly every podium prediction is correct.')
figure('Deliverables/4- Model Comparison/Holdout Curves.png','T10 Holdout Curves','ROC and precision-recall curves for all three frozen models. The dashed PR reference is test podium prevalence; model choice remains based on validation AP.',215)
figure('Deliverables/4- Model Comparison/Holdout Confusion Matrix.png','T11 Holdout Confusion Matrix','Selected NN at its frozen validation threshold: 142 true positives, 79 false positives, 62 false negatives and 1,076 true negatives.',235)
para('The NN has the highest test AP of these three models, but RF has slightly higher test F1 (0.682609). Different ranking and decision metrics can disagree; switching models after seeing that test result would compromise evaluation. These are corrected retrospective results on a reused test period, not a new blind test.', 'CaptionReport')

# 9. Ablation.
page('8. Feature-group ablation')
para('Remove one group at a time, refit preprocessing on training rows for the remaining features, and retrain logistic regression with its selected configuration fixed. Keep training/validation rows unchanged. AP is threshold-free; each reported F1 uses the candidate\'s validation-selected threshold.')
table(['Removed group', 'Features left', 'Validation AP', 'AP change', 'F1'],
      [[r.group_removed,str(r.feature_count),fmt(r.AP,6),f'{r.delta_AP:+.6f}',fmt(r.F1,6)] for _,r in ablation.iterrows()], [145,84,100,90,WIDTH-419], small=True)
figure('Deliverables/5- Experiments/Feature Group Ablation.png','T12 Feature Group Ablation','Every declared group is retrained against the full-feature logistic baseline. Negative changes mean removal hurts; positive changes are retained honestly.',260)
para('Removing grid hurts most (AP -0.017375); form and qualifying removals also reduce AP. Removing context improves AP by 0.010349, and removing standings improves it by 0.000566. Correlation and regularization permit substitution between inputs. These findings apply to the logistic ablation estimator; they do not establish that dropping context improves the selected NN. The primary comparison retains its predeclared full feature set.')

# 10. Preprocessing.
page('9. Two preprocessing-order experiments')
para('Comparison A changes median-impute then scale to scale observed values then impute in scaled units. StandardScaler ignores missing values during fitting, so fitting its variance before filling can alter the transformed data. Comparison B changes clip-impute-scale to impute-scale-clip, using training 1st/99th percentiles for continuous features only.')
table(['Order', 'Validation AP', 'F1', 'Threshold'],
      [[plain(r['order']),fmt(r.AP,6),fmt(r.F1,6),fmt(r.threshold,6)] for _,r in prep.iterrows()], [WIDTH-255,85,85,85], small=True)
figure('Deliverables/5- Experiments/Preprocessing Order Distributions.png','T13 Preprocessing Order Distributions','Observed transformations for A (qualifying_gap_pct) and B (driver_prior_starts). All bounds, medians and scales come from training only.',250)
para('A changes validation AP by +0.000206; B\'s second order changes AP by +0.000994 relative to its first order. qualifying_gap_pct has 1,987 missing training values; the maximum A1/A0 transformed difference is 10.2573. All 13 binary/indicator columns are protected from clipping. The 120-row value-effect table records ranges, dispersion and changed values for every input/order. These small validation differences are diagnostic, not proof of general superiority; the primary comparison keeps impute-then-scale.')

# 11. Coverage.
page('10. Historical-coverage sensitivity')
para('Refit the same selected logistic configuration on 2003-2018 instead of 1994-2018. Retain the same feature definitions and 2019-2021 validation rows, and refit preprocessing only on the shorter training window. Earlier completed events may still supply lagged histories.')
table(['Training window', 'Rows', 'Qualifying missing', 'Validation AP', 'F1'],
      [[r.training_window,f'{r.training_rows:,}',f'{r.qualifying_missing_rate:.2%}',fmt(r.AP,6),fmt(r.F1,6)] for _,r in window.iterrows()], [115,75,125,100,WIDTH-415])
figure('output/figures/training_window_sensitivity.png','T14 Training Window Sensitivity','Shortening the training window reduces qualifying-position missingness while changing both training size and era. Validation rows/configuration remain fixed.',285)
para('The shorter window changes validation AP from 0.714582 to 0.716418 (+0.001836), while the training sample falls from 9,799 to 6,411 rows. Qualifying-position missingness falls from 18.99% to 0.33%. Cars, entrants and rules also change, so this experiment does not isolate a causal effect of missing qualifying. The primary training window remains 1994-2018.')

# 12. Permutation.
page('11. Global permutation importance')
para('Explain the validation-selected NN. Permute one original validation input at a time and pass the perturbed data through frozen preprocessing and the model. Importance is the decrease in AP over eight seeded repeats; error bars show the repeat standard deviation.')
figure('Deliverables/6- XAI/Permutation Importance.png','T15 Permutation Importance','Validation AP loss after permuting each original feature. The largest losses identify inputs the fitted NN relies on under this perturbation procedure.',340)
table(['Leading input', 'Mean AP loss', 'Repeat SD'], [[r.feature,fmt(r.AP_drop_mean,6),fmt(r.AP_drop_std,6)] for _,r in permutation.head(5).iterrows()], [WIDTH-200,100,100], small=True)
para('Grid and qualifying position dominate this ranking; prior starts and standings/form add information. Related inputs can substitute, so individual permutation may understate a redundant group. Independent permutations can create unrealistic grid/form combinations. This measures predictive reliance, not a causal effect on podium outcomes.')

# 13. SHAP global.
page('12. Global SHAP explanation')
para('Kernel SHAP approximates the selected NN\'s podium probability using a seeded training background. The global validation sample is season-proportional: 21 rows from 2019, 17 from 2020 and 22 from 2021. Sampling uses no outcomes; any separately appended local-only case is excluded from global means and the beeswarm.')
figure('Deliverables/6- XAI/Shap Global.png','T16 Global SHAP','SHAP beeswarm for 60 validation predictions. Positive attribution raises model probability relative to the background; negative lowers it. Red/blue represent high/low standardized feature values.',370)
para('Grid, qualifying position and driver_form_podium_rate lead mean absolute SHAP importance. Low grid/qualifying-position numbers generally support higher predicted podium probability; higher recent podium form can also raise it. Attributions are measured in probability units and depend on the background/approximation. Correlated inputs share information, and SHAP attributions are not causal effects.')

# 14. Local SHAP.
page('13. Local SHAP: a single prediction')
para('Choose the validation prediction nearest 0.5 without consulting its outcome: raceId 1027, driverId 830 (2019, round 18). Explain this same case with SHAP and LIME. Its podium probability is 0.496954, exceeding the frozen 0.347110 threshold, although its observed label is 0. An explanation of a false positive can still faithfully describe model behavior.')
figure('Deliverables/6- XAI/Shap Local Waterfall.png','T17 Local SHAP Waterfall','Additive attribution from expected background probability to the selected prediction. Displayed feature values are standardized inputs; contributions are probability changes.',330)
table(['Check', 'Result'], [
    ['Model probability',f'{checks["local_probability"]:.12f}'],
    ['SHAP reconstruction',f'{checks["shap_reconstruction"]:.12f}'],
    ['Absolute reconstruction error',f'{checks["shap_absolute_error"]:.3e}'],
], [WIDTH-205,205], small=True)
para('Qualifying position and grid are the leading positive contributions for this row; round and year are leading negative contributions. The reconstruction check verifies additive consistency for this prediction. It does not establish causal validity, a perfect approximation for every coalition, or predictive correctness.')

# 15. LIME.
page('14. Local LIME and explanation fidelity')
para('LIME explains the same validation row using a local surrogate. Displayed conditions are in imputed original racing units; the prediction wrapper applies frozen scaling. Binary inputs and missingness flags are treated as categorical 0/1 states. Their perturbations remain binary, although independent perturbations can still break relationships such as grid and its missingness flag.')
figure('Deliverables/6- XAI/Lime Local.png','T18 Local LIME','LIME local surrogate for the same prediction as Figure T17. Green/red bars support or oppose the local podium prediction; the title reports weighted local R-squared.',310)
para('A grid of 6 or better and qualifying position of 7 or better support the local prediction under LIME\'s sampled neighborhood. The surrogate R-squared is 0.399350: about 39.9% of weighted neighborhood variation is explained, leaving substantial unexplained variation. Treat the displayed weights as approximate; they are not a strong stand-alone account of this network\'s behavior.')
para('Local SHAP supplies an additional attribution with checked additive reconstruction. Agreement in leading grid/qualifying signals is useful, but differences between explainers reflect their neighborhoods, backgrounds and approximation methods. Neither explanation demonstrates that changing a feature would cause a podium finish.')

# 16. Inference.
page('15. Inference and saved-model verification')
para('predict_podium(record) consumes the 19 numeric engineered pre-race inputs in the feature catalog. It is not a lookup from raw driver/race names: rolling histories and standings must already exclude the requested race. Omitted/null fields use frozen training medians and training-derived missingness flags; the response reports imputed fields.')
table(['Example', 'Podium probability', 'Binary call', 'Missing inputs'],
      [[r.example,fmt(r.podium_probability,6),str(r.podium_prediction), 'None' if r.example=='complete' else 'Qualifying + standings (6)' if r.example=='partially missing' else 'Grid only'] for _,r in inference.iterrows()],
      [130,125,75,WIDTH-330])
para('All examples use the same frozen NN threshold, 0.347110. Removing six qualifying/standings inputs changes probability from 0.595058 to 0.459572; missing grid changes it to 0.026343. These examples demonstrate handling, not calibrated confidence under arbitrary missingness. Heavy imputation reduces reliability.')
sub('Input consistency')
para('Omitted/null/nonpositive grid is treated as missing and derives grid_unassigned=1. A positive grid derives 0. Contradictory supplied flags, unknown keys, malformed numeric values and nonfinite numbers are rejected. Missing inputs must be omitted or represented as null, rather than NaN/infinity.')
sub('Native model persistence')
para('Training-fitted medians/scaler are saved with joblib, as are the logistic/forest estimators. The NN uses native Keras serialization. The notebook verifies reloaded-model prediction parity for all three estimators. The independent local review also reloaded the delivered models and reproduced every training/validation/test metric row to numerical precision.')
sub('Run configuration')
para('Install the project dependency files into the notebook\'s Python kernel. Dataset discovery searches the working directory and its ancestors for the complete 14-table Dataset folder, including when opened from Deliverables/1- Notebook. No manual path setting is needed for the delivered project layout. F1_DATA_DIR remains an optional override for custom locations; an explicit path is checked for every required CSV.')
para('A fresh local Run All with this configuration completed all 27 code cells. This demonstrates local execution with dependencies and dataset configured; it does not verify execution on Kaggle. Fresh outputs are written under output/ in the working directory.')

# 17. Limitations and coverage.
page('16. Limitations and two-report coverage')
sub('Evaluation confidence')
para('The corrected run follows inspection of earlier test results. The same 2022-2024 period is reused, so these are corrected retrospective scores, not a new independent blind test. In the current code, candidate grids, validation-based model/checkpoint/threshold rules and split boundaries remain fixed; no test-driven fitting or selection is used. Repeated validation diagnostics can still overfit one validation era.')
sub('Transfer and interpretation')
para('Historical qualifying coverage, rules, scoring, constructor identities and field sizes change. Final archived grid is assumed available before the start; genuine live use needs a timestamped pre-start snapshot. Qualifying gaps combine sessions/conditions. Weather, tyres and strategy are absent; probabilities are not externally calibrated, and independent binary calls need not yield exactly three podiums. Part I adds limitations specific to home-country mapping and mechanical-retirement labels.')
para('Permutation, SHAP and LIME describe predictive associations. Correlated features, synthetic perturbations, finite SHAP sampling/background choices and low LIME fidelity constrain interpretation. Feature importance is not proof that the feature causes podium outcomes.')
sub('Where each requirement is documented')
table(['Evidence', 'Report / location'], [
    ['Q1-Q3 answers and four question plots', 'Part I, Section 1, pages 2-4.'],
    ['Selected-feature rationale and two feature plots', 'Part I, Section 2, pages 5-6.'],
    ['Archive/era limitations', 'Part I, Section 3, page 7; supplemented above.'],
    ['Cleaning, EDA, split and correlation plots', 'Part II, Sections 1-4, Figures T1-T8.'],
    ['All-model metrics and learning/evaluation plots', 'Part II, Sections 5-7, Figures T9-T11.'],
    ['Ablation and preprocessing/coverage experiments', 'Part II, Sections 8-10, Figures T12-T14.'],
    ['Global/local XAI and noncausal interpretation', 'Part II, Sections 11-14, Figures T15-T18.'],
    ['Complete/partial inference and persistence', 'Part II, Section 15.'],
], [WIDTH-240,240], small=True)
para('Together, the reports contain all 24 notebook plots: 6 in Part I and 18 in Part II. Calculations, full candidate trials and supplementary audit tables remain available in the run-all notebook and exported results. Accessible Kaggle/GitHub publication and course-form submission follow finalization of the files.')
para('Sources: Milestone 1 brief; executed F1 Podium Prediction notebook; delivered audit/model/experiment/XAI/inference tables; exported learning-history, split, sensitivity and explanation-check evidence. Figure files are copied without alteration from those executed outputs.', 'CaptionReport')

def decorate(canvas, doc):
    canvas.saveState()
    canvas.setTitle('Formula 1 Podium Prediction - Technical Report (Part II)')
    canvas.setAuthor('F1 Podium Prediction Project')
    canvas.setFont('ReportSans',8)
    canvas.setFillColor(MUTED)
    canvas.drawString(44,H-27,'F1 Podium Prediction | Technical Report - Part II')
    canvas.setStrokeColor(colors.HexColor('#c4d4dd'))
    canvas.line(44,39,W-44,39)
    canvas.drawString(44,25,'Companion to the Analytical Report (Part I)')
    canvas.drawRightString(W-44,25,str(doc.page))
    canvas.restoreState()

assert len(manifest)==18
doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=53,bottomMargin=54,allowSplitting=1)
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
MD.write_text('\n'.join(markdown),encoding='utf-8')
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==original_hash
reader=PdfReader(PDF)
texts=[p.extract_text() for p in reader.pages]
assert len(reader.pages)==17, f'Unexpected page overflow: {len(reader.pages)} pages'
assert sum(len(p.images) for p in reader.pages)==18
joined=' '.join(('\n'.join(texts)).split())
assert all(f'Figure T{i}.' in joined for i in range(1,19))
assert all(s in joined for s in ['Primary selection metric','Secondary decision metric','0.399350','0.347110','not a new independent blind test'])
for item in manifest:
    assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()==item['sha256']
payload={'report':'Technical Report.pdf','pages':len(reader.pages),'figure_count':18,'original_analytical_pdf_unchanged':True,'original_analytical_pdf_sha256':original_hash,'combined_notebook_plot_count':24,'figures':manifest}
(SOURCE_OUT/'Figure Coverage.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(reader.pages),'figures':18,'original_unchanged':True},indent=2))
