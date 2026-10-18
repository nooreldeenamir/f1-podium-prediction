# %% [markdown]
# ## 5. Focused analytical report
#
# The report contains only the three requested areas: the three data-engineering
# answers with supporting figures, selected features and cutoff validity, and
# known limitations. Detailed implementation evidence remains in the notebook.

# %%
from datetime import datetime, timezone
import html
import json
import re
import unicodedata

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image as PDFImage, KeepTogether, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table as PDFTable, TableStyle,
)
from reportlab.lib.utils import ImageReader

report_dir = OUT / "pdf"
report_dir.mkdir(parents=True, exist_ok=True)
report_pdf_path = report_dir / "analytical_report.pdf"
report_md_path = OUT / "analytical_report.md"
report_tables_dir = OUT / "tables"
report_styles = getSampleStyleSheet()
report_ink = colors.HexColor("#142536")
report_blue = colors.HexColor("#175578")
report_light = colors.HexColor("#EDF3F7")
report_muted = colors.HexColor("#5A6B79")
report_styles.add(ParagraphStyle(
    name="ReportTitle", fontName="Helvetica-Bold", fontSize=31,
    leading=36, textColor=report_ink, spaceAfter=22,
))
report_styles.add(ParagraphStyle(
    name="ReportSubtitle", fontName="Helvetica", fontSize=13,
    leading=19, textColor=report_muted, spaceAfter=13,
))
report_styles.add(ParagraphStyle(
    name="ReportH1", fontName="Helvetica-Bold", fontSize=17,
    leading=21, textColor=report_blue, spaceBefore=12, spaceAfter=12,
    keepWithNext=True,
))
report_styles.add(ParagraphStyle(
    name="ReportH2", fontName="Helvetica-Bold", fontSize=11,
    leading=15, textColor=report_ink, spaceBefore=11, spaceAfter=6,
    keepWithNext=True,
))
report_styles.add(ParagraphStyle(
    name="ReportBody", fontName="Helvetica", fontSize=9.4,
    leading=13.2, textColor=report_ink, spaceAfter=8,
))
report_styles.add(ParagraphStyle(
    name="ReportSmall", fontName="Helvetica", fontSize=8,
    leading=11, textColor=report_muted, spaceAfter=6,
))
report_styles.add(ParagraphStyle(
    name="ReportCell", fontName="Helvetica", fontSize=7.1,
    leading=9.5, textColor=report_ink,
))
report_styles.add(ParagraphStyle(
    name="ReportCellHead", fontName="Helvetica-Bold", fontSize=7.2,
    leading=9.5, textColor=colors.white,
))
report_styles.add(ParagraphStyle(
    name="ReportCaption", fontName="Helvetica", fontSize=8,
    leading=11, textColor=report_muted, spaceAfter=8,
))
report_story = []
report_markdown = []
report_page_width, report_page_height = A4
report_width = report_page_width - 96


def report_ascii(value):
    """Use portable glyphs in the PDF's built-in fonts."""
    text_value = str(value)
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2011": "-", "\u2212": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2264": "<=", "\u2265": ">=", "\u2192": " -> ",
        "\u0394": "delta ", "\u00d7": " x ", "\u00b1": "+/-",
    }
    for original, replacement in replacements.items():
        text_value = text_value.replace(original, replacement)
    return unicodedata.normalize("NFKD", text_value).encode("ascii", "ignore").decode("ascii")


def report_paragraph(value, style="ReportBody"):
    safe_text = html.escape(report_ascii(value)).replace("\n", "<br/>")
    return Paragraph(safe_text, report_styles[style])


def report_heading(title, level=1, page_break=False):
    if page_break:
        report_story.append(PageBreak())
    report_story.append(report_paragraph(title, "ReportH1" if level == 1 else "ReportH2"))
    report_markdown.append("\n" + "#" * (level + 1) + " " + str(title) + "\n")


def report_text(value):
    if value is None or str(value).strip() == "":
        return
    report_story.append(report_paragraph(value))
    report_markdown.append(str(value) + "\n")


def report_read_table(filename):
    table_path = report_tables_dir / filename
    assert table_path.exists(), f"Required report evidence missing: {table_path}"
    return pd.read_csv(table_path)


def report_value(value):
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        if abs(value) < 1e5:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{value:,.0f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return report_ascii(value)


def report_add_table(frame, title=None, columns=None, max_rows=24, widths=None):
    if frame is None or frame.empty:
        return
    frame = frame.copy()
    if columns:
        available = [column for column in columns if column in frame.columns]
        if available:
            frame = frame[available]
    total_rows = len(frame)
    if max_rows is not None:
        frame = frame.head(max_rows)
    if title:
        report_heading(title, level=2)
    # Table cells wrap instead of overflowing; repeated headers survive page breaks.
    pdf_rows = [[report_paragraph(c.replace("_", " "), "ReportCellHead") for c in frame.columns]]
    for values in frame.itertuples(index=False, name=None):
        pdf_rows.append([report_paragraph(report_value(v), "ReportCell") for v in values])
    if widths is None:
        sample_lengths = [
            max(len(str(column)), min(45, frame[column].astype(str).str.len().quantile(.75)))
            for column in frame.columns
        ]
        weights = np.maximum(np.array(sample_lengths, dtype=float), 8)
        widths = list(report_width * weights / weights.sum())
    pdf_table = PDFTable(pdf_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    pdf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), report_blue),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, report_light]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), .6, report_blue),
    ]))
    report_story.extend([pdf_table, Spacer(1, 8)])
    md_frame = frame.map(report_value)
    md_rows = ["| " + " | ".join(map(str, md_frame.columns)) + " |",
               "| " + " | ".join(["---"] * len(md_frame.columns)) + " |"]
    md_rows.extend("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |"
                   for row in md_frame.itertuples(index=False, name=None))
    report_markdown.append("\n".join(md_rows) + "\n")
    if total_rows > len(frame):
        note = f"Showing {len(frame)} of {total_rows} rows. The complete table is in output/tables."
        report_story.append(report_paragraph(note, "ReportSmall"))
        report_markdown.append(note + "\n")


def report_mapping_lines(mapping):
    if not isinstance(mapping, dict):
        return [str(mapping)] if mapping else []
    lines = []
    for key, value in mapping.items():
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, default=str, ensure_ascii=False)
        elif isinstance(value, (float, int, np.number)):
            value = report_value(value)
        lines.append(f"{str(key).replace('_', ' ').capitalize()}: {value}")
    return lines


def report_footer(canvas, doc):
    canvas.saveState()
    page_width, page_height = doc.pageTemplate.pagesize
    canvas.setStrokeColor(colors.HexColor("#CBD6DE"))
    canvas.setLineWidth(.5)
    canvas.line(48, 39, page_width - 48, 39)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(report_muted)
    canvas.drawString(48, 27, "F1 Podium Prediction | Analytical report")
    canvas.drawRightString(page_width - 48, 27, f"{doc.page}")
    if doc.page > 1:
        canvas.drawString(48, page_height - 27, "RETROSPECTIVE EVIDENCE - PRE-RACE PREDICTION")
    canvas.restoreState()


# %%
# Keep the report focused on the three items required in the milestone brief.
report_analysis_summary = globals().get("analyses_summary", {})
report_catalog = globals().get("feature_catalog", pd.DataFrame())
report_generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

report_story.extend([
    Spacer(1, 42),
    report_paragraph("FORMULA 1", "ReportSubtitle"),
    report_paragraph("Podium Prediction\nAnalytical Report", "ReportTitle"),
    report_paragraph("Data-engineering answers, pre-race features, and limitations", "ReportSubtitle"),
    Spacer(1, 18),
    report_paragraph("Target: podium finish (positionOrder <= 3).\nPrediction cutoff: after qualifying and before race start.\nUnit: one driver entering one race.", "ReportSubtitle"),
    Spacer(1, 18),
    report_paragraph(f"Generated from the executed notebook: {report_generated}", "ReportSmall"),
])
report_markdown.extend([
    "# Formula 1 Podium Prediction: Analytical Report\n",
    f"Generated from the executed notebook: {report_generated}\n",
    "Target: podium finish (positionOrder <= 3). Prediction cutoff: after qualifying and before race start. Unit: one driver entering one race.\n",
])


report_figure_count = 0

def report_add_figure(filename, caption, max_height=300):
    global report_figure_count
    report_figure_count += 1
    numbered_caption = f"Figure {report_figure_count}. {caption}"
    figure_path = FIG / filename
    assert figure_path.is_file(), f"Required report figure missing: {figure_path}"
    image_width, image_height = ImageReader(str(figure_path)).getSize()
    scale = min(report_width / image_width, max_height / image_height)
    figure = PDFImage(str(figure_path), width=image_width * scale, height=image_height * scale)
    figure.hAlign = "CENTER"
    report_story.append(KeepTogether([
        report_paragraph(numbered_caption, "ReportCaption"), figure, Spacer(1, 12),
    ]))
    report_markdown.append(f"**{numbered_caption}**\n\n![{caption}](figures/{filename})\n")


report_heading("1. Answers to the three data-engineering questions", page_break=True)
report_heading("1.1 Front-row starts and circuit podium conversion", level=2)
report_text("Qualifying positions 1-2 and final grid slots 1-2 are compared separately because penalties can change the starting grid. The comparison uses 490 races with both definitions observed, excluding 87 races without complete front-row support. Recorded nonstarters are excluded from the start denominator. Circuit-era rankings require adequate support and are descriptive.")
report_text(report_analysis_summary.get("q1", "See the front-row conversion figures."))
report_add_figure("q1_frontrow_circuits.png", "Circuit podium conversion from qualifying and actual grid front rows, with uncertainty intervals.", 285)
report_add_figure("q1_era_changes.png", "Change in front-row conversion across the 2014 era boundary for circuits with common support.", 260)

report_heading("1.2 Home-country podium association after grid control", level=2)
report_text("Home-country status is based on an explicit mapping from recorded driver nationality to circuit country. Home and away rates are standardized over the same observed exact-grid-position by era strata. Whole races are resampled for uncertainty, preserving dependence among competitors. This controls observed starting position and broad era, not driver or team strength.")
report_text(report_analysis_summary.get("q2", "See the standardized home-country comparison figure."))
report_add_figure("q2_home_advantage.png", "Crude and grid/era-standardized home and away podium rates with race-bootstrap intervals.", 280)

report_heading("1.3 Constructor mechanical retirements by era", level=2)
report_text("The denominator is race starts; the numerator requires an explicit mechanical component cause and fewer recorded laps than the winner as an early-end proxy. Crash, driver-caused, and regulatory outcomes are excluded from the mechanical numerator. Positive-lap withdrawals remain starts. Strict and broader cause maps are compared for 2014-2021 and 2022-2024.")
report_text(report_analysis_summary.get("q3", "See the constructor mechanical-retirement figure."))
report_add_figure("q3_constructor_reliability.png", "Constructor mechanical-retirement rates by era, with strict and broad definitions and uncertainty intervals.", 315)

report_heading("2. Selected features and pre-race validity", page_break=True)
report_text("The prediction cutoff is after qualifying and before the race starts. Every selected predictor is available by that cutoff or is derived solely from completed earlier races. Current-race results, points, status, laps, pit stops, and post-race standings are excluded. Driver and constructor histories are shifted so the current race cannot contribute to its own features.")
report_text("Feature selection combines training-period exploratory evidence with domain reasoning. The table lists every selected input, its motivation, and the information available at the cutoff. The accompanying figures show training-only associations and feature effects; these are exploratory evidence, not causal effects or proof that a feature improves the final model.")
if report_catalog.empty:
    report_catalog = report_read_table("feature_catalog.csv")
report_feature_table = report_catalog.copy()
report_feature_table["group"] = report_feature_table["group"].str.title()
report_add_table(
    report_feature_table,
    "Selected feature rationale and cutoff availability",
    columns=["feature", "group", "motivation", "availability"],
    max_rows=None,
    widths=[94, 62, 160, report_width - 316],
)
report_add_figure("11_feature_evidence.png", "Training-only missingness and univariate association evidence for selected features.", 280)
report_add_figure("12_feature_effects.png", "Observed feature effects and podium rates used to motivate feature groups.", 300)

report_heading("3. Known limitations", page_break=True)
report_text("Historical qualifying coverage is sparse and uneven. Analyses that require qualifying data use supported periods and common coverage, while the modeling sensitivity compares a shorter training window. Historical coverage and timing conventions may still change the observed relationships.")
report_text("Formula 1 rules, cars, calendars, scoring systems, constructor identities, and field composition change across eras. The 2014 boundary is a useful descriptive comparison, not an isolated causal experiment. A model trained on past seasons may transfer poorly after regulatory changes or to new drivers and teams.")
report_text("The archived final grid is treated as information available before the race. The archive does not provide a publication timestamp for every grid update; live use requires a genuine pre-start snapshot. Historical form and standings are only valid when constructed from completed earlier races.")
report_text("Home-country status is a nationality-based proxy, not residence or personal identification. The three data-engineering analyses are observational; unmeasured driver, team, circuit, and selection differences remain. Mechanical retirement labels and recorded lap counts are imperfect proxies, and constructor IDs do not automatically unify renamed teams.")
report_text("These limitations constrain interpretation and transfer. Plots and feature associations describe the supplied archive; they do not establish causes or guarantee future prediction performance.")

report_figure_paths = [FIG / name for name in [
    "q1_frontrow_circuits.png", "q1_era_changes.png", "q2_home_advantage.png",
    "q3_constructor_reliability.png", "11_feature_evidence.png", "12_feature_effects.png",
]]
report_caption_map = globals().get("FIG_CAPTIONS", {})
report_figure_manifest_names = [path.name for path in report_figure_paths]

report_document = BaseDocTemplate(
    str(report_pdf_path), pagesize=A4, rightMargin=48, leftMargin=48,
    topMargin=45, bottomMargin=52, title="Formula 1 Podium Prediction Companion",
    author="ACL Project", subject="Leakage-safe pre-race modeling and analytical evidence",
)
report_landscape_width, report_landscape_height = landscape(A4)
report_document.addPageTemplates([
    PageTemplate(id="Portrait", pagesize=A4,
                 frames=[Frame(48, 52, report_width, report_page_height - 97,
                               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
                 onPage=report_footer),
    PageTemplate(id="Landscape", pagesize=landscape(A4),
                 frames=[Frame(48, 52, report_landscape_width - 96, report_landscape_height - 97,
                               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
                 onPage=report_footer),
])
report_document.build(report_story)
report_md_path.write_text("\n".join(report_markdown), encoding="utf-8")
(OUT / "report_figure_manifest.json").write_text(json.dumps({
    "figure_count": len(report_figure_paths),
    "figures": [path.name for path in report_figure_paths],
    "report": str(report_pdf_path.relative_to(OUT)),
    "generated_utc": report_generated,
}, indent=2), encoding="utf-8")
assert report_pdf_path.exists() and report_pdf_path.stat().st_size > 0
assert report_md_path.exists()
print(f"Analytical PDF: {report_pdf_path.resolve()}")
print(f"Markdown report: {report_md_path.resolve()}")
print(f"Focused report includes {len(report_figure_paths)} supporting figures.")

# %% [markdown]
# ### Report verification
#
# A text-structure check below verifies that the PDF contains the expected
# analytical sections. The report should also receive visual review before
# submission: text extraction cannot detect clipped tables or unreadable figures.

# %%
import fitz

with fitz.open(report_pdf_path) as report_check:
    report_extracted = "\n".join(page.get_text() for page in report_check)
    report_required_headings = [
        "Answers to the three data-engineering questions",
        "Front-row starts and circuit podium conversion",
        "Home-country podium association after grid control",
        "Constructor mechanical retirements by era",
        "Selected features and pre-race validity", "Known limitations",
    ]
    for report_required in report_required_headings:
        assert report_required in report_extracted, f"Missing report section: {report_required}"
    report_check_pages = len(report_check)
print(f"Report structure verified: {report_check_pages} pages.")
