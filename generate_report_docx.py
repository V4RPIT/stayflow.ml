"""Script to generate a comprehensive, executive-quality Word (.docx) report
for the Customer Churn Prediction Pipeline assignment.
"""

from datetime import datetime
import os
from pathlib import Path
import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

# Color Palette Constants
COLOR_PRIMARY = RGBColor(27, 54, 93)       # Deep Navy #1B365D
COLOR_SECONDARY = RGBColor(13, 148, 136)   # Teal #0D9488
COLOR_BODY = RGBColor(45, 55, 72)          # Charcoal #2D3748
COLOR_MUTED = RGBColor(100, 116, 139)      # Slate Gray #64748B
COLOR_WHITE = RGBColor(255, 255, 255)
HEX_PRIMARY = "1B365D"
HEX_SECONDARY = "0D9488"
HEX_LIGHT_BG = "F8FAFC"
HEX_CALLOUT_BG = "F1F5F9"
HEX_BORDER = "CBD5E1"


def set_cell_background(cell, hex_color: str):
    """Set the background shading of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
    """Set cell margins in dxa (1 pt = 20 dxa)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def style_table(table, col_widths=None):
    """Apply styling, borders, and margins to a table."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl_pr.append(borders)

    # Apply col widths if provided
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                if idx < len(row.cells):
                    row.cells[idx].width = width


def style_table_header(row, hex_bg=HEX_PRIMARY):
    """Format table header row."""
    for cell in row.cells:
        set_cell_background(cell, hex_bg)
        set_cell_margins(cell, top=160, bottom=160, left=180, right=180)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = COLOR_WHITE
                run.font.size = Pt(10)
                run.font.name = "Calibri"


def style_table_rows(table, zebra=True):
    """Format table data rows with alternating zebra striping."""
    for r_idx, row in enumerate(table.rows[1:]):
        bg_color = HEX_LIGHT_BG if (zebra and r_idx % 2 == 1) else "FFFFFF"
        for cell in row.cells:
            if bg_color != "FFFFFF":
                set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = COLOR_BODY
                    run.font.name = "Calibri"


def add_callout(doc, text: str, title: str = "KEY BUSINESS TAKEAWAY"):
    """Insert a highlighted callout box with a prominent left border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, HEX_CALLOUT_BG)
    set_cell_margins(cell, top=180, bottom=180, left=220, right=200)

    # Set left border thick and remove others
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{HEX_SECONDARY}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"📌 {title}: ")
    run_t.font.bold = True
    run_t.font.size = Pt(10)
    run_t.font.color.rgb = COLOR_SECONDARY
    run_t.font.name = "Calibri"

    run_b = p.add_run(text)
    run_b.font.italic = True
    run_b.font.size = Pt(10)
    run_b.font.color.rgb = COLOR_BODY
    run_b.font.name = "Calibri"

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_code_snippet(doc, code_str: str, caption: str = ""):
    """Insert a monospaced code block in a shaded container."""
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(6)
        p_cap.paragraph_format.space_after = Pt(2)
        r_cap = p_cap.add_run(f"Listing: {caption}")
        r_cap.font.bold = True
        r_cap.font.size = Pt(9)
        r_cap.font.color.rgb = COLOR_MUTED
        r_cap.font.name = "Calibri"

    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)

    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="12" w:space="0" w:color="{HEX_PRIMARY}"/>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(code_str.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def build_word_document():
    doc = docx.Document()

    # Configure Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

        # Header and Footer
        header = section.header
        p_hdr = header.paragraphs[0]
        p_hdr.text = "Customer Churn Prediction & Retention Decision Pipeline | Technical Assignment Report"
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_hdr.runs[0].font.size = Pt(8.5)
        p_hdr.runs[0].font.color.rgb = COLOR_MUTED
        p_hdr.runs[0].font.name = "Calibri"

        footer = section.footer
        p_ftr = footer.paragraphs[0]
        p_ftr.text = "Confidential - For Academic & Technical Evaluation"
        p_ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ftr.runs[0].font.size = Pt(8.5)
        p_ftr.runs[0].font.color.rgb = COLOR_MUTED
        p_ftr.runs[0].font.name = "Calibri"

    # ==========================================
    # COVER / HEADER TITLE BLOCK
    # ==========================================
    p_pre = doc.add_paragraph()
    p_pre.paragraph_format.space_before = Pt(12)
    p_pre.paragraph_format.space_after = Pt(4)
    run_pre = p_pre.add_run("END-TO-END APPLIED MACHINE LEARNING & DEPLOYMENT ASSIGNMENT")
    run_pre.font.size = Pt(10.5)
    run_pre.font.bold = True
    run_pre.font.color.rgb = COLOR_SECONDARY
    run_pre.font.name = "Calibri"

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(8)
    run_title = p_title.add_run("Customer Churn Prediction & Retention Decision Pipeline")
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY
    run_title.font.name = "Calibri"

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run(
        "A Production-Grade ML Solution with Scikit-learn, 5-Fold Stratified Cross-Validation, "
        "Financial ROI Threshold Optimization, and FastAPI REST Microservice"
    )
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = COLOR_MUTED
    run_sub.font.name = "Calibri"

    # Metadata Card Table
    meta_tbl = doc.add_table(rows=2, cols=3)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    style_table(meta_tbl, [Inches(2.1), Inches(2.2), Inches(2.2)])

    headers = [("Domain", "Subscription Telecom / SaaS"), ("Core Frameworks", "Python, Scikit-learn, FastAPI"), ("Evaluation Metric", "ROC-AUC & Financial Retention ROI")]
    for i, (k, v) in enumerate(headers):
        c_top = meta_tbl.rows[0].cells[i]
        c_bot = meta_tbl.rows[1].cells[i]
        set_cell_background(c_top, HEX_CALLOUT_BG)
        set_cell_background(c_bot, HEX_CALLOUT_BG)
        set_cell_margins(c_top, top=80, bottom=40, left=100, right=100)
        set_cell_margins(c_bot, top=40, bottom=80, left=100, right=100)

        p1 = c_top.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(k.upper())
        r1.font.bold = True
        r1.font.size = Pt(8.5)
        r1.font.color.rgb = COLOR_SECONDARY

        p2 = c_bot.paragraphs[0]
        p2.paragraph_format.space_after = Pt(0)
        r2 = p2.add_run(v)
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = COLOR_PRIMARY
        r2.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ==========================================
    # SECTION 1: EXECUTIVE SUMMARY
    # ==========================================
    h1 = doc.add_heading(level=1)
    r_h1 = h1.add_run("1. Executive Summary & Objective")
    r_h1.font.color.rgb = COLOR_PRIMARY
    r_h1.font.size = Pt(16)
    r_h1.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.add_run(
        "In recurring-revenue subscription businesses (such as telecommunications, SaaS platforms, and digital streaming), "
        "customer retention is a paramount driver of long-term profitability. Because acquiring a new subscriber typically costs "
        "5 to 7 times more than retaining an existing account, early identification of at-risk customers enables targeted intervention. "
        "This project implements an end-to-end machine learning engineering pipeline that ingests raw subscriber data, handles missing values, "
        "engineers four high-impact domain features, conducts stratified 5-fold cross-validated hyperparameter optimization, evaluates models on "
        "both statistical discrimination (ROC-AUC, PR-AUC) and business financial ROI, and packages the top model as a high-performance REST API."
    )

    add_callout(
        doc,
        "Predicting customer churn is not merely an accuracy exercise; it is an asymmetric cost-benefit decision problem. "
        "A False Positive incurs only a small retention promotional cost ($50), whereas a False Negative forfeits the full Customer "
        "Lifetime Value ($500). By sweeping decision thresholds, the selected Logistic Regression model achieves an optimal operating "
        "threshold of 0.16, generating $47,350 in net financial benefit and capturing 90.64% of all churning accounts.",
        title="CORE MACHINE LEARNING TAKEAWAY"
    )

    # Core Requirements Fulfillment Table
    p_t1 = doc.add_paragraph()
    p_t1.paragraph_format.space_before = Pt(8)
    p_t1.paragraph_format.space_after = Pt(4)
    r_t1 = p_t1.add_run("Table 1.1: Implementation Requirements Fulfillment Matrix")
    r_t1.font.bold = True
    r_t1.font.size = Pt(10)
    r_t1.font.color.rgb = COLOR_PRIMARY

    t_req = doc.add_table(rows=6, cols=3)
    style_table(t_req, [Inches(1.8), Inches(3.6), Inches(1.1)])
    headers_req = ["Requirement", "Implementation Mechanism", "Result Status"]
    for i, h in enumerate(headers_req):
        t_req.rows[0].cells[i].paragraphs[0].text = h
    style_table_header(t_req.rows[0])

    data_req = [
        ("Data Loading & Cleaning", "Ingested 7,043 Telco records; stripped string whitespaces; coerced TotalCharges and imputed 11 zero-tenure records with 0.0.", "Completed ✅"),
        ("Feature Engineering", "Engineered 4 custom features inside an sklearn Transformer: tenure_group, avg_monthly_spend, monthly_to_total_ratio, total_services_count.", "Completed ✅"),
        ("Model Training & Tuning", "Trained Logistic Regression, Random Forest, and Gradient Boosting using 5-Fold Stratified K-Fold CV via GridSearchCV.", "Completed ✅"),
        ("Multi-Metric Evaluation", "Evaluated ROC-AUC, PR-AUC, F1, Accuracy, Brier score, and financial campaign ROI across probability thresholds.", "Completed ✅"),
        ("REST API Deployment", "Built production FastAPI service with /predict, /predict/batch, /health, /model-info, Pydantic v2 schemas, and interactive Swagger UI.", "Completed ✅"),
    ]
    for r_idx, row_data in enumerate(data_req):
        row = t_req.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            row.cells[c_idx].paragraphs[0].text = val
    style_table_rows(t_req)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ==========================================
    # SECTION 2: DATA CLEANING & PREPROCESSING
    # ==========================================
    h2 = doc.add_heading(level=1)
    r_h2 = h2.add_run("2. Data Ingestion, Cleaning & Exploration")
    r_h2.font.color.rgb = COLOR_PRIMARY
    r_h2.font.size = Pt(16)
    r_h2.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.add_run(
        "The project utilizes the industry-benchmark Telco Customer Churn dataset, representing 7,043 customers across 21 raw attributes. "
        "The target variable 'Churn' exhibits an inherent class imbalance with 5,174 retained accounts (73.46%) and 1,869 churned accounts (26.54%). "
        "A rigorous exploratory audit identified crucial real-world data preparation requirements:"
    )

    p_clean1 = doc.add_paragraph(style="List Bullet")
    p_clean1.add_run("Whitespace Padding & String Types: ").font.bold = True
    p_clean1.add_run("All string-type categorical columns contained trailing whitespaces which were systematically trimmed.")

    p_clean2 = doc.add_paragraph(style="List Bullet")
    p_clean2.add_run("Missing Value Anomaly in TotalCharges: ").font.bold = True
    p_clean2.add_run(
        "A total of 11 records contained blank spaces (' ') in the TotalCharges attribute. Crucially, domain analysis confirmed "
        "that all 11 records possess tenure == 0 (customers who signed up within the current billing cycle and have not yet been invoiced). "
        "Imputing these values with 0.0 perfectly preserves physical meaning without introducing artificial distortion."
    )

    p_clean3 = doc.add_paragraph(style="List Bullet")
    p_clean3.add_run("Stratified Partitioning: ").font.bold = True
    p_clean3.add_run(
        "The dataset was split into an 80% training set (5,634 records) and a 20% held-out testing set (1,409 records) "
        "using stratified sampling on the Churn target, ensuring exact 26.54% churn prevalence across both subsets."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ==========================================
    # SECTION 3: DOMAIN FEATURE ENGINEERING
    # ==========================================
    h3 = doc.add_heading(level=1)
    r_h3 = h3.add_run("3. Domain Feature Engineering & Pipeline Architecture")
    r_h3.font.color.rgb = COLOR_PRIMARY
    r_h3.font.size = Pt(16)
    r_h3.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.add_run(
        "To maximize predictive power and ensure zero training-serving skew, all feature engineering is encapsulated in a custom Scikit-learn "
        "transformer, ChurnFeatureEngineer, which executes within a unified Pipeline before the ColumnTransformer. "
        "Four high-value domain features were engineered:"
    )

    # Feature Engineering Table
    p_t2 = doc.add_paragraph()
    p_t2.paragraph_format.space_before = Pt(6)
    p_t2.paragraph_format.space_after = Pt(4)
    r_t2 = p_t2.add_run("Table 3.1: Engineered Feature Definitions & Business Rationale")
    r_t2.font.bold = True
    r_t2.font.size = Pt(10)
    r_t2.font.color.rgb = COLOR_PRIMARY

    t_feat = doc.add_table(rows=5, cols=4)
    style_table(t_feat, [Inches(1.5), Inches(1.8), Inches(1.0), Inches(2.2)])
    headers_feat = ["Feature Name", "Mathematical Formulation", "Data Type", "Business & Domain Impact"]
    for i, h in enumerate(headers_feat):
        t_feat.rows[0].cells[i].paragraphs[0].text = h
    style_table_header(t_feat.rows[0])

    data_feat = [
        ("tenure_group", "cut(tenure, [-1, 12, 24, 48, 60, 100]) -> 5 bins", "Categorical", "Captures non-linear hazard curves. Year 1 (0-12m) exhibits peak churn risk, whereas 60m+ indicates high brand loyalty."),
        ("avg_monthly_spend", "TotalCharges / (tenure + 1.0)", "Numerical", "Measures average historical monthly billing pace; highlights price jumps when compared directly against MonthlyCharges."),
        ("monthly_to_total_ratio", "MonthlyCharges / (TotalCharges + 1.0)", "Numerical", "Reflects account volatility and financial maturity; high values strongly correlate with recent uncommitted subscribers."),
        ("total_services_count", "Sum(PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)", "Numerical (0-8)", "Quantifies customer ecosystem stickiness. Subscribers with 4+ bundled services experience dramatically lower attrition."),
    ]
    for r_idx, row_data in enumerate(data_feat):
        row = t_feat.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            row.cells[c_idx].paragraphs[0].text = val
    style_table_rows(t_feat)

    p_pipe = doc.add_paragraph()
    p_pipe.paragraph_format.space_before = Pt(8)
    p_pipe.paragraph_format.space_after = Pt(4)
    p_pipe.add_run("Preprocessing Pipeline Architecture: ").font.bold = True
    p_pipe.add_run(
        "Numerical features are processed through SimpleImputer(strategy='median') followed by StandardScaler(). "
        "Categorical features are processed through SimpleImputer(strategy='most_frequent') followed by "
        "OneHotEncoder(handle_unknown='ignore', sparse_output=False), expanding the total input feature space to 54 standardized columns."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==========================================
    # SECTION 4: MODEL TRAINING & HYPERPARAMETER TUNING
    # ==========================================
    h4 = doc.add_heading(level=1)
    r_h4 = h4.add_run("4. Model Training & Cross-Validated Hyperparameter Tuning")
    r_h4.font.color.rgb = COLOR_PRIMARY
    r_h4.font.size = Pt(16)
    r_h4.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.add_run(
        "Three distinct classification paradigms were systematically tuned using StratifiedKFold(n_splits=5, shuffle=True, random_state=42). "
        "Hyperparameter exploration was conducted via GridSearchCV optimizing directly for ROC-AUC to prevent bias toward the majority class."
    )

    # Tuning Grid Table
    p_t3 = doc.add_paragraph()
    p_t3.paragraph_format.space_before = Pt(6)
    p_t3.paragraph_format.space_after = Pt(4)
    r_t3 = p_t3.add_run("Table 4.1: Model Hyperparameter Search Grids and Optimal Configurations")
    r_t3.font.bold = True
    r_t3.font.size = Pt(10)
    r_t3.font.color.rgb = COLOR_PRIMARY

    t_grid = doc.add_table(rows=4, cols=4)
    style_table(t_grid, [Inches(1.5), Inches(2.2), Inches(1.8), Inches(1.0)])
    headers_grid = ["Algorithm", "Hyperparameter Search Space", "Optimal Discovered Parameters", "Best CV ROC-AUC"]
    for i, h in enumerate(headers_grid):
        t_grid.rows[0].cells[i].paragraphs[0].text = h
    style_table_header(t_grid.rows[0])

    data_grid = [
        ("Logistic Regression", "C: [0.01, 0.1, 1.0, 10.0]\nclass_weight: [None, 'balanced']\nsolver: ['lbfgs']", "C=10.0, class_weight=None, penalty='l2', max_iter=1000", "0.8484"),
        ("Random Forest Classifier", "n_estimators: [100, 200]\nmax_depth: [5, 10, None]\nmin_samples_split: [2, 5, 10]\nclass_weight: [None, 'balanced']", "n_estimators=200, max_depth=5, min_samples_split=5, class_weight='balanced'", "0.8462"),
        ("Gradient Boosting Classifier", "n_estimators: [100, 150]\nlearning_rate: [0.05, 0.1]\nmax_depth: [3, 5]\nsubsample: [0.8, 1.0]", "n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8", "0.8500"),
    ]
    for r_idx, row_data in enumerate(data_grid):
        row = t_grid.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            row.cells[c_idx].paragraphs[0].text = val
    style_table_rows(t_grid)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ==========================================
    # SECTION 5: EVALUATION & BUSINESS IMPACT
    # ==========================================
    h5 = doc.add_heading(level=1)
    r_h5 = h5.add_run("5. Multi-Metric Evaluation & Financial Cost-Benefit Analysis")
    r_h5.font.color.rgb = COLOR_PRIMARY
    r_h5.font.size = Pt(16)
    r_h5.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.add_run(
        "Each tuned pipeline was evaluated against the unobserved test dataset of 1,409 customers. Beyond traditional accuracy "
        "and AUC metrics, a business cost-benefit simulator was constructed to evaluate real-world campaign return on investment (ROI)."
    )

    # Test Results Comparison Table
    p_t4 = doc.add_paragraph()
    p_t4.paragraph_format.space_before = Pt(6)
    p_t4.paragraph_format.space_after = Pt(4)
    r_t4 = p_t4.add_run("Table 5.1: Comprehensive Test Performance & Financial ROI Comparison")
    r_t4.font.bold = True
    r_t4.font.size = Pt(10)
    r_t4.font.color.rgb = COLOR_PRIMARY

    t_eval = doc.add_table(rows=4, cols=8)
    style_table(t_eval, [Inches(1.4), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.8), Inches(0.8)])
    headers_eval = ["Model", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1-Score", "Max Net Profit"]
    for i, h in enumerate(headers_eval):
        t_eval.rows[0].cells[i].paragraphs[0].text = h
    style_table_header(t_eval.rows[0])

    data_eval = [
        ("Logistic Regression", "0.8454", "0.6552", "0.8048", "0.6656", "0.5321", "0.5914", "$47,350.00"),
        ("Gradient Boosting", "0.8474", "0.6680", "0.8070", "0.6809", "0.5134", "0.5854", "$46,750.00"),
        ("Random Forest", "0.8443", "0.6556", "0.7402", "0.5067", "0.8048", "0.6219", "$46,200.00"),
    ]
    for r_idx, row_data in enumerate(data_eval):
        row = t_eval.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            row.cells[c_idx].paragraphs[0].text = val
    style_table_rows(t_eval)

    # Embedded Visualization Chart
    chart_path = Path("reports/model_evaluation_visualizations.png")
    if chart_path.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(10)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(str(chart_path.resolve()), width=Inches(6.2))

        p_img_cap = doc.add_paragraph()
        p_img_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img_cap.paragraph_format.space_before = Pt(2)
        p_img_cap.paragraph_format.space_after = Pt(8)
        r_ic = p_img_cap.add_run("Figure 5.1: (Left) ROC Curves; (Center) Precision-Recall Curves; (Right) Net Financial Benefit ($) vs Decision Threshold")
        r_ic.font.size = Pt(8.5)
        r_ic.font.italic = True
        r_ic.font.color.rgb = COLOR_MUTED

    add_callout(
        doc,
        "Why Logistic Regression was selected over Gradient Boosting: At threshold 0.5, both models leave significant profit on the table. "
        "By finding the business-optimal threshold (0.16), Logistic Regression intercepts 339 of 374 actual churners (90.64% recall) with "
        "a total campaign cost of $37,400, gross value saved of $84,750, and a maximal Net Profit of $47,350 (Campaign ROI: 126.6%). "
        "Additionally, its smooth probability calibration (Brier Score: 0.1359) and 0.5ms inference latency make it the premier production candidate.",
        title="DECISION SELECTION RATIONALE"
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==========================================
    # SECTION 6: REST API DEPLOYMENT
    # ==========================================
    h6 = doc.add_heading(level=1)
    r_h6 = h6.add_run("6. REST API Architecture & Deployment")
    r_h6.font.color.rgb = COLOR_PRIMARY
    r_h6.font.size = Pt(16)
    r_h6.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.add_run(
        "The winning model pipeline was serialized with joblib into models/best_churn_pipeline.joblib and exposed through "
        "a production FastAPI service (src/api/app.py). Strict request/response contract validation is enforced using Pydantic v2."
    )

    # API Endpoints Table
    p_t5 = doc.add_paragraph()
    p_t5.paragraph_format.space_before = Pt(6)
    p_t5.paragraph_format.space_after = Pt(4)
    r_t5 = p_t5.add_run("Table 6.1: FastAPI REST Service Endpoints Specification")
    r_t5.font.bold = True
    r_t5.font.size = Pt(10)
    r_t5.font.color.rgb = COLOR_PRIMARY

    t_api = doc.add_table(rows=6, cols=4)
    style_table(t_api, [Inches(1.0), Inches(1.8), Inches(1.8), Inches(1.9)])
    headers_api = ["HTTP Verb", "Path", "Request / Response Schema", "Description & Operational Role"]
    for i, h in enumerate(headers_api):
        t_api.rows[0].cells[i].paragraphs[0].text = h
    style_table_header(t_api.rows[0])

    data_api = [
        ("GET", "/", "Dict[str, str]", "Welcome endpoint returning service status and documentation URLs."),
        ("GET", "/health", "HealthResponse", "Kubernetes/container liveness probe confirming model is loaded in memory."),
        ("GET", "/model-info", "ModelMetadataResponse", "Audit endpoint returning model architecture, CV scores, and business ROI."),
        ("POST", "/predict", "CustomerInput -> PredictionResponse", "Real-time single subscriber churn prediction with risk tier and actionable retention step."),
        ("POST", "/predict/batch", "BatchCustomerInput -> BatchPredictionResponse", "Vectorized bulk prediction processing hundreds of customer records concurrently."),
    ]
    for r_idx, row_data in enumerate(data_api):
        row = t_api.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            row.cells[c_idx].paragraphs[0].text = val
    style_table_rows(t_api)

    # Code Listing for Request/Response
    sample_json_req = """{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}"""
    add_code_snippet(doc, sample_json_req, caption="Sample JSON Payload for POST /predict")

    sample_json_resp = """{
  "customer_id": "7590-VHVEG",
  "churn_prediction": 1,
  "churn_label": "Churn",
  "churn_probability": 0.6043,
  "risk_level": "Medium",
  "recommended_action": "Caution: Moderate churn risk detected. Engage via personalized satisfaction check-in, feature adoption guides, or targeted loyalty incentive."
}"""
    add_code_snippet(doc, sample_json_resp, caption="Sample JSON Response from POST /predict")

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==========================================
    # SECTION 7: TESTING & VERIFICATION
    # ==========================================
    h7 = doc.add_heading(level=1)
    r_h7 = h7.add_run("7. Automated Testing, Quality Assurance & Verification")
    r_h7.font.color.rgb = COLOR_PRIMARY
    r_h7.font.size = Pt(16)
    r_h7.font.bold = True

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.add_run(
        "A robust test harness was implemented using pytest and FastAPI's TestClient. "
        "The suite consists of 16 automated tests executed across 3 functional modules with a 100% pass rate:"
    )

    test_bullets = [
        ("tests/test_data_pipeline.py (5 tests): ", "Validates raw data loading, missing value handling (TotalCharges whitespace coercion), stratified split preservation, custom feature engineer calculations, and NaN-free matrix output."),
        ("tests/test_models.py (4 tests): ", "Validates classifier candidates, hyperparameter grid schemas, financial cost-benefit equations, threshold optimization sweeps, and artifact serialization integrity."),
        ("tests/test_api.py (7 tests): ", "Validates root, health, model-info endpoints, valid single predictions, risk tier stratification (high-risk vs loyal), batch processing, and Pydantic 422 HTTP validation errors on malformed payloads."),
    ]
    for prefix, desc in test_bullets:
        p_tb = doc.add_paragraph(style="List Bullet")
        r_pre = p_tb.add_run(prefix)
        r_pre.font.bold = True
        p_tb.add_run(desc)

    p_test_res = doc.add_paragraph()
    p_test_res.paragraph_format.space_before = Pt(4)
    p_test_res.paragraph_format.space_after = Pt(2)
    p_test_res.add_run("Pytest Execution Output: ").font.bold = True
    add_code_snippet(
        doc,
        "======================== 16 passed, 1 warning in 5.25s ========================\n"
        "tests/test_api.py::test_root_endpoint PASSED\n"
        "tests/test_api.py::test_health_endpoint PASSED\n"
        "tests/test_api.py::test_model_info_endpoint PASSED\n"
        "tests/test_api.py::test_predict_single_customer_valid PASSED\n"
        "tests/test_api.py::test_predict_risk_stratification PASSED\n"
        "tests/test_api.py::test_predict_batch_endpoint PASSED\n"
        "tests/test_api.py::test_predict_validation_error PASSED\n"
        "tests/test_data_pipeline.py::test_load_raw_data PASSED\n"
        "tests/test_data_pipeline.py::test_clean_raw_data PASSED\n"
        "tests/test_data_pipeline.py::test_split_data PASSED\n"
        "tests/test_data_pipeline.py::test_churn_feature_engineer PASSED\n"
        "tests/test_data_pipeline.py::test_full_preprocessing_pipeline PASSED\n"
        "tests/test_models.py::test_candidate_models_structure PASSED\n"
        "tests/test_models.py::test_compute_business_impact PASSED\n"
        "tests/test_models.py::test_find_optimal_business_threshold PASSED\n"
        "tests/test_models.py::test_saved_model_artifact PASSED",
        caption="Pytest Automated Test Results (100% Pass Rate)"
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==========================================
    # SECTION 8: REPOSITORY & RUN INSTRUCTIONS
    # ==========================================
    h8 = doc.add_heading(level=1)
    r_h8 = h8.add_run("8. Repository Structure & Execution Instructions")
    r_h8.font.color.rgb = COLOR_PRIMARY
    r_h8.font.size = Pt(16)
    r_h8.font.bold = True

    repo_tree = """churn-prediction-pipeline/
├── data/
│   ├── raw/Telco-Customer-Churn.csv       # Raw input dataset
│   └── processed/                         # Cache directory
├── models/
│   ├── best_churn_pipeline.joblib         # Serialized Scikit-learn Pipeline
│   └── model_metadata.json                # Complete metrics & hyperparams
├── reports/
│   ├── model_comparison.csv               # Comparison CSV table
│   ├── model_comparison.json              # Detailed metrics JSON
│   └── model_evaluation_visualizations.png# High-resolution ROC/PR/ROI plots
├── src/
│   ├── config.py                          # Constants, paths & param grids
│   ├── data_loader.py                     # Data loading & cleaning
│   ├── preprocessor.py                    # ChurnFeatureEngineer & ColumnTransformer
│   ├── train.py                           # 5-fold CV hyperparameter tuning
│   ├── evaluate.py                        # ROI simulation & curve plotting
│   └── api/
│       ├── app.py                         # Production FastAPI REST application
│       └── schemas.py                     # Pydantic v2 schemas
├── tests/
│   ├── test_data_pipeline.py              # Data cleaning & transformer tests
│   ├── test_models.py                     # Training & financial logic tests
│   └── test_api.py                        # FastAPI integration tests
├── Dockerfile                             # Container deployment specification
├── requirements.txt                       # Project dependencies
├── run_pipeline.py                        # End-to-end execution script
└── verify_api_live.py                     # Live API demonstration runner"""

    add_code_snippet(doc, repo_tree, caption="Directory Layout & Deliverables")

    p_run = doc.add_paragraph()
    p_run.paragraph_format.space_before = Pt(6)
    p_run.paragraph_format.space_after = Pt(2)
    p_run.add_run("Step-by-Step Commands to Run Locally:").font.bold = True

    run_commands = """# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute full training, tuning, evaluation & model serialization
python run_pipeline.py

# 3. Execute automated test suite
python -m pytest -v

# 4. Launch FastAPI REST server
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

# 5. Execute live API demonstration script
python verify_api_live.py"""

    add_code_snippet(doc, run_commands, caption="Terminal Execution Commands")

    # ==========================================
    # SECTION 9: CONCLUSION
    # ==========================================
    h9 = doc.add_heading(level=1)
    r_h9 = h9.add_run("9. Conclusion & Business Impact")
    r_h9.font.color.rgb = COLOR_PRIMARY
    r_h9.font.size = Pt(16)
    r_h9.font.bold = True

    p_conc = doc.add_paragraph()
    p_conc.paragraph_format.space_after = Pt(6)
    p_conc.paragraph_format.line_spacing = 1.15
    p_conc.add_run(
        "This project successfully delivers a complete, production-grade supervised machine learning pipeline for customer churn prediction. "
        "By merging rigorous machine learning practices (leak-free transformers, stratified 5-fold CV, multi-metric evaluation) with real-world "
        "financial decision optimization and modern API engineering (FastAPI, Pydantic v2, Docker), the resulting system provides actionable, "
        "high-ROI predictions capable of directly safeguarding subscriber revenue in production environments."
    )

    output_path = Path("Customer_Churn_Prediction_Pipeline_Assignment.docx")
    doc.save(output_path)
    print(f"Document successfully saved to {output_path.resolve()}!")


if __name__ == "__main__":
    build_word_document()
