import os
import base64
import datetime

REPORTS_DIR = "reports"
PLOTS_DIR = "reports/plots"
os.makedirs(REPORTS_DIR, exist_ok=True)


def encode_image(image_path):
    if image_path is None:
        return None
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_header(best_tuned_name, problem_type, timestamp):
    return f"""
    <div class="header">
        <h1>AutoML Analysis Report</h1>
        <p>Generated on {datetime.datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</p>
        <div class="header-pills">
            <span class="pill">Task: {problem_type.title()}</span>
            <span class="pill">Best Model: {best_tuned_name}</span>
        </div>
    </div>
    """


def build_summary_banner(quality_report, eval_metrics,
                         best_tuned_name, problem_type):
    """
    Top summary — 3 key numbers a non-ML person cares about.
    """
    quality_score = quality_report["quality_score"]
    quality_label = quality_report["quality_label"]

    if problem_type == 'classification':
        perf_value = f"{eval_metrics.get('accuracy', 0) * 100:.1f}%"
        perf_label = "Model Accuracy"
        perf_desc = "Out of every 100 predictions, this many are correct."
    else:
        perf_value = f"{eval_metrics.get('r2', 0):.2f}"
        perf_label = "R² Score"
        perf_desc = "How well the model explains the data (1.0 = perfect)."

    if quality_score >= 80:
        q_color = "#2ecc71"
    elif quality_score >= 60:
        q_color = "#f39c12"
    else:
        q_color = "#e74c3c"

    return f"""
    <div class="section">
        <h2>Quick Summary</h2>
        <p class="section-desc">
            Here are the three most important things to know
            about your data and model.
        </p>
        <div class="banner-grid">
            <div class="banner-card">
                <div class="banner-icon">📊</div>
                <div class="banner-value" style="color:{q_color}">
                    {quality_score}/100
                </div>
                <div class="banner-label">Data Quality</div>
                <div class="banner-desc">
                    {quality_label} quality.
                    Higher score means cleaner,
                    more reliable data.
                </div>
            </div>
            <div class="banner-card">
                <div class="banner-icon">🤖</div>
                <div class="banner-value" style="color:#3498db">
                    {best_tuned_name}
                </div>
                <div class="banner-label">Best Model Found</div>
                <div class="banner-desc">
                    This algorithm performed best
                    on your dataset after testing
                    7 different approaches.
                </div>
            </div>
            <div class="banner-card">
                <div class="banner-icon">🎯</div>
                <div class="banner-value" style="color:#2ecc71">
                    {perf_value}
                </div>
                <div class="banner-label">{perf_label}</div>
                <div class="banner-desc">{perf_desc}</div>
            </div>
        </div>
    </div>
    """


def build_quality_section(quality_report):
    score = quality_report["quality_score"]
    label = quality_report["quality_label"]

    if score >= 80:
        color = "#2ecc71"
        advice = "Your data is in good shape. The model should learn reliably from it."
    elif score >= 60:
        color = "#f39c12"
        advice = "Your data has some issues. The model will still work but fixing these issues would improve results."
    else:
        color = "#e74c3c"
        advice = "Your data has serious issues. Consider cleaning it before trusting the model results."

    warnings_html = ""
    for w in quality_report["warnings"]:
        if "CRITICAL" in w:
            tag = "critical"
            icon = "🔴"
        elif "WARNING" in w:
            tag = "warning"
            icon = "🟡"
        else:
            tag = "info"
            icon = "🔵"
        clean = w.replace("CRITICAL: ", "").replace(
            "WARNING: ", "").replace("INFO: ", "")
        warnings_html += f"""
        <div class="warn-{tag}">
            {icon} {clean}
        </div>
        """

    return f"""
    <div class="section">
        <h2>📋 Step 1 — How Good is Your Data?</h2>
        <p class="section-desc">
            Before training any model, we check the quality
            of your data. Good data = better predictions.
        </p>
        <div class="score-row">
            <div class="score-box" style="border-color:{color}">
                <span class="score-number" style="color:{color}">
                    {score}/100
                </span>
                <span class="score-label">{label}</span>
            </div>
            <div class="score-advice">
                <p>{advice}</p>
            </div>
        </div>
        <h3 style="margin-top:20px; margin-bottom:10px;">
            What we found in your data:
        </h3>
        <div class="warnings-box">
            {warnings_html}
        </div>
    </div>
    """


def build_eda_section(eda_stats, eda_plot_paths):
    dist = eda_stats.get('class_distribution', {})
    dist_html = ""
    total = eda_stats['total_rows']
    for cls, count in dist.items():
        pct = round(count / total * 100, 1)
        dist_html += f"""
        <div class="dist-bar">
            <span class="dist-label">Class {cls}</span>
            <div class="dist-track">
                <div class="dist-fill"
                     style="width:{pct}%"></div>
            </div>
            <span class="dist-pct">
                {count} rows ({pct}%)
            </span>
        </div>
        """

    plots_html = ""
    plot_labels = {
        'target': ('Class Distribution',
                   'How many samples belong to each category.'),
        'missing': ('Missing Values',
                    'Columns with empty/missing data.'),
        'distributions': ('Feature Distributions',
                          'How values are spread across each column.'),
        'correlation': ('Feature Correlation',
                        'Which features move together.'),
        'outliers': ('Outlier Detection',
                     'Unusual values that differ from the rest.')
    }

    for key, path in eda_plot_paths.items():
        if path:
            img = encode_image(path)
            if img:
                label, desc = plot_labels.get(
                    key, (key, ''))
                plots_html += f"""
                <div class="plot-card">
                    <div class="plot-title">{label}</div>
                    <div class="plot-desc">{desc}</div>
                    <img src="{img}" alt="{label}">
                </div>
                """

    return f"""
    <div class="section">
        <h2>🔍 Step 2 — Understanding Your Data</h2>
        <p class="section-desc">
            A closer look at what's inside your dataset
            before any model training begins.
        </p>

        <div class="info-grid">
            <div class="info-card">
                <div class="info-value">
                    {eda_stats['total_rows']:,}
                </div>
                <div class="info-label">Total Rows</div>
            </div>
            <div class="info-card">
                <div class="info-value">
                    {eda_stats['total_columns']}
                </div>
                <div class="info-label">Total Columns</div>
            </div>
            <div class="info-card">
                <div class="info-value">
                    {eda_stats['missing_percentage']}%
                </div>
                <div class="info-label">Missing Data</div>
            </div>
            <div class="info-card">
                <div class="info-value">
                    {eda_stats['duplicate_rows']}
                </div>
                <div class="info-label">Duplicate Rows</div>
            </div>
        </div>

        <h3 style="margin:20px 0 10px">
            How your target column is distributed:
        </h3>
        <div class="dist-container">
            {dist_html}
        </div>

        <h3 style="margin:20px 0 10px">Charts:</h3>
        <div class="plots-grid">
            {plots_html}
        </div>
    </div>
    """


def build_model_section(leaderboard, best_tuned_name,
                        tuned_results, eval_metrics,
                        eval_plot_paths, problem_type):
    # Model comparison
    model_cards = ""
    medals = ["🥇", "🥈", "🥉"]
    for i, entry in enumerate(leaderboard[:7]):
        medal = medals[i] if i < 3 else f"{i+1}."
        is_best = entry['model'] == leaderboard[0]['model']
        border = (
            "border: 2px solid #2ecc71;"
            if is_best else ""
        )
        pct = round(entry['mean_score'] * 100, 1)
        model_cards += f"""
        <div class="model-card" style="{border}">
            <div class="model-rank">{medal}</div>
            <div class="model-name">{entry['model']}</div>
            <div class="model-bar-track">
                <div class="model-bar-fill"
                     style="width:{pct}%">
                </div>
            </div>
            <div class="model-score">{pct}%</div>
        </div>
        """

    # Metrics in plain English
    if problem_type == 'classification':
        metrics_html = f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">
                    {eval_metrics.get('accuracy',0)*100:.1f}%
                </div>
                <div class="metric-name">Accuracy</div>
                <div class="metric-explain">
                    Out of every 100 predictions,
                    this many were correct.
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">🔍</div>
                <div class="metric-value">
                    {eval_metrics.get('precision',0)*100:.1f}%
                </div>
                <div class="metric-name">Precision</div>
                <div class="metric-explain">
                    When the model says "yes",
                    how often is it actually right?
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📡</div>
                <div class="metric-value">
                    {eval_metrics.get('recall',0)*100:.1f}%
                </div>
                <div class="metric-name">Recall</div>
                <div class="metric-explain">
                    Out of all actual "yes" cases,
                    how many did the model catch?
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">⚖️</div>
                <div class="metric-value">
                    {eval_metrics.get('f1',0)*100:.1f}%
                </div>
                <div class="metric-name">F1 Score</div>
                <div class="metric-explain">
                    Balance between precision and recall.
                    Higher is better.
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-value">
                    {eval_metrics.get('roc_auc',0)*100:.1f}%
                </div>
                <div class="metric-name">Overall Ability</div>
                <div class="metric-explain">
                    How well the model separates
                    the two groups. 50% = random guess,
                    100% = perfect.
                </div>
            </div>
        </div>
        """
    else:
        metrics_html = f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">
                    {eval_metrics.get('r2',0):.2f}
                </div>
                <div class="metric-name">R² Score</div>
                <div class="metric-explain">
                    How much of the variation in your
                    target the model explains.
                    1.0 = perfect.
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📏</div>
                <div class="metric-value">
                    {eval_metrics.get('rmse',0):.2f}
                </div>
                <div class="metric-name">Average Error</div>
                <div class="metric-explain">
                    On average, predictions are off
                    by this much. Lower is better.
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-value">
                    {eval_metrics.get('mae',0):.2f}
                </div>
                <div class="metric-name">Mean Error</div>
                <div class="metric-explain">
                    Simple average of how far off
                    predictions are.
                </div>
            </div>
        </div>
        """

    # Eval plots
    eval_plot_labels = {
        'confusion_matrix': (
            'Where the Model Made Mistakes',
            'Shows correct and incorrect predictions broken down by category.'
        ),
        'roc_curve': (
            'Model Confidence Curve',
            'Shows how well the model separates the two groups at different thresholds.'
        ),
        'pred_vs_actual': (
            'Predicted vs Actual',
            'Each dot is one prediction. Dots on the diagonal line = perfect predictions.'
        )
    }

    eval_plots_html = ""
    for key, path in eval_plot_paths.items():
        img = encode_image(path)
        if img:
            label, desc = eval_plot_labels.get(
                key, (key, ''))
            eval_plots_html += f"""
            <div class="plot-card">
                <div class="plot-title">{label}</div>
                <div class="plot-desc">{desc}</div>
                <img src="{img}" alt="{label}">
            </div>
            """

    return f"""
    <div class="section">
        <h2>🤖 Step 3 — Which Model Performed Best?</h2>
        <p class="section-desc">
            We tested 7 different machine learning algorithms
            on your data. Here is how they ranked:
        </p>
        <div class="model-list">
            {model_cards}
        </div>
        <div class="winner-box">
            ✅ Winner: <b>{best_tuned_name}</b> —
            fine-tuned to squeeze out the best possible
            performance on your specific dataset.
        </div>
    </div>

    <div class="section">
        <h2>📊 Step 4 — How Good Are the Predictions?</h2>
        <p class="section-desc">
            These numbers tell you how well the winning model
            performs on data it has never seen before.
        </p>
        {metrics_html}
        <div class="plots-grid" style="margin-top:20px">
            {eval_plots_html}
        </div>
    </div>
    """


def build_feature_section(importance_results,
                          fi_plot_paths, summary):
    fi_plot_labels = {
        'tree_importance': (
            'What the Model Thinks Matters',
            'Features the winning model relied on most during training.'
        ),
        'rf_importance': (
            'Confirmed Important Features',
            'Average importance across 100 decision trees — more reliable than a single model.'
        ),
        'permutation_importance': (
            'What Actually Drives Predictions',
            'We scrambled each feature and measured how much accuracy dropped. Bigger drop = more important.'
        )
    }

    plots_html = ""
    for key, path in fi_plot_paths.items():
        img = encode_image(path)
        if img:
            label, desc = fi_plot_labels.get(key, (key, ''))
            plots_html += f"""
            <div class="plot-card">
                <div class="plot-title">{label}</div>
                <div class="plot-desc">{desc}</div>
                <img src="{img}" alt="{label}">
            </div>
            """

    return f"""
    <div class="section">
        <h2>💡 Step 5 — What Drives the Predictions?</h2>
        <p class="section-desc">
            Understanding which factors matter most helps
            you trust and act on the model's predictions.
        </p>
        <div class="insight-box">
            <div class="insight-icon">💬</div>
            <div class="insight-text">{summary}</div>
        </div>
        <div class="plots-grid" style="margin-top:20px">
            {plots_html}
        </div>
    </div>
    """


def build_recommendation_section(best_tuned_name,
                                  tuned_results,
                                  eval_metrics,
                                  summary,
                                  problem_type):
    best_params = tuned_results[best_tuned_name]["best_params"]
    best_score = tuned_results[best_tuned_name]["best_score"]

    params_html = ""
    for param, value in best_params.items():
        params_html += f"""
        <div class="param-row">
            <span class="param-name">{param}</span>
            <span class="param-value">{value}</span>
        </div>
        """

    if problem_type == 'classification':
        headline_metric = (
            f"{eval_metrics.get('accuracy', 0)*100:.1f}% accuracy "
            f"on unseen data"
        )
    else:
        headline_metric = (
            f"R² of {eval_metrics.get('r2', 0):.2f} "
            f"on unseen data"
        )

    return f"""
    <div class="section recommendation-section">
        <h2>✅ Final Recommendation</h2>
        <p class="section-desc">
            Here is everything you need to know about
            the model we recommend for your dataset.
        </p>
        <div class="rec-card">
            <div class="rec-header">
                <span class="rec-trophy">🏆</span>
                <span class="rec-model">{best_tuned_name}</span>
            </div>
            <div class="rec-headline">
                Achieved <b>{headline_metric}</b>
            </div>

            <div class="rec-block">
                <h4>What this model learned:</h4>
                <p>{summary}</p>
            </div>

            <div class="rec-block">
                <h4>Technical settings used:</h4>
                <div class="params-list">
                    {params_html}
                </div>
            </div>

            <div class="rec-block">
                <h4>What you can do next:</h4>
                <ul class="next-steps">
                    <li>Use this model to predict outcomes
                        on new data.</li>
                    <li>If results seem off, collect more
                        data and retrain.</li>
                    <li>The feature importance charts above
                        tell you which columns matter most —
                        focus on improving data quality there
                        first.</li>
                </ul>
            </div>
        </div>
    </div>
    """


def get_css():
    return """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Segoe UI', Arial, sans-serif;
        background: #f0f2f5;
        color: #2c3e50;
        padding: 24px;
        max-width: 1100px;
        margin: 0 auto;
    }
    .header {
        background: linear-gradient(135deg, #1a252f, #2980b9);
        color: white;
        padding: 36px;
        border-radius: 14px;
        margin-bottom: 24px;
        text-align: center;
    }
    .header h1 {
        font-size: 2em;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    .header p {
        opacity: 0.8;
        font-size: 0.9em;
        margin-bottom: 14px;
    }
    .header-pills {
        display: flex;
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
    }
    .pill {
        background: rgba(255,255,255,0.18);
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 0.88em;
    }
    .section {
        background: white;
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }
    .section h2 {
        font-size: 1.25em;
        color: #1a252f;
        margin-bottom: 8px;
        padding-bottom: 10px;
        border-bottom: 2px solid #eaecef;
    }
    .section-desc {
        color: #7f8c8d;
        font-size: 0.92em;
        margin-bottom: 18px;
        line-height: 1.6;
    }
    .banner-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .banner-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #eaecef;
    }
    .banner-icon { font-size: 1.8em; margin-bottom: 8px; }
    .banner-value {
        font-size: 1.6em;
        font-weight: bold;
        margin-bottom: 4px;
    }
    .banner-label {
        font-size: 0.82em;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .banner-desc {
        font-size: 0.82em;
        color: #7f8c8d;
        line-height: 1.5;
    }
    .score-row {
        display: flex;
        align-items: center;
        gap: 24px;
        margin: 16px 0;
        flex-wrap: wrap;
    }
    .score-box {
        border: 3px solid;
        border-radius: 12px;
        padding: 16px 32px;
        text-align: center;
        min-width: 140px;
    }
    .score-number {
        display: block;
        font-size: 2.2em;
        font-weight: bold;
    }
    .score-label {
        font-size: 1em;
        color: #7f8c8d;
    }
    .score-advice {
        flex: 1;
        color: #555;
        font-size: 0.95em;
        line-height: 1.7;
    }
    .warn-critical {
        background: #fdecea;
        border-left: 4px solid #e74c3c;
        padding: 9px 14px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.89em;
        line-height: 1.5;
    }
    .warn-warning {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 9px 14px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.89em;
        line-height: 1.5;
    }
    .warn-info {
        background: #eaf4fb;
        border-left: 4px solid #3498db;
        padding: 9px 14px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.89em;
        line-height: 1.5;
    }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .info-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #eaecef;
    }
    .info-value {
        font-size: 1.6em;
        font-weight: bold;
        color: #2980b9;
    }
    .info-label {
        font-size: 0.78em;
        color: #7f8c8d;
        margin-top: 4px;
        text-transform: uppercase;
    }
    .dist-container { margin: 10px 0; }
    .dist-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 8px 0;
        font-size: 0.88em;
    }
    .dist-label {
        width: 70px;
        font-weight: bold;
        color: #2c3e50;
    }
    .dist-track {
        flex: 1;
        background: #ecf0f1;
        border-radius: 6px;
        height: 18px;
        overflow: hidden;
    }
    .dist-fill {
        height: 100%;
        background: linear-gradient(90deg, #3498db, #2980b9);
        border-radius: 6px;
    }
    .dist-pct { width: 130px; color: #7f8c8d; }
    .plots-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 18px;
    }
    .plot-card {
        border: 1px solid #eaecef;
        border-radius: 10px;
        overflow: hidden;
        background: #fafafa;
    }
    .plot-title {
        font-weight: bold;
        font-size: 0.9em;
        padding: 10px 14px 4px;
        color: #2c3e50;
    }
    .plot-desc {
        font-size: 0.8em;
        color: #7f8c8d;
        padding: 0 14px 10px;
        line-height: 1.4;
    }
    .plot-card img {
        width: 100%;
        display: block;
    }
    .model-list { margin: 16px 0; }
    .model-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        border-radius: 8px;
        margin: 6px 0;
        background: #f8f9fa;
    }
    .model-rank { font-size: 1.2em; width: 30px; }
    .model-name {
        width: 180px;
        font-size: 0.9em;
        font-weight: 500;
    }
    .model-bar-track {
        flex: 1;
        background: #ecf0f1;
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
    }
    .model-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #3498db, #1abc9c);
        border-radius: 6px;
    }
    .model-score {
        width: 50px;
        text-align: right;
        font-size: 0.88em;
        font-weight: bold;
        color: #2c3e50;
    }
    .winner-box {
        background: #eafaf1;
        border: 1px solid #2ecc71;
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 16px;
        font-size: 0.92em;
        color: #1e8449;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 14px;
        margin: 16px 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        border: 1px solid #eaecef;
    }
    .metric-icon { font-size: 1.5em; margin-bottom: 6px; }
    .metric-value {
        font-size: 1.6em;
        font-weight: bold;
        color: #2980b9;
        margin-bottom: 4px;
    }
    .metric-name {
        font-size: 0.82em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 6px;
    }
    .metric-explain {
        font-size: 0.78em;
        color: #7f8c8d;
        line-height: 1.4;
    }
    .insight-box {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 16px 20px;
        display: flex;
        gap: 14px;
        align-items: flex-start;
    }
    .insight-icon { font-size: 1.5em; }
    .insight-text {
        font-size: 0.95em;
        line-height: 1.7;
        color: #2c3e50;
    }
    .recommendation-section {
        border-top: 4px solid #2ecc71;
    }
    .rec-card {
        background: #f8fffe;
        border: 1px solid #d5f5e3;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
    }
    .rec-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .rec-trophy { font-size: 2em; }
    .rec-model {
        font-size: 1.4em;
        font-weight: bold;
        color: #1e8449;
    }
    .rec-headline {
        font-size: 1em;
        color: #555;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid #d5f5e3;
    }
    .rec-block {
        margin-bottom: 18px;
    }
    .rec-block h4 {
        font-size: 0.9em;
        color: #1e8449;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .rec-block p {
        font-size: 0.92em;
        color: #555;
        line-height: 1.7;
    }
    .params-list { margin-top: 6px; }
    .param-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #eafaf1;
        font-size: 0.88em;
    }
    .param-name { color: #7f8c8d; }
    .param-value { font-weight: bold; color: #2c3e50; }
    .next-steps {
        padding-left: 18px;
        font-size: 0.9em;
        color: #555;
        line-height: 1.9;
    }
    .footer {
        text-align: center;
        color: #aab;
        font-size: 0.82em;
        margin-top: 20px;
        padding: 10px;
    }
    """


def generate_report(quality_report, eda_stats,
                    eda_plot_paths, leaderboard,
                    cv_results, best_tuned_name,
                    tuned_results, eval_metrics,
                    eval_plot_paths, importance_results,
                    fi_plot_paths, summary, problem_type):
    print("\n" + "="*50)
    print("GENERATING HTML REPORT")
    print("="*50)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{timestamp}.html"
    report_path = os.path.join(REPORTS_DIR, report_filename)

    header = build_header(best_tuned_name,
                          problem_type, timestamp)
    summary_banner = build_summary_banner(
        quality_report, eval_metrics,
        best_tuned_name, problem_type)
    quality_section = build_quality_section(quality_report)
    eda_section = build_eda_section(eda_stats, eda_plot_paths)
    model_section = build_model_section(
        leaderboard, best_tuned_name, tuned_results,
        eval_metrics, eval_plot_paths, problem_type)
    fi_section = build_feature_section(
        importance_results, fi_plot_paths, summary)
    rec_section = build_recommendation_section(
        best_tuned_name, tuned_results,
        eval_metrics, summary, problem_type)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">
    <title>AutoML Report</title>
    <style>{get_css()}</style>
</head>
<body>
    {header}
    {summary_banner}
    {quality_section}
    {eda_section}
    {model_section}
    {fi_section}
    {rec_section}
    <div class="footer">
        Generated by AutoML Pipeline Builder
    </div>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReport saved: {report_path}")
    print("Open this file in your browser to view.")
    print("\n" + "="*50)
    print("REPORT GENERATION COMPLETE")
    print("="*50)

    return report_path